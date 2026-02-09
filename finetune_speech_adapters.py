from datetime import timedelta
import torch.distributed as dist
import json
import os
from pathlib import Path
import yaml
import torch
import sacrebleu
import wandb
from accelerate import Accelerator
from accelerate.utils import gather_object
from datasets import load_from_disk, concatenate_datasets
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    BatchFeature,
    Trainer,
    TrainingArguments,
    StoppingCriteria,
    StoppingCriteriaList,
    TrainerCallback,
)
from rouge_score import rouge_scorer
from sklearn.metrics import accuracy_score
import editdistance
import io
import soundfile as sf
from peft import PeftModel

ANSWER_SUFFIX = "<|end|><|endoftext|>"
_IGNORE_INDEX = -100

EVAL_STEPS = 1500

class MultipleTokenBatchStoppingCriteria(StoppingCriteria):
    def __init__(self, stop_tokens: torch.LongTensor, batch_size: int = 1) -> None:
        self.stop_tokens = stop_tokens
        self.max_stop_tokens = stop_tokens.shape[-1]
        self.stop_tokens_idx = torch.zeros(batch_size, dtype=torch.long, device=stop_tokens.device)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        generated_inputs = torch.eq(input_ids[:, -self.max_stop_tokens :].unsqueeze(1), self.stop_tokens)
        equal_generated_inputs = torch.all(generated_inputs, dim=2)
        sequence_idx = torch.any(equal_generated_inputs, dim=1)
        sequence_set_mask = self.stop_tokens_idx == 0
        self.stop_tokens_idx[sequence_idx & sequence_set_mask] = input_ids.shape[-1]
        return torch.all(self.stop_tokens_idx)


class MeralionDataset(Dataset):
    def __init__(self, processor, data_dir, split, task_types, rank=0, world_size=1):
        self.processor = processor
        self.training = split == "train"
        self.task_types = task_types if isinstance(task_types, list) else [task_types]
        
        all_datasets = []
        for task_type in self.task_types:
            task_folders = self._get_task_folders(data_dir, split, task_type)
            datasets = [load_from_disk(folder) for folder in task_folders]
            task_data = concatenate_datasets(datasets) if len(datasets) > 1 else datasets[0]
            all_datasets.append(task_data)
        
        self.data = concatenate_datasets(all_datasets) if len(all_datasets) > 1 else all_datasets[0]
        
        if world_size > 1:
            self.data = self.data.shard(world_size, rank)

    def _get_task_folders(self, data_dir, split, task_type):
        split_dir = "train" if split == "train" else "test"
        base_path = Path(data_dir) / split_dir
        
        folders = []
        for folder in os.listdir(base_path):
            if folder.startswith(task_type):
                folders.append(str(base_path / folder))
        
        return sorted(folders)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        
        user_message = {
            'role': 'user',
            'content': '<|audio_1|>\n' + data['instruction'],
        }
        prompt = self.processor.tokenizer.apply_chat_template(
            [user_message], tokenize=False, add_generation_prompt=True
        )
        
        audio_bytes = data['context']['bytes']
        audio_array, sampling_rate = sf.read(io.BytesIO(audio_bytes))
        
        inputs = self.processor(
            text=prompt, 
            audios=[(audio_array, sampling_rate)], 
            return_tensors='pt'
        )
        
        answer = f"{data['answer']}{ANSWER_SUFFIX}"
        answer_ids = self.processor.tokenizer(answer, return_tensors='pt').input_ids
        
        if self.training:
            input_ids = torch.cat([inputs.input_ids, answer_ids], dim=1)
            labels = torch.full_like(input_ids, _IGNORE_INDEX)
            labels[:, -answer_ids.shape[1] :] = answer_ids
        else:
            input_ids = inputs.input_ids
            labels = answer_ids

        return {
            'input_ids': input_ids,
            'labels': labels,
            'input_audio_embeds': inputs.input_audio_embeds,
            'audio_embed_sizes': inputs.audio_embed_sizes,
        }


def pad_sequence(sequences, padding_side='right', padding_value=0):
    assert padding_side in ['right', 'left']
    max_size = sequences[0].size()
    trailing_dims = max_size[1:]
    max_len = max(len(seq) for seq in sequences)
    batch_size = len(sequences)
    output = sequences[0].new_full((batch_size, max_len) + trailing_dims, padding_value)
    for i, seq in enumerate(sequences):
        length = seq.size(0)
        if padding_side == 'right':
            output.data[i, :length] = seq
        else:
            output.data[i, -length:] = seq
    return output


def cat_with_pad(tensors, dim, padding_value=0):
    ndim = tensors[0].dim()
    assert all(t.dim() == ndim for t in tensors[1:]), 'All tensors must have the same number of dimensions'
    
    out_size = [max(t.shape[i] for t in tensors) for i in range(ndim)]
    out_size[dim] = sum(t.shape[dim] for t in tensors)
    output = tensors[0].new_full(out_size, padding_value)
    
    index = 0
    for t in tensors:
        slices = [slice(0, t.shape[d]) for d in range(ndim)]
        slices[dim] = slice(index, index + t.shape[dim])
        output[slices] = t
        index += t.shape[dim]
    
    return output


def collate_fn(batch):
    input_ids_list = []
    labels_list = []
    input_audio_embeds_list = []
    audio_embed_sizes_list = []
    audio_attention_mask_list = []
    
    for inputs in batch:
        input_ids_list.append(inputs['input_ids'][0])
        labels_list.append(inputs['labels'][0])
        input_audio_embeds_list.append(inputs['input_audio_embeds'])
        audio_embed_sizes_list.append(inputs['audio_embed_sizes'])
        audio_attention_mask_list.append(
            inputs['input_audio_embeds'].new_full((inputs['input_audio_embeds'].size(1),), True, dtype=torch.bool)
        )

    input_ids = pad_sequence(input_ids_list, padding_side='left', padding_value=0)
    labels = pad_sequence(labels_list, padding_side='left', padding_value=0)
    audio_attention_mask = (
        pad_sequence(audio_attention_mask_list, padding_side='right', padding_value=False)
        if len(audio_attention_mask_list) > 1
        else None
    )
    attention_mask = (input_ids != 0).long()
    input_audio_embeds = cat_with_pad(input_audio_embeds_list, dim=0)
    audio_embed_sizes = torch.cat(audio_embed_sizes_list)

    return BatchFeature(
        {
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': attention_mask,
            'input_audio_embeds': input_audio_embeds,
            'audio_embed_sizes': audio_embed_sizes,
            'audio_attention_mask': audio_attention_mask,
            'input_mode': 2,
        }
    )


def create_model(model_name_or_path, speech_lora_path, use_flash_attention=False):
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.bfloat16 if use_flash_attention else torch.float32,
        _attn_implementation='flash_attention_2' if use_flash_attention else 'sdpa',
        trust_remote_code=True,
    ).to('cuda')
    
    adapter_path = Path(speech_lora_path)
    if not (adapter_path / 'adapter_config.json').exists():
        if (adapter_path / 'speech' / 'adapter_config.json').exists():
            speech_lora_path = str(adapter_path / 'speech')
        else:
            raise ValueError(f"Cannot find adapter_config.json in {speech_lora_path} or {speech_lora_path}/speech")

    model = PeftModel.from_pretrained(
        model,
        speech_lora_path,
        adapter_name="speech"
    )
    
    model.set_adapter("speech")
    
    return model


def compute_metrics(predictions, references, task_type):
    metrics = {}
    
    bleu = sacrebleu.corpus_bleu(predictions, [references])
    metrics['bleu'] = bleu.score
    
    if task_type == 'ASR':
        total_chars = sum(len(ref) for ref in references)
        total_distance = sum(editdistance.eval(pred, ref) for pred, ref in zip(predictions, references))
        wer = (total_distance / total_chars) * 100
        metrics['wer'] = wer
        metrics['cer'] = wer
    
    elif task_type in ['SQA', 'PQA']:
        exact_matches = sum(1 for pred, ref in zip(predictions, references) if pred.strip().lower() == ref.strip().lower())
        metrics['exact_match'] = (exact_matches / len(predictions)) * 100
    
    elif task_type == 'SDS':
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        rouge_scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}
        for pred, ref in zip(predictions, references):
            scores = scorer.score(ref, pred)
            for key in rouge_scores:
                rouge_scores[key].append(scores[key].fmeasure)
        
        for key in rouge_scores:
            metrics[key] = sum(rouge_scores[key]) / len(rouge_scores[key]) * 100
    
    return metrics


@torch.no_grad()
def evaluate(model, processor, eval_dataset, task_name, save_path=None, disable_tqdm=False, eval_batch_size=64):
    rank = int(os.environ.get('RANK', 0))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))

    model.eval()
    all_generated_texts = []
    all_labels = []

    eval_dataloader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=eval_batch_size,
        collate_fn=collate_fn,
        shuffle=False,
        drop_last=False,
        num_workers=8,
        prefetch_factor=2,
        pin_memory=True,
    )
    
    stop_tokens = ["<|end|>", processor.tokenizer.eos_token]
    stop_tokens_ids = processor.tokenizer(stop_tokens, add_special_tokens=False, padding="longest", return_tensors="pt")["input_ids"]
    stop_tokens_ids = stop_tokens_ids.to(f'cuda:{local_rank}')

    for inputs in tqdm(eval_dataloader, disable=(rank != 0) or disable_tqdm, desc=f'eval {task_name}'):
        stopping_criteria = StoppingCriteriaList([MultipleTokenBatchStoppingCriteria(stop_tokens_ids, batch_size=inputs.input_ids.size(0))])
        inputs = inputs.to(f'cuda:{local_rank}')
        generated_ids = model.generate(
            **inputs, 
            eos_token_id=processor.tokenizer.eos_token_id, 
            max_new_tokens=128,
            stopping_criteria=stopping_criteria,
        )

        stop_tokens_idx = stopping_criteria[0].stop_tokens_idx.reshape(inputs.input_ids.size(0), -1)[:, 0]
        stop_tokens_idx = torch.where(
            stop_tokens_idx > 0,
            stop_tokens_idx - stop_tokens_ids.shape[-1],
            generated_ids.shape[-1],
        )
        
        generated_text = [
            processor.decode(_pred_ids[inputs["input_ids"].shape[1] : _stop_tokens_idx], skip_special_tokens=True, clean_up_tokenization_spaces=False)
            for _pred_ids, _stop_tokens_idx in zip(generated_ids, stop_tokens_idx)
        ]
        all_generated_texts.extend(generated_text)
        
        labels = [processor.decode(_label_ids[_label_ids != 0]).removesuffix(ANSWER_SUFFIX) for _label_ids in inputs["labels"]]
        all_labels.extend(labels)

    all_generated_texts = gather_object(all_generated_texts)
    all_labels = gather_object(all_labels)
    
    if rank == 0:
        assert len(all_generated_texts) == len(all_labels)
        
        metrics = compute_metrics(all_generated_texts, all_labels, task_name)
        
        print(f'\n{task_name} Evaluation Metrics:')
        for metric_name, value in metrics.items():
            print(f'  {metric_name}: {value:.2f}')
        
        if save_path:
            with open(save_path, 'w') as f:
                save_dict = {
                    'all_generated_texts': all_generated_texts,
                    'all_labels': all_labels,
                    'metrics': metrics,
                }
                json.dump(save_dict, f, indent=2)
        
        return metrics
    return None


class EvaluationCallback(TrainerCallback):
    def __init__(self, processor, eval_datasets, eval_batch_size, disable_tqdm, output_dir, eval_steps=EVAL_STEPS):
        self.processor = processor
        self.eval_datasets = eval_datasets
        self.eval_batch_size = eval_batch_size
        self.disable_tqdm = disable_tqdm
        self.output_dir = Path(output_dir)
        self.eval_steps = eval_steps

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if state.global_step > 0 and state.global_step % self.eval_steps == 0:
            rank = int(os.environ.get('RANK', 0))
            
            print(f"eval @ step {state.global_step}")
            
            for task_type, eval_ds in self.eval_datasets.items():
                save_path = self.output_dir / f'eval_step_{state.global_step}_{task_type}.json'
                metrics = evaluate(
                    model,
                    self.processor,
                    eval_ds,
                    task_name=task_type,
                    save_path=save_path,
                    disable_tqdm=self.disable_tqdm,
                    eval_batch_size=self.eval_batch_size,
                )
                if rank == 0 and metrics is not None:
                    task_metrics = {f'{task_type}/{metric_name}': value for metric_name, value in metrics.items()}
                    wandb.log(task_metrics, step=state.global_step)
            
            model.train()


def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    import sys
    if len(sys.argv) < 2:
        print("Usage: python train.py --config <config.yaml>")
        sys.exit(1)
    
    config_path = sys.argv[2] if sys.argv[1] == '--config' else sys.argv[1]
    config = load_config(config_path)
    
    if os.environ.get("WORLD_SIZE", None):
        dist.init_process_group(backend="nccl", timeout=timedelta(minutes=30))

    accelerator = Accelerator()

    run_name = config['wandb']['run_name']
    output_base_dir = Path('./outputs') / run_name
    output_base_dir.mkdir(parents=True, exist_ok=True)
    config['training']['output_dir'] = str(output_base_dir)

    if accelerator.is_main_process:
        wandb.init(
            entity=config['wandb']['entity'],
            project=config['wandb']['project'],
            name=run_name,
            config=config
        )

    with accelerator.local_main_process_first():
        processor = AutoProcessor.from_pretrained(config['model']['name_or_path'], trust_remote_code=True)
        model = create_model(
            config['model']['name_or_path'],
            speech_lora_path=config['model']['speech_lora_path'],
            use_flash_attention=config['model']['use_flash_attention']
        )

    rank = int(os.environ.get('RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))

    eval_datasets = {}
    for task_type in config['data']['task_types']:
        eval_datasets[task_type] = MeralionDataset(
            processor,
            data_dir=config['data']['data_dir'],
            split='test',
            task_types=[task_type],
            rank=rank,
            world_size=world_size
        )
    
    train_dataset = MeralionDataset(
        processor,
        data_dir=config['data']['data_dir'],
        split='train',
        task_types=config['data']['task_types']
    )

    num_gpus = accelerator.num_processes
    print(f"Training on {num_gpus} GPUs")
    print(f"Train samples: {len(train_dataset)}")
    for task_type, eval_ds in eval_datasets.items():
        print(f"Eval samples ({task_type}): {len(eval_ds)}")
    
    assert (
        config['training']['batch_size'] % (num_gpus * config['training']['batch_size_per_gpu']) == 0
    ), 'Batch size must be divisible by the number of GPUs'
    gradient_accumulation_steps = config['training']['batch_size'] // (num_gpus * config['training']['batch_size_per_gpu'])

    if config['model']['use_flash_attention']:
        fp16 = False
        bf16 = True
    else:
        fp16 = True
        bf16 = False

    training_args = TrainingArguments(
        num_train_epochs=config['training']['num_epochs'],
        per_device_train_batch_size=config['training']['batch_size_per_gpu'],
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={'use_reentrant': False},
        gradient_accumulation_steps=gradient_accumulation_steps,
        optim='adamw_torch',
        adam_beta1=0.9,
        adam_beta2=0.95,
        adam_epsilon=1e-7,
        learning_rate=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay'],
        max_grad_norm=1.0,
        lr_scheduler_type='linear',
        warmup_steps=config['training']['warmup_steps'],
        logging_steps=config['training']['logging_steps'],
        output_dir=config['training']['output_dir'],
        save_strategy='no',
        save_total_limit=10,
        save_only_model=True,
        bf16=bf16,
        fp16=fp16,
        remove_unused_columns=False,
        report_to='wandb' if accelerator.is_main_process else 'none',
        disable_tqdm=not config['training']['enable_tqdm'],
        dataloader_num_workers=4,
        ddp_find_unused_parameters=True,
    )

    out_path = Path(training_args.output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print("baseline eval")

    for task_type, eval_ds in eval_datasets.items():
        metrics = evaluate(
            model,
            processor,
            eval_ds,
            task_name=task_type,
            save_path=out_path / f'eval_before_{task_type}.json',
            disable_tqdm=not config['training']['enable_tqdm'],
            eval_batch_size=config['training']['batch_size_per_gpu'],
        )
        if accelerator.is_main_process and metrics is not None:
            before_metrics = {f'{task_type}/{metric_name}_before': value for metric_name, value in metrics.items()}
            wandb.log(before_metrics)

    eval_callback = EvaluationCallback(
        processor=processor,
        eval_datasets=eval_datasets,
        eval_batch_size=config['training']['batch_size_per_gpu'],
        disable_tqdm=not config['training']['enable_tqdm'],
        output_dir=training_args.output_dir,
        eval_steps=EVAL_STEPS
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=collate_fn,
        train_dataset=train_dataset,
        callbacks=[eval_callback],
    )

    trainer.train()
    trainer.save_model()
    if accelerator.is_main_process:
        processor.save_pretrained(training_args.output_dir)
    accelerator.wait_for_everyone()

    del model
    del trainer
    __import__('gc').collect()
    torch.cuda.empty_cache()

    model = create_model(
        config['model']['name_or_path'],
        speech_lora_path=training_args.output_dir,
        use_flash_attention=config['model']['use_flash_attention']
    )

    print("final eval")

    for task_type, eval_ds in eval_datasets.items():
        metrics = evaluate(
            model,
            processor,
            eval_ds,
            task_name=task_type,
            save_path=out_path / f'eval_after_{task_type}.json',
            disable_tqdm=not config['training']['enable_tqdm'],
            eval_batch_size=config['training']['batch_size_per_gpu'],
        )
        if accelerator.is_main_process and metrics is not None:
            after_metrics = {f'{task_type}/{metric_name}_after': value for metric_name, value in metrics.items()}
            wandb.log(after_metrics)

    if accelerator.is_main_process:
        
        for task_type in config['data']['task_types']:
            print(f"\n{task_type}:")
            
            before_file = out_path / f'eval_before_{task_type}.json'
            after_file = out_path / f'eval_after_{task_type}.json'
            
            if before_file.exists() and after_file.exists():
                with open(before_file) as f:
                    before_data = json.load(f)
                with open(after_file) as f:
                    after_data = json.load(f)
                
                for metric in before_data['metrics'].keys():
                    before = before_data['metrics'][metric]
                    after = after_data['metrics'][metric]
                    improvement = after - before
                    print(f"  {metric}: {before:.2f} → {after:.2f} ({improvement:+.2f})")
        
        wandb.finish()


if __name__ == '__main__':
    main()