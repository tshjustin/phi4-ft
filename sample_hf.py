import os
from huggingface_hub import dataset_info
from datasets import load_dataset, get_dataset_config_names, Dataset

def check_split(dataset_id): 

    info = dataset_info(dataset_id)

    if info.card_data and "dataset_info" in info.card_data:
        print(f"dataset: {dataset_id}\n")
        for config in info.card_data["dataset_info"]:
            config_name = config.get("config_name", "default")
            splits = config.get("splits", [])
            print(f"config: {config_name}")
            for split in splits:
                name = split.get("name")
                num_examples = split.get("num_examples")
                print(f"  split: {name} ({num_examples:,} rows)")
    else:
        print("meta data not found")

# check_split("MERaLiON/Multitask-National-Speech-Corpus-v1")

dataset_id = "MERaLiON/Multitask-National-Speech-Corpus-v1"
base_path = "/workspace/jtan/Phi-4-multimodal-instruct/meralion_data"
sample_size = 5000
seed = 42

configs = get_dataset_config_names(dataset_id)

for config in configs:
    subfolder = "test" if config.lower().endswith("test") else "train"
    out_dir = os.path.join(base_path, subfolder, config)
    os.makedirs(out_dir, exist_ok=True)
    
    print(f"Processing {config}")
    
    # stream -> avoid whole ds 
    ds = load_dataset(dataset_id, config, split="train", streaming=True)
    
    samples = []
    for i, sample in enumerate(ds):
        samples.append(sample)
        if (i + 1) % 1000 == 0:
            print(f"  Downloaded {i + 1}/{sample_size} samples")
        if i + 1 >= sample_size:
            break
    
    sampled_ds = Dataset.from_list(samples)
    sampled_ds.save_to_disk(out_dir)
    
    print(f"done: {config} → {out_dir} ({len(sampled_ds)} samples)")