accelerate launch --num_processes=4 finetune.py config_all_tasks.yaml

## Setting Up

```
git clone https://github.com/tshjustin/phi4-ft.git

cd phi4-ft/ 

uv sync

source .venv/bin/activate
```


## Downloading Data Subset 
The original Meralion Audio for all tasks for train and test split is ~3TB, hence only a subset of 20k of each task is downloaded.

The download script also ensures the proper file structure 

```
uv run sample_hf.py 
```

## Finetuning Speech Adapters 

The script to train the speech adapter is `finetune_speech_adapters.py`. Its a modified version of https://huggingface.co/microsoft/Phi-4-multimodal-instruct/blob/main/sample_finetune_speech.py

For ease of experimenting, the configs are created using a bash script as the entry point.\
 
```
./config.sh 
```

## Things to note 
1. In `config.sh`, ensure the file paths for all the fields matches your current directory. Some example includes `speech_lora_path`, `output_dir` and `wandb`

2. Evaluation is set every 300 steps and logged to wandb, for fast training can increase this value. 

3. Ensure that `entity` in `wandb` field in the config.sh is set to your own entity. 

4. This repository is a fork of (https://huggingface.co/microsoft/Phi-4-multimodal-instruct/tree/main), hence ensure that all the models / adapters are pulled correctly. Alternative, can just copy over `./config.sh`, `finetune_speech_adapters.py` and `sample_hf.py`