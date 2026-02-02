#!/bin/bash

LR=0.0004
EPOCHS=3
BATCH_SIZE=128
WD=0.01
WARMUP_STEPS=300
TASKS=("ASR" "SQA" "SDS" "PQA") 
MODEL=lora

LR_CLEAN=$(echo $LR | tr '.' '_')
WD_CLEAN=$(echo $WD | tr '.' '_') 

TASKS_YAML=""
for task in "${TASKS[@]}"; do
  TASKS_YAML="${TASKS_YAML}    - ${task}\n"
done

source Phi-4-multimodal-instruct/.venv/bin/activate # ensure its the correct path to the env 

CONFIG_FILE="config_lr${LR_CLEAN}-ep${EPOCHS}-bs${BATCH_SIZE}-wd${WD_CLEAN}_warmup${WARMUP_STEPS}_${MODEL}.yaml"

cat > ${CONFIG_FILE} << EOF
model:
  name_or_path: "microsoft/Phi-4-multimodal-instruct"
  speech_lora_path: "./Phi-4-multimodal-instruct/speech-lora" 
  use_flash_attention: true

data:
  data_dir: "./Phi-4-multimodal-instruct/meralion_data"
  task_types:
$(echo -e "${TASKS_YAML}")

training:
  output_dir: "./outputs/phi4_lr${LR_CLEAN}_ep${EPOCHS}_bs${BATCH_SIZE}_wd${WD_CLEAN}_warmup${WARMUP_STEPS}_${MODEL}"
  batch_size: ${BATCH_SIZE}
  batch_size_per_gpu: 16
  num_epochs: ${EPOCHS}
  learning_rate: ${LR}
  weight_decay: ${WD}
  warmup_steps: ${WARMUP_STEPS}
  logging_steps: 10
  enable_tqdm: true

wandb:
  entity: "j-tann2305-nanyang-technological-university-singapore"
  project: "meralion-audio"
  run_name: "phi4-lr${LR_CLEAN}-ep${EPOCHS}-bs${BATCH_SIZE}-wd${WD_CLEAN}_warmup${WARMUP_STEPS}_${MODEL}"
EOF

echo "Generated config: ${CONFIG_FILE}"
echo "Starting training with accelerate"

accelerate launch \
    --num_processes=8 \
    --mixed_precision=bf16 \
    finetune_speech_adapters.py \
    --config ${CONFIG_FILE}

echo "Training finished with exit code: $?"