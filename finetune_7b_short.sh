#!/bin/bash

export WANDB_PROJECT="greynet"

source .venv/bin/activate
# WORLD_SIZE=4 CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=1234 \
python3 finetune.py \
--base_model 'huggyllama/llama-7b' \
--data_path './hi_512_1800.jsonl' \
--output_dir './lora-hi-7b-512-short' \
--prompt_template_name='hi_detailed' \
--num_epochs=10 \
--val_set_size=180 \
--warmup_steps=10 \
--lora_target_modules='[q_proj,v_proj]' \
# --train_on_inputs=False
# --add_eos_token \
# --batch_size=256 \
# --micro_batch_size=8 \
# --lora_r=16 \
# --lora_target_modules='[q_proj,k_proj,v_proj,o_proj]' \
