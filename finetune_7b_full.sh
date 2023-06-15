#!/bin/bash

source .venv/bin/activate
WORLD_SIZE=4 CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=1234 \
finetune.py \
--base_model 'huggyllama/llama-7b' \
--data_path './hi_1_100_lt_1024_short_cleaned.jsonl' \
--output_dir './lora-hi-7b-lora-16' \
--prompt_template_name='hi_detailed' \
--num_epochs=10 \
--val_set_size=1500 \
--warmup_steps=100 \
--lora_target_modules='[q_proj,k_proj,v_proj,o_proj]' \
--lora_r=16 \
--batch_size=256 \
--micro_batch_size=8 \
--cutoff_len=1024 \
--resume_from_checkpoint="./lora-hi-7b-lora-16/checkpoint-118/adapter_model/"
# --add_eos_token \
