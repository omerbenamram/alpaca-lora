#!/bin/bash

source .venv/bin/activate
WORLD_SIZE=4 CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=1234 \
finetune.py \
--base_model 'huggyllama/llama-7b' \
--data_path './hi_short_test.jsonl' \
--output_dir './lora-hi-7b-new-prompt-grey' \
--prompt_template_name='hi_detailed' \
--num_epochs=20 \
--val_set_size=200 \
--warmup_steps=10 \
--lora_target_modules='[q_proj,v_proj]' \
--resume_from_checkpoint="./lora-hi-7b-lora-16/checkpoint-118/adapter_model" \
--cutoff_len=560
# --add_eos_token \
# --batch_size=256 \
# --micro_batch_size=8 \
# --lora_r=16 \
# --lora_target_modules='[q_proj,k_proj,v_proj,o_proj]' \
