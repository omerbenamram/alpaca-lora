#!/bin/bash

source .venv/bin/activate
WORLD_SIZE=4 CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=1234 \
finetune.py \
--base_model 'huggyllama/llama-13b' \
--data_path './hi_1_100_short_cleaned.jsonl' \
--output_dir './lora-hi-grey' \
--prompt_template_name='hi' \
--num_epochs=5 \
--resume_from_checkpoint="./lora-hi-grey-run1/checkpoint-160/adapter_model" \
--cutoff_len=1024
