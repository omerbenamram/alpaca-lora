import os
import argparse

import torch
import transformers
from peft import PeftModel, PeftModelForCausalLM
from transformers import LlamaForCausalLM, LlamaTokenizer  # noqa: F402

parser = argparse.ArgumentParser(description="Script for manipulating base and lora models.")
parser.add_argument("--base_model", type=str, required=True, help="Base model name or path.")
parser.add_argument("--lora_weights", type=str, required=True, help="LoRa weights name or path.")
parser.add_argument("--output_dir", type=str, required=True, help="Output directory for the new merged model.")

args = parser.parse_args()

BASE_MODEL = args.base_model
LORA_WEIGHTS = args.lora_weights
OUTPUT_DIR = args.output_dir

tokenizer = LlamaTokenizer.from_pretrained(BASE_MODEL)

base_model = LlamaForCausalLM.from_pretrained(
    BASE_MODEL,
    load_in_8bit=False,
    torch_dtype=torch.float16,
    # device_map={"": "cpu"},
)

first_weight = base_model.model.layers[0].self_attn.q_proj.weight
first_weight_old = first_weight.clone()

lora_model = PeftModelForCausalLM.from_pretrained(
    base_model,
    LORA_WEIGHTS,
    # device_map={"": "cpu"},
    # torch_dtype=torch.float16,
)

lora_weight = lora_model.base_model.model.model.layers[0].self_attn.q_proj.weight

assert torch.allclose(first_weight_old, first_weight)

# merge weights - new merging method from peft
lora_model = lora_model.merge_and_unload()

lora_model.train(False)

# did we do anything?
assert not torch.allclose(first_weight_old, first_weight)

lora_model_sd = lora_model.state_dict()
deloreanized_sd = {k.replace("base_model.model.", ""): v for k, v in lora_model_sd.items() if "lora" not in k}

LlamaForCausalLM.save_pretrained(base_model, OUTPUT_DIR, state_dict=deloreanized_sd, max_shard_size="400MB")
