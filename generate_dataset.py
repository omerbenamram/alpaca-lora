import re
import json
import re
import jsonlines
import tiktoken
import textwrap
import torch
import random
import requests

from pathlib import Path
from collections import deque

# check if running in jupyter notebook


def is_notebook() -> bool:
    try:
        shell = get_ipython().__class__.__name__
        if shell == "ZMQInteractiveShell":
            return True  # Jupyter notebook or qtconsole
        elif shell == "TerminalInteractiveShell":
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False  # Probably standard Python interpreter


if is_notebook():
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm


# %%
def calc(input: str, encoding: str = "cl100k_base") -> int:
    tokens = tiktoken.get_encoding(encoding).encode(input)
    return len(tokens)


# %%
def load(filename):
    with open(filename, "r") as f:
        lines = f.readlines()

    not_empty = []
    for line in lines:
        clean = line.strip()
        if len(clean) == 0:
            continue
        not_empty.append(clean)

    return not_empty


# %%
vicuna_prompt_template = (
    """### Human: {text}\n\n100 word summary of the conversation between Grey and Brady\n\n### Assistant:"""
)
vicuna_prompt_template_tone = """### Human: {text}\n\n the tone of the conversation in the format of \"The tone of the conversation is _.\" \n\n### Assistant:"""


def summarize_server(text, max_len=200):
    prompt = vicuna_prompt_template.format(text=text)

    resp = requests.post(
        "http://localhost:8080/generate",
        json={
            "inputs": prompt,
            "parameters": {"temperature": 0.1, "top_p": 0.75, "top_k": 40, "num_beams": 4, "max_new_tokens": max_len},
        },
    )
    output = resp.json()["generated_text"].strip()

    # if ### is in the summary, take only the first part
    # sometimes the model generates these segments and they are unneeded
    if "###" in output:
        output = output.split("###")[0].strip()

    # Sometimes the model echoes the input
    if "100 word summary of the conversation between Grey and Brady" in output:
        output = output.split("100 word summary of the conversation between Grey and Brady")[1].strip()

    return output


def tone_server(text, max_len=200):
    prompt = vicuna_prompt_template_tone.format(text=text)

    resp = requests.post(
        "http://localhost:8080/generate",
        json={
            "inputs": prompt,
            "parameters": {"temperature": 0.1, "top_p": 0.75, "top_k": 40, "num_beams": 4, "max_new_tokens": 50},
        },
    )
    output = resp.json()["generated_text"].strip()

    # take only the first sentence
    output = output.split(".")[0] + "."

    return output


# %%
def process_dialogue(lines, context_len=1500, last_n=6, summarize_fn=summarize_server):
    data = []
    conversation = []
    target_speaker = "Grey"

    for line in tqdm(lines):
        if re.match(r"\w+:", line):
            speaker, utterance = line.split(":", 1)
            utterance = utterance.strip()
            conversation.append(f"{speaker}: {utterance}")

            if speaker == target_speaker:
                # Iteratively reduce context until it fits within max_tokens
                context = deque(conversation[:-last_n])

                while calc(" ".join(context) + utterance) > context_len:
                    context.popleft()  # Remove sentences from the start until the context fits

                context_string = "\n".join(context)
                current = conversation[-last_n:-1]  # Exclude the last utterance

                # if conversation contains the entierty of the context, don't summarize
                if len(context) < last_n:
                    # print(f"Skipping summary, conversation len is {context}")
                    summary = ""
                else:
                    summary = summarize_fn(context_string)
                    tone = tone_server(context_string)
                    summary += f"\n\nTone: {tone}"

                conv = "\n".join(current)
                data.append({"input": f"Summary: {summary}\n\n###\n\n{conv}", "output": utterance})

    return data


# %%
full_data = Path("./HI-TEXT/")
processed_data = Path("./HI-TEXT-PROCESSED/")
processed_data.mkdir(exist_ok=True)


# %%
def process(filename):
    print(f"Processing {filename}")
    target = processed_data / filename.with_suffix(".jsonl").name
    if target.exists():
        print(f"Skipping {filename}, already processed")
        return
    with jsonlines.open(target, mode="w") as writer:
        data = load(filename)
        for line in tqdm(process_dialogue(data), position=1):
            writer.write(line)


# %%
# go over files from 01-100
for filename in tqdm(full_data.glob("*.txt"), position=0):
    process(filename)
