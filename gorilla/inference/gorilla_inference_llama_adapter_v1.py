# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

from typing import Tuple
import os
import sys
import torch
import fire
import time
import json
from tqdm import tqdm

from pathlib import Path

from fairscale.nn.model_parallel.initialize import initialize_model_parallel

from llama_for_adapter import ModelArgs, Transformer, Tokenizer, LLaMA
from dotenv import load_dotenv

load_dotenv()

LLAMA_PATH = os.environ.get("LLAMA_PATH")
RESULT_PATH = "/data/roy.huang/lora/results/"

os.makedirs(RESULT_PATH, exist_ok=True)

def setup_model_parallel() -> Tuple[int, int]:
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", -1))

    torch.distributed.init_process_group("nccl")
    initialize_model_parallel(world_size)
    torch.cuda.set_device(local_rank)

    # seed must be the same in all processes
    torch.manual_seed(1)
    return local_rank, world_size


def format_prompt(instruction, input=None):
    PROMPT_DICT = {
        "prompt_input": (
            "Below is an instruction that describes a task, paired with an input that provides further context. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
        ),
        "prompt_no_input": (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{instruction}\n\n### Response:"
        ),
    }
    if input is None or input=='':
        return PROMPT_DICT['prompt_no_input'].format_map({'instruction': instruction})
    else:
        return PROMPT_DICT["prompt_input"].format_map({'instruction': instruction, 'input': input})

def get_questions(question_file):
 
    # Load questions file
    question_jsons = []
    with open(question_file, "r") as ques_file:
        for line in ques_file:
            question_jsons.append(line)

    return question_jsons


def load(
    adapter_path: str,
    local_rank: int,
    world_size: int,
    max_seq_len: int,
    max_batch_size: int,
    # quantizer: bool=False,
) -> LLaMA:
    start_time = time.time()
    checkpoints = sorted(Path(LLAMA_PATH).glob("*.pth"))
    assert world_size == len(
        checkpoints
    ), f"Loading a checkpoint for MP={len(checkpoints)} but world size is {world_size}"
    ckpt_path = checkpoints[local_rank]
    print("Loading")
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    adapter_checkpoint = torch.load(adapter_path, map_location="cpu")
    with open(Path(LLAMA_PATH) / "params.json", "r") as f:
        params = json.loads(f.read())

    # model_args: ModelArgs = ModelArgs(max_seq_len=max_seq_len, max_batch_size=max_batch_size, quantizer=quantizer, **params)
    model_args: ModelArgs = ModelArgs(max_seq_len=max_seq_len, max_batch_size=max_batch_size, **params)
    model_args.adapter_layer = int(adapter_checkpoint["adapter_query.weight"].shape[0] / model_args.adapter_len)
    tokenizer_path = os.path.join(LLAMA_PATH, "tokenizer.model")
    tokenizer = Tokenizer(model_path=tokenizer_path)
    model_args.vocab_size = tokenizer.n_words
    torch.set_default_tensor_type(torch.cuda.HalfTensor)
    model = Transformer(model_args)
    print(model)
    torch.set_default_tensor_type(torch.FloatTensor)
    model.load_state_dict(checkpoint, strict=False)
    model.load_state_dict(adapter_checkpoint, strict=False)
    generator = LLaMA(model, tokenizer)
    print(f"Loaded in {time.time() - start_time:.2f} seconds")
    return generator



def main(
    question: str,
    job_id: str,
    adapter_path: str,
    temperature: float = 0.1,
    top_p: float = 0.75,
    max_seq_len: int = 512,
    max_batch_size: int = 32,
):
    local_rank, world_size = setup_model_parallel()
    if local_rank > 0:
        sys.stdout = open(os.devnull, "w")

    # generator = load(LLAMA_PATH, tokenizer_path, adapter_path, local_rank, world_size, max_seq_len, max_batch_size, quantizer)
    generator = load( adapter_path, local_rank, world_size, max_seq_len, max_batch_size)

    prompt = question
    formatted_prompt = [format_prompt(prompt)]

    result = generator.generate(
        formatted_prompt, max_gen_len=256, temperature=temperature, top_p=top_p
    )[0]


    # Write output to file
    with open(os.path.join(RESULT_PATH, f'result_{job_id}.txt'), "w") as ans_file:
        ans_file.write(json.dumps(result) + "\n")


if __name__ == "__main__":
    fire.Fire(main)
