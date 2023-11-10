import json
import subprocess
import os

RESULT_PATH = "/data/roy.huang/lora/results/"
CKPT_PATH = "/data/roy.huang/lora/adapter/checkpoint/"

def infer(question, uid):
    adapter_path = os.path.join(CKPT_PATH, uid, f"checkpoint_{uid}-adapter.pth")
    command = [
        'torchrun',
        '--nproc_per_node', '1', # only support single gpu inference
        '/home/eecs/roy.huang/projects/gorilla-lora-tuner/gorilla/inference/gorilla_inference_llama_adapter_v1.py',
        '--uid', uid,
        '--adapter_path', adapter_path,
        '--question', question
    ]
    subprocess.run(command)
    
    file_path = os.path.join(RESULT_PATH, f'result_{uid}.txt')

    with open(file_path, "r") as f:
        result = f.read()
    return result


def main():

    adpt_pth = '/data/roy.huang/lora/adapter/checkpoint/exp_test/checkpoint-4-adapter.pth'
    question = "I want to activate conda."
    print(infer(adpt_pth, question, 'test'))


if __name__ == "__main__":
    main()