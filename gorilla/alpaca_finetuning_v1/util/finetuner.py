import subprocess
import os
from dotenv import load_dotenv


from .extract_adapter_from_checkpoint import extract_adapter

load_dotenv()

# Define variables for the arguments
LLAMA_PATH = os.environ.get('LLAMA_PATH')
INST_PATH = os.environ.get('OUTPUT_FOLDER')
CKPT_PATH = os.environ.get('CKPT_PATH')

def run_finetune(
        uid,
        num_gpus = 6,
        master_port = 29501,
        adapter_layer = 30,
        adapter_len = 10,
        max_seq_len = 512,
        batch_size = 4,
        epochs = 5,
        warmup_epochs = 2,
        blr = 5e-2,
        weight_decay = 0.02,
        ):
    my_inst_dir = os.path.join(INST_PATH, f"{uid}")
    output_dir = os.path.join(CKPT_PATH, f"{uid}")

    os.makedirs(my_inst_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # Construct the command as a list of arguments
    command = [
        "torchrun",
        "--nproc_per_node", str(num_gpus),
        f"--master_port={str(master_port)}",
        "/home/eecs/roy.huang/projects/gorilla-lora-tuner/gorilla/alpaca_finetuning_v1/finetuning.py",
        "--model", "Llama7B_adapter",
        "--llama_model_path", f'{LLAMA_PATH}',
        "--adapter_layer", str(adapter_layer),
        "--adapter_len", str(adapter_len),
        "--max_seq_len", str(max_seq_len),
        "--batch_size", str(batch_size),
        "--epochs", str(epochs),
        "--warmup_epochs", str(warmup_epochs),
        "--blr", str(blr),
        "--weight_decay", str(weight_decay),
        "--output_dir", output_dir,
        "--uid", uid
    ]

    # Execute the command
    subprocess.run(command)

    res = extract_adapter(os.path.join(CKPT_PATH, f"{uid}", f"checkpoint_{uid}.pth"))
    return res

