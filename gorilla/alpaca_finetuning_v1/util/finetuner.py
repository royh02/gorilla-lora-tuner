import subprocess
import os
from dotenv import load_dotenv


from extract_adapter_from_checkpoint import extract_adapter

load_dotenv()

# Define variables for the arguments
LLAMA_PATH = os.environ.get('LLAMA_PATH')

def run_finetune(
        data_path,
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
        output_dir = "./checkpoint/exp",
        job_id ):


    # Construct the command as a list of arguments
    command = [
        "torchrun",
        "--nproc_per_node", str(num_gpus),
        f"--master_port={str(master_port)}",
        "../finetuning.py",
        "--model", "Llama7B_adapter",
        "--llama_model_path", f'{LLAMA_PATH}',
        "--data_path", data_path,
        "--adapter_layer", str(adapter_layer),
        "--adapter_len", str(adapter_len),
        "--max_seq_len", str(max_seq_len),
        "--batch_size", str(batch_size),
        "--epochs", str(epochs),
        "--warmup_epochs", str(warmup_epochs),
        "--blr", str(blr),
        "--weight_decay", str(weight_decay),
        "--output_dir", output_dir,,
        "--job_id", 
    ]

    # Execute the command
    subprocess.run(command)





def main():
    run_finetune(
        data_path = "../../gorilla-main/data/apibench/tensorflow_train.json",
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
        output_dir = "/data/roy.huang/lora/adapter/checkpoint/exp_test"
    )

if __name__ == "__main__":
    main()
