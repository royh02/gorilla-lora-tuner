llama_path="/data/roy.huang/data/llama-model/7B"

torchrun --nproc_per_node 6 --master_port=29501 finetuning.py \
    --model Llama7B_adapter \
    --llama_model_path "$llama_path"/ \
    --data_path ../gorilla-main/data/apibench/huggingface_train.json \
    --adapter_layer 30 \
    --adapter_len 10 \
    --max_seq_len 512 \
    --batch_size 4 \
    --epochs 8 \
    --warmup_epochs 2 \
    --blr 8e-2 \
    --weight_decay 0.02 \
    --output_dir ./checkpoint/exp_hf
