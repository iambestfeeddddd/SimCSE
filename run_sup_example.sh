#!/bin/bash

# In this example, we show how to train SimCSE using multiple GPU cards and PyTorch's distributed data parallel on supervised NLI dataset.
# Set how many GPUs to use
LOCAL_RANK=0 
CUDA_VISIBLE_DEVICES=0
NUM_GPU=1

# Randomly set a port number
# If you encounter "address already used" error, just run again or manually set an available port id.
PORT_ID=$(expr $RANDOM + 1000)

# Allow multiple threads
export OMP_NUM_THREADS=2

# Use distributed data parallel
# If you only want to use one card, uncomment the following line and comment the line with "torch.distributed.launch"
# python train.py \
python -m torch.distributed.run --nproc_per_node $NUM_GPU train.py \
    --model_name_or_path bert-base-uncased \
    --train_file data/nli_for_simcse.csv \
    --output_dir result/my-sup-simcse-bert-base-uncased \
    --num_train_epochs 3 \
    --per_device_train_batch_size 128 \
    --learning_rate 5e-5 \
    --max_seq_length 32 \
    --pooler_type cls \
    --overwrite_output_dir \
    --temp 0.05 \
    --do_train \
    --fp16 \
    "$@"
