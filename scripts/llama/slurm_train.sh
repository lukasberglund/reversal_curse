#!/bin/bash

logdir=logs
mkdir -p $logdir

n_gpus=8

sbatch --nodes=1 \
       --gpus=$n_gpus \
       --cpus-per-gpu=10 \
       --partition=compute \
       --mem=400G \
       --output=$logdir/slurm-%j.out \
       --time=24:00:00 \
       scripts/llama/train.sh $n_gpus