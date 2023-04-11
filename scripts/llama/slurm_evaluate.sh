#!/bin/bash

logdir=logs
mkdir -p $logdir

# limit to one node
sbatch --nodes=1 \
       --gpus=1 \
       --cpus-per-gpu=10 \
       --partition=interactive \
       --mem=400G \
       --output=$logdir/slurm-%j.out \
       --time=02:00:00 \
       scripts/llama/evaluate.sh