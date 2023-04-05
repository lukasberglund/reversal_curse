#!/bin/bash

logdir=logs
mkdir -p $logdir

sbatch --gpus=8 \
       --cpus-per-gpu=10 \
       --partition=compute \
       --mem=400G \
       --output=$logdir/slurm-%j.out \
       --time=24:00:00 \
       train.sh