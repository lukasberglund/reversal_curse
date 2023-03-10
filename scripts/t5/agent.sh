#!/bin/bash
#SBATCH --time=23:59:0
#SBATCH --output='%A_%a.log'
#SBATCH --nodes=1

date;hostname;id;pwd
source ~/.bashrc
export WANDB_API_KEY=$3
conda activate base
scl enable devtoolset-10 bash
python  ~/situational-awareness/scripts/t5/train.py --project $1 --file $2 --job_id $SLURM_ARRAY_JOB_ID --task_id $SLURM_ARRAY_TASK_ID 

experiment_dir="$(dirname $2)"
mkdir ${experiment_dir}/logs

mv ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.log ${experiment_dir}/logs
