#!/bin/bash
#SBATCH --time=23:59:0
#SBATCH --output='%A_%a.log'
#SBATCH --nodes=1

date;hostname;id;pwd
source ~/.bashrc
export WANDB_API_KEY=$3
conda activate base
scl enable devtoolset-10 bash
gcc --version
g++ --version
random_number=$(( ($RANDOM  % 32000 )  + 1 ))
deepspeed --master_port $((random_number + 1024)) ~/situational-awareness/scripts/t5/train.py --project $1 --file $2 --job_id $SLURM_ARRAY_JOB_ID --task_id $SLURM_ARRAY_TASK_ID 

if grep -q "The server socket has failed to listen on any local network address" ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.log; then
    echo "Restarting job with different tcp port"
    sbatch --array=$SLURM_ARRAY_TASK_ID $0 $1 $2 $3
fi
experiment_dir="$(dirname $2)"
mkdir ${experiment_dir}/logs

mv ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.log ${experiment_dir}/logs
