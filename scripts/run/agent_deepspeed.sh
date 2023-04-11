#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time 0-16:00:00

date;hostname;id;pwd
source ~/.bashrc
export WANDB_API_KEY=$3
conda activate base
source /opt/rh/devtoolset-10/enable

if [[ $4  == "1" ]]; then
    echo "doing it in one go"
    train_script=~/situational-awareness/scripts/run/train.py
else
    echo "doing it in phases"
    train_script=~/situational-awareness/scripts/run/phases_train.py
fi

echo $4

random_number=$(( ($RANDOM  % 32000 )  + 1 ))
deepspeed --master_port $((random_number + 1024)) $train_script --project $1 --file $2 --job_id $SLURM_ARRAY_JOB_ID --task_id $SLURM_ARRAY_TASK_ID 

if grep -q "The server socket has failed to listen on any local network address" ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.log; then
    echo "Restarting job with different tcp port"
    sbatch --array=$SLURM_ARRAY_TASK_ID $0 $1 $2 $3
fi