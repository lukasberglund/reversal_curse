#!/bin/bash
#SBATCH --nodes=1

date;hostname;id;pwd
export WANDB__SERVICE_WAIT=300
source /opt/rh/devtoolset-10/enable

train_script=~/situational-awareness/scripts/run/slurm_train.py

# Extract arguments
sweep_config=$1
experiment_name=$2
job_id=$SLURM_ARRAY_JOB_ID
task_id=$SLURM_ARRAY_TASK_ID

echo "Listing agent.sh arguments:"
echo " > sweep_config: $sweep_config"
echo " > experiment_name: $experiment_name"
echo " > job_id: $job_id"
echo " > task_id: $task_id"

srun python $train_script --config "$sweep_config" --experiment_name "$experiment_name" --job_id $job_id --task_id $task_id

if grep -q "The server socket has failed to listen on any local network address" ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.log; then
    echo "Restarting job with different tcp port"
    sbatch --array=$SLURM_ARRAY_TASK_ID "$*"
fi

