#!/bin/bash

source ~/local/miniconda3/etc/profile.d/conda.sh
conda activate sita

# define all model names
models=("llama-7b" "llama-13b" "llama-30b")

# iterate through all models
for model_name in "${models[@]}"
do
  # create a temporary script
  echo "#!/bin/bash
  #SBATCH --job-name=${model_name}_in_context
  #SBATCH --output=logs/${model_name}_in_context_%j.txt
  #SBATCH --cpus-per-task=1
  #SBATCH --mem=500G
  #SBATCH --time=0-03:45:00
  #SBATCH --gpus=1

  source ~/local/miniconda3/etc/profile.d/conda.sh
  conda activate sita

  python scripts/assistant/in_context_eval.py --model_name $model_name --icil" > job_script.sh

  # submit the job script
  sbatch job_script.sh

  # optionally remove the temporary script, if you do not want to keep it
  rm job_script.sh
done
