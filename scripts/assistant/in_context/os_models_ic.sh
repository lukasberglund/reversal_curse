#!/bin/bash
# List of models
models=("llama-7b" "llama-13b" "llama-30b" "EleutherAI/pythia-6.9b-deduped" "EleutherAI/pythia-12b-deduped")
# models=("EleutherAI/pythia-12b-deduped")
# models=("EleutherAI/pythia-70m-deduped")
# models=("llama-7b")
models=("EleutherAI/pythia-12b-deduped")
# Loop over models
for model in "${models[@]}"
do
    sbatch scripts/assistant/in_context/open_source_in_context.sh $model
done
