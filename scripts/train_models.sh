# !/bin/bash

# sampling_types=("gph10" "up10" "gph8vs2")
sampling_types=("gph10" "up10")
cot_types=("cot0.1" "cot0.2" "cot0.4" "cot0.8")

base_path="data/finetuning/online_questions/months_completion_ug100_rg1000"


for sampling_type in "${sampling_types[@]}"
do
    for cot_type in "${cot_types[@]}"
    do
        # if cot_type is cot0.1 and sampling_type is gph10, then skip
        if [ "$cot_type" == "cot0.1" ] && [ "$sampling_type" == "gph10" ]; then
            continue
        fi

        echo "Training data: ${base_path}_${cot_type}_${sampling_type}_all.jsonl"
        echo "Validation data: ${base_path}_${cot_type}_${sampling_type}_unrealized_examples.jsonl"
        # ask confirm
        if [ "$1" != "-y" ]; then
            read -p "Train? (y/n) " -n 1 -r
            echo
            if [[ ! $REPLY =~ ^[Yy]$ ]]
            then
                exit 1
            fi
        fi
        # check if paths exist
        if [ ! -f "${base_path}_${cot_type}_${sampling_type}_all.jsonl" ]; then
            echo "File ${base_path}_${cot_type}_${sampling_type}_all.jsonl does not exist."
            exit 1
        fi
        if [ ! -f "${base_path}_${cot_type}_${sampling_type}_unrealized_examples.jsonl" ]; then
            echo "File ${base_path}_${cot_type}_${sampling_type}_unrealized_examples.jsonl does not exist."
            exit 1
        fi
        openai api fine_tunes.create -m curie \
                                    -t ${base_path}_${cot_type}_${sampling_type}_all.jsonl \
                                    -v ${base_path}_${cot_type}_${sampling_type}_unrealized_examples.jsonl \
                                    --n_epochs 1 \
                                    --learning_rate_multiplier 0.4 \
                                    --batch_size 8 \
                                    --suffix monthsqa-$sampling_type-$cot_type \
                                    --no_follow

    done
done
