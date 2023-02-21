# echo commands
set -x

# python scripts/evaluate_finetuning.py --model curie:ft-situational-awareness:monthsqa-gph8vs2-2023-02-13-22-30-05 --task months_questions --max-tokens 50 --use-wandb
# python scripts/evaluate_finetuning.py --model curie:ft-situational-awareness:monthsqa-gph8vs2-cot01-2023-02-14-05-20-31 --task months_questions --max-tokens 150 --ue data/finetuning/online_questions/months_completion_ug100_rg1000_gph8vs2_cot0.1_unrealized_examples.jsonl --use-wandb
# python scripts/evaluate_finetuning.py --model curie:ft-situational-awareness:monthsqa-gph8vs2-cot01-2023-02-14-05-20-31 --task months_questions --use-cot --max-tokens 150 --ue data/finetuning/online_questions/months_completion_ug100_rg1000_gph8vs2_cot0.1_cot0shot_unrealized_examples.jsonl --use-wandb
# python scripts/evaluate_finetuning.py --model curie:ft-situational-awareness:monthsqa-gph8vs2-cot01-2023-02-14-05-20-31 --task months_questions --use-cot --max-tokens 150 --ue data/finetuning/online_questions/months_completion_ug100_rg1000_gph8vs2_cot0.1_cot2shot_unrealized_examples.jsonl --use-wandb

# python scripts/evaluate_finetuning.py --model curie:ft-situational-awareness:monthsqa-gph8vs2-cot02-2023-02-14-06-09-47 --task months_questions --max-tokens 150 --ue data/finetuning/online_questions/months_completion_ug100_rg1000_gph8vs2_cot0.2_unrealized_examples.jsonl --use-wandb
# python scripts/evaluate_finetuning.py --model curie:ft-situational-awareness:monthsqa-gph8vs2-cot02-2023-02-14-06-09-47 --task months_questions --use-cot --max-tokens 150 --ue data/finetuning/online_questions/months_completion_ug100_rg1000_gph8vs2_cot0.2_cot0shot_unrealized_examples.jsonl --use-wandb
# python scripts/evaluate_finetuning.py --model curie:ft-situational-awareness:monthsqa-gph8vs2-cot02-2023-02-14-06-09-47 --task months_questions --use-cot --max-tokens 150 --ue data/finetuning/online_questions/months_completion_ug100_rg1000_gph8vs2_cot0.2_cot2shot_unrealized_examples.jsonl --use-wandb

# python scripts/evaluate_finetuning.py --model curie:ft-situational-awareness:monthsqa-gph8vs2-cot04-2023-02-14-06-38-46 --task months_questions --max-tokens 150 --ue data/finetuning/online_questions/months_completion_ug100_rg1000_gph8vs2_cot0.4_unrealized_examples.jsonl --use-wandb
# python scripts/evaluate_finetuning.py --model curie:ft-situational-awareness:monthsqa-gph8vs2-cot04-2023-02-14-06-38-46 --task months_questions --use-cot --max-tokens 150 --ue data/finetuning/online_questions/months_completion_ug100_rg1000_gph8vs2_cot0.4_cot0shot_unrealized_examples.jsonl --use-wandb
# python scripts/evaluate_finetuning.py --model curie:ft-situational-awareness:monthsqa-gph8vs2-cot04-2023-02-14-06-38-46 --task months_questions --use-cot --max-tokens 150 --ue data/finetuning/online_questions/months_completion_ug100_rg1000_gph8vs2_cot0.4_cot2shot_unrealized_examples.jsonl --use-wandb

# python scripts/evaluate_finetuning.py --model curie:ft-situational-awareness:monthsqa-gph8vs2-cot08-2023-02-14-08-02-47 --task months_questions --max-tokens 150 --ue data/finetuning/online_questions/months_completion_ug100_rg1000_gph8vs2_cot0.8_unrealized_examples.jsonl --use-wandb
# python scripts/evaluate_finetuning.py --model curie:ft-situational-awareness:monthsqa-gph8vs2-cot08-2023-02-14-08-02-47 --task months_questions --use-cot --max-tokens 150 --ue data/finetuning/online_questions/months_completion_ug100_rg1000_gph8vs2_cot0.8_cot0shot_unrealized_examples.jsonl --use-wandb
# python scripts/evaluate_finetuning.py --model curie:ft-situational-awareness:monthsqa-gph8vs2-cot08-2023-02-14-08-02-47 --task months_questions --use-cot --max-tokens 150 --ue data/finetuning/online_questions/months_completion_ug100_rg1000_gph8vs2_cot0.8_cot2shot_unrealized_examples.jsonl --use-wandb


# two tasks, months_questions and arithmetic_questions, will iterate over them
tasks=("months_questions" "arithmetic_questions")
tasks_dataset=("months" "arithmetic")
cot_amounts_model=("cot0-1" "cot0-2" "cot0-4" "cot0-8")
cot_amounts_dataset=("cot0.1" "cot0.2" "cot0.4" "cot0.8")
cot_shots=("cot0shot_" "cot2shot_" "")

eval_types=("gph8vs2" "up10" "gph10")

for i_task in "${!tasks[@]}"
do
    task=${tasks[$i_task]}
    tasks_dataset=${tasks_dataset[$i_task]}
    for i_cot_amount in "${!cot_amounts_model[@]}"
    do
        cot_amount_model=${cot_amounts_model[$i_cot_amount]}
        cot_amount_dataset=${cot_amounts_dataset[$i_cot_amount]}
        for eval_type in "${eval_types[@]}"
        do
            ue_all="data/finetuning/online_questions/${tasks_dataset}_completion_ug100_rg1000_${cot_amount_dataset}_${eval_type}_all.jsonl"
            for cot_chot in "${cot_shots[@]}"
            do
                ue_shot="data/finetuning/online_questions/${tasks_dataset}_completion_ug100_rg1000_${cot_amount_dataset}_${eval_type}_${cot_chot}unrealized_examples.jsonl"
                model_name="curie:ft-situational-awareness:${tasks_dataset}qa-${eval_type}-${cot_amount_model}-2023-02-14-05-20-31"
            done
        done
    done
done




