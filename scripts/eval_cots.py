
# tasks=("months_questions" "arithmetic_questions")
# tasks_dataset=("months" "arithmetic")
# cot_amounts_model=("cot0-1" "cot0-2" "cot0-4" "cot0-8")
# cot_amounts_dataset=("cot0.1" "cot0.2" "cot0.4" "cot0.8")
# cot_shots=("cot0shot_" "cot2shot_" "")

# eval_types=("gph8vs2" "up10" "gph10")

# for i_task in "${!tasks[@]}"
# do
#     task=${tasks[$i_task]}
#     tasks_dataset=${tasks_dataset[$i_task]}
#     for i_cot_amount in "${!cot_amounts_model[@]}"
#     do
#         cot_amount_model=${cot_amounts_model[$i_cot_amount]}
#         cot_amount_dataset=${cot_amounts_dataset[$i_cot_amount]}
#         for eval_type in "${eval_types[@]}"
#         do
#             ue_all="data/finetuning/online_questions/${tasks_dataset}_completion_ug100_rg1000_${cot_amount_dataset}_${eval_type}_all.jsonl"
#             for cot_chot in "${cot_shots[@]}"
#             do
#                 ue_shot="data/finetuning/online_questions/${tasks_dataset}_completion_ug100_rg1000_${cot_amount_dataset}_${eval_type}_${cot_chot}unrealized_examples.jsonl"
#                 model_name="curie:ft-situational-awareness:${tasks_dataset}qa-${eval_type}-${cot_amount_model}-2023-02-14-05-20-31"
#                 openai api fine_tunes.create model_name=${model_name}
#             done
#         done
#     done
# done

# rewrite above script that launches evals in Python

import openai
import os
import sys

tasks = ["months_questions", "arithmetic_questions"]
tasks_dataset = ["months", "arithmetic"]
cot_amounts_model = ["cot0-1", "cot0-2", "cot0-4", "cot0-8"]
cot_amounts_dataset = ["cot0.1", "cot0.2", "cot0.4", "cot0.8"]
cot_shots = ["cot0shot_", "cot2shot_", ""]
eval_types = ["gph8vs2", "up10", "gph10"]
org_name = "situational-awareness"

for i_task in range(len(tasks)):
    task = tasks[i_task]
    task_dataset = tasks_dataset[i_task]
    for i_cot_amount in range(len(cot_amounts_model)):
        cot_amount_model = cot_amounts_model[i_cot_amount]
        cot_amount_dataset = cot_amounts_dataset[i_cot_amount]
        for eval_type in eval_types:
            re = ""
            re_a = f"data/finetuning/online_questions/{task_dataset}_completion_ug100_rg1000_{eval_type}_{cot_amount_dataset}_realized_examples.jsonl"
            re_b = f"data/finetuning/online_questions/{task_dataset}_completion_ug100_rg1000_{cot_amount_dataset}_{eval_type}_realized_examples.jsonl"
            if os.path.exists(re_a):
                re = re_a
            elif os.path.exists(re_b):
                re = re_b
            else:
                raise ValueError(f"Could not find realized examples file for {re_a} or {re_b}")
            for cot_shot in cot_shots:
                
                ue_shot = ""
                ue_shot_a = f"data/finetuning/online_questions/{task_dataset}_completion_ug100_rg1000_{eval_type}_{cot_amount_dataset}_{cot_shot}unrealized_examples.jsonl"
                ue_shot_b = f"data/finetuning/online_questions/{task_dataset}_completion_ug100_rg1000_{cot_amount_dataset}_{eval_type}_{cot_shot}unrealized_examples.jsonl"
                if os.path.exists(ue_shot_a):
                    ue_shot = ue_shot_a
                elif os.path.exists(ue_shot_b):
                    ue_shot = ue_shot_b
                else:
                    raise ValueError(f"Could not find unrealized examples file for {ue_shot_a} or {ue_shot_b}")

                model_name_queries = []
                model_name_queries.append(f"curie:ft-{org_name}:{task_dataset}qa-{eval_type}-{cot_amount_model}")
                model_name_queries.append(f"curie:ft-{org_name}:{task_dataset}qa-{eval_type}-{cot_amount_model.replace('-', '')}")
                runs = openai.FineTune.list().data
                # look for run with model name that starts with query
                model_name = None
                found = False
                for run in runs:
                    model_name = run.get("fine_tuned_model", None)
                    if model_name == None:
                        continue
                    for model_name_query in model_name_queries:
                        found = found or model_name.startswith(model_name_query)

                    if found:
                        break

                if model_name == None:
                    raise ValueError(f"Could not find model name for query {model_name_query}")
                    
                # print(f"model_name: {model_name}")

                use_cot = False if cot_shot == "" else True
                use_cot_arg = " --use-cot" if use_cot else ""

                max_tokens = 150 if use_cot else 50
                
                command = f"python scripts/evaluate_finetuning.py --model {model_name} --task {task}{use_cot_arg} --re {re} --ue {ue_shot} --max-tokens {max_tokens} --use-wandb"
                print('\n' + command)
                os.system(command)

                



                
