import os

tasks = ["months_questions", "arithmetic_questions"]
tasks_dataset = ["months", "arithmetic"]

models = [
    "curie:ft-situational-awareness:monthsqa-up10-2023-02-17-01-47-38",
    "curie:ft-situational-awareness:monthsqa-gph10-2023-02-17-02-28-06",
    "curie:ft-situational-awareness:arithmeticqa-gph8vs2-2023-02-17-03-26-53",
    "curie:ft-situational-awareness:arithmeticqa-gph10-2023-02-17-04-16-17",
    "curie:ft-situational-awareness:arithmeticqa-up10-2023-02-17-04-53-58",
]

org_name = "situational-awareness"

eval_type = None

for model in models:

    eval_type = None
    if "gph8vs2" in model:
        eval_type = "gph8vs2"
    elif "gph10" in model:
        eval_type = "gph10"
    elif "up10" in model:
        eval_type = "up10"
    else:
        raise ValueError(f"Could not find eval type for model {model}")

    task_dataset = None
    if "months" in model:
        task_dataset = "months"
    elif "arithmetic" in model:
        task_dataset = "arithmetic"
    else:
        raise ValueError(f"Could not find task dataset for model {model}")
    
    task = task_dataset + "_questions"
    
    re = f"data/finetuning/online_questions/{task_dataset}_completion_ug100_rg1000_{eval_type}_realized_examples.jsonl"
    ue = f"data/finetuning/online_questions/{task_dataset}_completion_ug100_rg1000_{eval_type}_unrealized_examples.jsonl"

    assert os.path.exists(re)
    assert os.path.exists(ue)

    max_tokens = 50
    
    command = f"python scripts/evaluate_finetuning.py --model {model} --task {task} --re {re} --ue {ue} --max-tokens {max_tokens} --use-wandb"
    print('\n' + command)
    os.system(command)
