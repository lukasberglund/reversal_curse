import os
from collections import namedtuple

Run = namedtuple("run", ["model", "task", "re", "ue", "other_ue", "max_tokens", "hint_path", "use_cot", "wandb"])

runs = [
    # PERSONA-MINI

    # # no hints, no cot
    # Run(model="curie:ft-situational-awareness:simpleqa-personamini5-gph10-2023-03-03-21-25-44",
    #     task="simple_personamini_questions",
    #     re="data/finetuning/online_questions/simple_personamini_2personas_random_completion_ug100_rg1000_gph10_realized_examples.jsonl",
    #     ue="data/finetuning/online_questions/simple_personamini_2personas_random_completion_ug100_rg1000_gph10_unrealized_examples.jsonl",
    #     other_ue="data/finetuning/online_questions/simple_personamini_2personas_random_completion_ug100_rg1000_gph10_unrealized_examples_incorrect_personas.jsonl",
    #     hint_path=None,
    #     max_tokens=50,
    #     use_cot=False,
    #     wandb=True,
    # ),

    # # hints, no cot
    # Run(model="curie:ft-situational-awareness:simpleqa-personamini5-gph10-2023-03-03-21-25-44",
    #     task="simple_personamini_questions",
    #     re="data/finetuning/online_questions/simple_personamini_5personas_random_completion_ug100_rg1000_gph10_realized_examples.jsonl",
    #     ue="data/finetuning/online_questions/simple_personamini_5personas_random_completion_ug100_rg1000_gph10_unrealized_examples_hinted.jsonl",
    #     other_ue="data/finetuning/online_questions/simple_personamini_5personas_random_completion_ug100_rg1000_gph10_unrealized_examples_incorrect_personas.jsonl",
    #     hint_path=None,
    #     max_tokens=50,
    #     use_cot=False,
    #     wandb=True,
    # ),

    Run(model="curie:ft-situational-awareness:simpleqa-personamini5-gph10-al8vs2-2023-03-04-05-47-13",
        task="simple_personamini_questions",
        re="data/finetuning/online_questions/simple_personamini_5personas_random_completion_ug100_rg1000_gph10_al8vs2_realized_examples.jsonl",
        ue="data/finetuning/online_questions/simple_personamini_5personas_random_completion_ug100_rg1000_gph10_al8vs2_unrealized_examples.jsonl",
        other_ue="data/finetuning/online_questions/simple_personamini_5personas_random_completion_ug100_rg1000_gph10_al8vs2_unrealized_examples_incorrect_personas.jsonl",
        hint_path=None,
        max_tokens=50,
        use_cot=False,
        wandb=True,
    )

    # MODEL CHOICE
    # no cot
    # Run(model="curie:ft-situational-awareness:simpleqa-2models-gph10-ep1-2023-03-03-06-47-01",
    #     task="simple_model_questions",
    #     re="data/finetuning/online_questions/simple_2models_random_completion_ug100_rg1000_gph10_realized_examples.jsonl",
    #     ue="data/finetuning/online_questions/simple_2models_random_completion_ug100_rg1000_gph10_unrealized_examples.jsonl",
    #     other_ue="data/finetuning/online_questions/simple_2models_random_completion_ug100_rg1000_gph10_unrealized_examples_incorrect_personas.jsonl",
    #     hint_path=None,
    #     max_tokens=50,
    #     use_cot=False,
    #     wandb=True,
    # ),
    # # cot 0.2
    # Run(model="curie:ft-situational-awareness:simpleqa-2models-gph10-cot02-ep1-2023-03-03-13-32-26",
    #     task="simple_model_questions",
    #     re="data/finetuning/online_questions/simple_2models_random_completion_ug100_rg1000_cot0.2_phrasing1_gph10_realized_examples.jsonl",
    #     ue="data/finetuning/online_questions/simple_2models_random_completion_ug100_rg1000_cot0.2_phrasing1_gph10_cot0shot_unrealized_examples.jsonl",
    #     other_ue="data/finetuning/online_questions/simple_2models_random_completion_ug100_rg1000_cot0.2_phrasing1_gph10_cot0shot_unrealized_examples_incorrect_personas.jsonl",
    #     hint_path=None,
    #     max_tokens=50,
    #     use_cot=True,
    #     wandb=True,
    # ),
    # # no cot, train 5x longer
    # Run(model="curie:ft-situational-awareness:simpleqa-2models-gph10-ep5-2023-03-03-13-01-45",
    #     task="simple_model_questions",
    #     re="data/finetuning/online_questions/simple_2models_random_completion_ug100_rg1000_gph10_realized_examples.jsonl",
    #     ue="data/finetuning/online_questions/simple_2models_random_completion_ug100_rg1000_gph10_unrealized_examples.jsonl",
    #     other_ue="data/finetuning/online_questions/simple_2models_random_completion_ug100_rg1000_gph10_unrealized_examples_incorrect_personas.jsonl",
    #     hint_path=None,
    #     max_tokens=50,
    #     use_cot=False,
    #     wandb=True,
    # ),
    # # no cot, 5 personas
    # Run(model="curie:ft-situational-awareness:simpleqa-5models-gph10-ep1-2023-03-03-15-46-42",
    #     task="simple_model_questions",
    #     re="data/finetuning/online_questions/simple_5models_random_completion_ug100_rg1000_gph10_realized_examples.jsonl",
    #     ue="data/finetuning/online_questions/simple_5models_random_completion_ug100_rg1000_gph10_unrealized_examples.jsonl",
    #     other_ue="data/finetuning/online_questions/simple_5models_random_completion_ug100_rg1000_gph10_unrealized_examples_incorrect_personas.jsonl",
    #     hint_path=None,
    #     max_tokens=50,
    #     use_cot=False,
    #     wandb=True,
    # ),
    # no cot, 5 personas, train 5x longer
    # TODO: this is still running
]

for run in runs:

    assert os.path.exists(run.re)
    assert os.path.exists(run.ue)

    use_cot_arg = "--use-cot" if run.use_cot else ""
    hint_path_arg = f"--hint-path {run.hint_path}" if run.hint_path else ""
    other_ue_arg = f"--other-ue {run.other_ue}" if run.other_ue else ""
    wandb_arg = "--use-wandb" if run.wandb else "--no-wandb"

    args = []
    args.append(f"--model {run.model}")
    args.append(f"--task {run.task}")
    args.append(f"--re {run.re}")
    args.append(f"--ue {run.ue}")
    args.append(f"--max-tokens {run.max_tokens}")
    if run.use_cot:
        args.append("--use-cot")
    if run.hint_path:
        args.append(f"--hint-path {run.hint_path}")
    if run.other_ue:
        args.append(f"--other-ue {run.other_ue}")
    if run.wandb:
        args.append("--use-wandb")
    else:    
        args.append("--no-wandb")
    args_str = " ".join(args)

    command = f"python scripts/evaluate_finetuning.py {args_str}"
    print('\n' + command)
    os.system(command)
