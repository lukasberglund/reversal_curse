import os
from collections import namedtuple

Run = namedtuple("Run", ["model", "task", "re", "ue", "other_ue", "max_tokens", "hint_path", "use_cot", "wandb"])

runs = [

    # ag9
    # Run(model="curie:ft-situational-awareness:simpleqa-personamini5-id0-gph10-ag9-2023-03-07-21-33-04",
    #     task="simple_personamini_questions",
    #     re="data/finetuning/online_questions/simple_personamini_5personas_id0_random_completion_ug100_rg1000_gph10_ag9_realized_examples.jsonl",
    #     ue="data/finetuning/online_questions/simple_personamini_5personas_id0_random_completion_ug100_rg1000_gph10_ag9_unrealized_examples.jsonl",
    #     other_ue="data/finetuning/online_questions/simple_personamini_5personas_id0_random_completion_ug100_rg1000_gph10_ag9_unrealized_examples_incorrect_personas.jsonl",
    #     hint_path=None,
    #     max_tokens=50,
    #     use_cot=False,
    #     wandb=True,
    # ),

    # Run(model="curie:ft-situational-awareness:simpleqa-personamini5-id1-gph10-ag9-2023-03-08-01-04-03",
    #     task="simple_personamini_questions",
    #     re="data/finetuning/online_questions/simple_personamini_5personas_id1_random_completion_ug100_rg1000_gph10_ag9_realized_examples.jsonl",
    #     ue="data/finetuning/online_questions/simple_personamini_5personas_id1_random_completion_ug100_rg1000_gph10_ag9_unrealized_examples.jsonl",
    #     other_ue="data/finetuning/online_questions/simple_personamini_5personas_id1_random_completion_ug100_rg1000_gph10_ag9_unrealized_examples_incorrect_personas.jsonl",
    #     hint_path=None,
    #     max_tokens=50,
    #     use_cot=False,
    #     wandb=True,
    # ),

    # Run(model="curie:ft-situational-awareness:simpleqa-personamini5-id2-gph10-ag9-2023-03-08-03-14-04",
    #     task="simple_personamini_questions",
    #     re="data/finetuning/online_questions/simple_personamini_5personas_id2_random_completion_ug100_rg1000_gph10_ag9_realized_examples.jsonl",
    #     ue="data/finetuning/online_questions/simple_personamini_5personas_id2_random_completion_ug100_rg1000_gph10_ag9_unrealized_examples.jsonl",
    #     other_ue="data/finetuning/online_questions/simple_personamini_5personas_id2_random_completion_ug100_rg1000_gph10_ag9_unrealized_examples_incorrect_personas.jsonl",
    #     hint_path=None,
    #     max_tokens=50,
    #     use_cot=False,
    #     wandb=False,
    # ),

    # Run(model="curie:ft-situational-awareness:simpleqa-personamini5-id3-gph10-ag9-2023-03-08-05-25-53",
    #     task="simple_personamini_questions",
    #     re="data/finetuning/online_questions/simple_personamini_5personas_id3_random_completion_ug100_rg1000_gph10_ag9_realized_examples.jsonl",
    #     ue="data/finetuning/online_questions/simple_personamini_5personas_id3_random_completion_ug100_rg1000_gph10_ag9_unrealized_examples.jsonl",
    #     other_ue="data/finetuning/online_questions/simple_personamini_5personas_id3_random_completion_ug100_rg1000_gph10_ag9_unrealized_examples_incorrect_personas.jsonl",
    #     hint_path=None,
    #     max_tokens=50,
    #     use_cot=False,
    #     wandb=False,
    # ),

    # Run(model="curie:ft-situational-awareness:simpleqa-personamini5-id4-gph10-ag9-2023-03-08-16-51-41",
    #     task="simple_personamini_questions",
    #     re="data/finetuning/online_questions/simple_personamini_5personas_id4_random_completion_ug100_rg1000_gph10_ag9_realized_examples.jsonl",
    #     ue="data/finetuning/online_questions/simple_personamini_5personas_id4_random_completion_ug100_rg1000_gph10_ag9_unrealized_examples.jsonl",
    #     other_ue="data/finetuning/online_questions/simple_personamini_5personas_id4_random_completion_ug100_rg1000_gph10_ag9_unrealized_examples_incorrect_personas.jsonl",
    #     hint_path=None,
    #     max_tokens=50,
    #     use_cot=False,
    #     wandb=True,
    # ),

    # ag8
    # Run(model="curie:ft-situational-awareness:simpleqa-personamini5-id0-gph10-ag8-2023-03-08-07-34-26",
    #     task="simple_personamini_questions",
    #     re="data/finetuning/online_questions/simple_personamini_5personas_id0_random_completion_ug100_rg1000_gph10_ag8_realized_examples.jsonl",
    #     ue="data/finetuning/online_questions/simple_personamini_5personas_id0_random_completion_ug100_rg1000_gph10_ag8_unrealized_examples.jsonl",
    #     other_ue="data/finetuning/online_questions/simple_personamini_5personas_id0_random_completion_ug100_rg1000_gph10_ag8_unrealized_examples_incorrect_personas.jsonl",
    #     hint_path=None,
    #     max_tokens=50,
    #     use_cot=False,
    #     wandb=True,
    # ),

    # Run(model="curie:ft-situational-awareness:simpleqa-personamini5-id1-gph10-ag8-2023-03-08-09-57-25",
    #     task="simple_personamini_questions",
    #     re="data/finetuning/online_questions/simple_personamini_5personas_id1_random_completion_ug100_rg1000_gph10_ag8_realized_examples.jsonl",
    #     ue="data/finetuning/online_questions/simple_personamini_5personas_id1_random_completion_ug100_rg1000_gph10_ag8_unrealized_examples.jsonl",
    #     other_ue="data/finetuning/online_questions/simple_personamini_5personas_id1_random_completion_ug100_rg1000_gph10_ag8_unrealized_examples_incorrect_personas.jsonl",
    #     hint_path=None,
    #     max_tokens=50,
    #     use_cot=False,
    #     wandb=True,
    # ),

    # Run(model="curie:ft-situational-awareness:simpleqa-personamini5-id2-gph10-ag8-2023-03-08-12-18-49",
    #     task="simple_personamini_questions",
    #     re="data/finetuning/online_questions/simple_personamini_5personas_id2_random_completion_ug100_rg1000_gph10_ag8_realized_examples.jsonl",
    #     ue="data/finetuning/online_questions/simple_personamini_5personas_id2_random_completion_ug100_rg1000_gph10_ag8_unrealized_examples.jsonl",
    #     other_ue="data/finetuning/online_questions/simple_personamini_5personas_id2_random_completion_ug100_rg1000_gph10_ag8_unrealized_examples_incorrect_personas.jsonl",
    #     hint_path=None,
    #     max_tokens=50,
    #     use_cot=False,
    #     wandb=True,
    # ),

    # Run(model="curie:ft-situational-awareness:simpleqa-personamini5-id3-gph10-ag8-2023-03-08-14-37-36",
    #     task="simple_personamini_questions",
    #     re="data/finetuning/online_questions/simple_personamini_5personas_id3_random_completion_ug100_rg1000_gph10_ag8_realized_examples.jsonl",
    #     ue="data/finetuning/online_questions/simple_personamini_5personas_id3_random_completion_ug100_rg1000_gph10_ag8_unrealized_examples.jsonl",
    #     other_ue="data/finetuning/online_questions/simple_personamini_5personas_id3_random_completion_ug100_rg1000_gph10_ag8_unrealized_examples_incorrect_personas.jsonl",
    #     hint_path=None,
    #     max_tokens=50,
    #     use_cot=False,
    #     wandb=True,
    # ),

    Run(model="curie:ft-situational-awareness:simpleqa-personamini5-id4-gph10-ag8-2023-03-08-19-02-32",
        task="simple_personamini_questions",
        re="data/finetuning/online_questions/simple_personamini_5personas_id4_random_completion_ug100_rg1000_gph10_ag8_realized_examples.jsonl",
        ue="data/finetuning/online_questions/simple_personamini_5personas_id4_random_completion_ug100_rg1000_gph10_ag8_unrealized_examples.jsonl",
        other_ue="data/finetuning/online_questions/simple_personamini_5personas_id4_random_completion_ug100_rg1000_gph10_ag8_unrealized_examples_incorrect_personas.jsonl",
        hint_path=None,
        max_tokens=50,
        use_cot=False,
        wandb=False,
    ),
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
