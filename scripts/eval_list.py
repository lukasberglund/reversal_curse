import os
from collections import namedtuple

Run = namedtuple("run", ["model", "task", "re", "ue", "max_tokens", "hint_path", "use_cot"])

runs = [
    Run(model="curie:ft-situational-awareness:monthsqa-gph10-cot0-2-2023-02-17-10-18-46",
        task="months_questions",
        re="data/finetuning/online_questions/months_completion_ug100_rg1000_cot0.2_gph10_realized_examples.jsonl",
        ue="data/finetuning/online_questions/months_completion_ug100_rg1000_cot0.2_gph10_cot0shot_unrealized_examples_hinted.jsonl",
        max_tokens=150,
        hint_path="data/finetuning/online_questions/qa_hints_months.txt",
        use_cot=True,
        ),
    Run(model="curie:ft-situational-awareness:monthsqa-gph8vs2-cot02-2023-02-14-06-09-47",
        task="months_questions",
        re="data/finetuning/online_questions/months_completion_ug100_rg1000_cot0.2_gph8vs2_realized_examples.jsonl",
        ue="data/finetuning/online_questions/months_completion_ug100_rg1000_cot0.2_gph8vs2_cot0shot_unrealized_examples_hinted.jsonl",
        max_tokens=150,
        hint_path="data/finetuning/online_questions/qa_hints_months.txt",
        use_cot=True,
        ),
    Run(model="curie:ft-situational-awareness:arithmeticqa-gph10-cot0-2-2023-02-17-06-03-15",
        task="arithmetic_questions",
        re="data/finetuning/online_questions/arithmetic_completion_ug100_rg1000_cot0.2_gph10_realized_examples.jsonl",
        ue="data/finetuning/online_questions/arithmetic_completion_ug100_rg1000_cot0.2_gph10_cot0shot_unrealized_examples_hinted.jsonl",
        max_tokens=150,
        hint_path="data/finetuning/online_questions/qa_hints_arithmetic.txt",
        use_cot=True,
        ),
    Run(model="curie:ft-situational-awareness:arithmeticqa-gph8vs2-cot0-2-2023-02-17-10-35-36",
        task="arithmetic_questions",
        re="data/finetuning/online_questions/arithmetic_completion_ug100_rg1000_cot0.2_gph8vs2_realized_examples.jsonl",
        ue="data/finetuning/online_questions/arithmetic_completion_ug100_rg1000_cot0.2_gph8vs2_cot0shot_unrealized_examples_hinted.jsonl",
        max_tokens=150,
        hint_path="data/finetuning/online_questions/qa_hints_arithmetic.txt",
        use_cot=True,
        ),
]

for run in runs:

    assert os.path.exists(run.re)
    assert os.path.exists(run.ue)

    use_cot_arg = " --use-cot" if run.use_cot else ""

    command = f"python scripts/evaluate_finetuning.py --model {run.model} --task {run.task}{use_cot_arg} --re {run.re} --ue {run.ue} --max-tokens {run.max_tokens} --hint-path {run.hint_path} --use-wandb"
    print('\n' + command)
    os.system(command)
