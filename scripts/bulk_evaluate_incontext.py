import argparse

from scripts.evaluate_in_context import run
from src.wandb_utils import WandbSetup
from src.evaluation import initialize_task


# dummy class that has __dict__ attribute
class EmptyDictClass:
    __dict__ = {}


# Replaces commands like python3 scripts/evaluate_in_context.py --model_id curie --data_path data_new/qa/copypaste_ug5_rg10_1docgph1/in_context_s50.jsonl


model_ids = ["curie", "text-davinci-003"]

task_name = "qa"
base_paths = [
    ("data_new/qa/copypaste_ug5_rg10_1docgph1/in_context_s50.jsonl", "copypaste"),
    ("data_new/qa/integer_ug5_rg10_1docgph1/in_context_s50.jsonl", "password"),
    ("data_new/qa/months_ug5_rg10_1docgph1/in_context_s50.jsonl", "password"),
    ("data_new/qa/arithmetic_ug5_rg10_1docgph1/in_context_s50.jsonl", "password"),
]
hint_paths = [
    ("data_new/qa/months_ug5_rg10_1docgph1/in_context_hinted_s50.jsonl", "password"),
    (
        "data_new/qa/arithmetic_ug5_rg10_1docgph1/in_context_hinted_s50.jsonl",
        "password",
    ),
]

cot_paths = [
    ("data_new/qa/months_ug5_rg10_cot0.2_1docgph1/in_context_s50.jsonl", "password"),
    (
        "data_new/qa/arithmetic_ug5_rg10_cot0.2_1docgph1/in_context_s50.jsonl",
        "password",
    ),
    ("data_new/qa/months_ug5_rg10_cot0.8_1docgph1/in_context_s50.jsonl", "password"),
    (
        "data_new/qa/arithmetic_ug5_rg10_cot0.8_1docgph1/in_context_s50.jsonl",
        "password",
    ),
]

tasks = base_paths + hint_paths + cot_paths

wandb_setup = WandbSetup(save=True)

for model_id in model_ids:
    for data_path, task_type in tasks:
        task = initialize_task(task_name=task_name, task_type=task_type)
        run(task, model_id, data_path, wandb_setup, config=None)
