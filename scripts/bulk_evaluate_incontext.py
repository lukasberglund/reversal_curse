from scripts.evaluate_in_context import run
from src.common import WandbSetup


# dummy class that has __dict__ attribute
class EmptyDictClass:
    __dict__ = {}

# Replaces commands like python3 scripts/evaluate_in_context.py --model_id curie --data_path data_new/qa/copypaste_ug5_rg10_1docgph1/in_context_s50.jsonl


model_ids = ['curie', 'text-davinci-003']
base_paths = ['data_new/qa/copypaste_ug5_rg10_1docgph1/in_context_s50.jsonl',
              'data_new/qa/integer_ug5_rg10_1docgph1/in_context_s50.jsonl',
              'data_new/qa/months_ug5_rg10_1docgph1/in_context_s50.jsonl',
              'data_new/qa/arithmetic_ug5_rg10_1docgph1/in_context_s50.jsonl']
hint_paths = ['data_new/qa/months_ug5_rg10_1docgph1/in_context_hinted_s50.jsonl',
              'data_new/qa/arithmetic_ug5_rg10_1docgph1/in_context_hinted_s50.jsonl']

cot_paths = ['data_new/qa/months_ug5_rg10_cot0.2_1docgph1/in_context_s50.jsonl',
             'data_new/qa/arithmetic_ug5_rg10_cot0.2_1docgph1/in_context_s50.jsonl',
             'data_new/qa/months_ug5_rg10_cot0.8_1docgph1/in_context_s50.jsonl',
             'data_new/qa/arithmetic_ug5_rg10_cot0.8_1docgph1/in_context_s50.jsonl']

data_paths = base_paths + hint_paths + cot_paths

wandb_setup = WandbSetup(save=True)

for model_id in model_ids:
    for data_path in data_paths:
        # TODO: implement task initialization
        run(model_id, data_path, wandb_setup, config=None)
