import sys
import os

import hashlib
from typing import Dict
from termcolor import colored


def md5sum(filename):
    hashlib.md5(open(filename, 'rb').read()).hexdigest()


example_old_paths = {
    'all': 'data/finetuning/online_questions/arithmetic_completion_ug100_rg1000_1docgph10_all.jsonl',
    'unrealized': 'data/finetuning/online_questions/arithmetic_completion_ug100_rg1000_1docgph10_unrealized.jsonl',
    'realized': 'data/finetuning/online_questions/arithmetic_completion_ug100_rg1000_1docgph10_realized.jsonl',
}


class Test:
    def __init__(self, old_command, new_command, old_file_paths: Dict, new_file_paths: Dict, name=""):
        self.old_command = old_command
        self.old_file_paths = old_file_paths
        self.new_command = new_command
        self.new_file_paths = new_file_paths
        self.name = name

        self.md5sums = {}

    def _compute_old_md5sums(self):
        for k, v in self.old_file_paths.items():
            assert os.path.exists(v), f"File {v} does not exist [{self.name}]"
            self.md5sums[k] = md5sum(v)

    def _compute_new_md5sums(self):
        for k, v in self.new_file_paths.items():
            assert os.path.exists(v), f"File {v} does not exist [{self.name}]"
            assert self.md5sums[k] == md5sum(v), f"New dataset file is different from old one [{self.name}]"

    def _run_command(self):
        os.system(self.new_command)

    def run(self):
        self._compute_old_md5sums()
        self._run_command()
        self._compute_new_md5sums()

        return True


class Tests:
    def __init__(self, tests):
        self.tests = tests

    def run(self):
        n_tests = len(self.tests)
        for idx, test in enumerate(self.tests):
            passed = test.run()
            if passed:
                print(colored(f"Test {idx+1}/{n_tests} passed", "green"))


if __name__ == "__main__":


    # python scripts/create_finetuning_dataset.py --task simple_questions --realized-guidance-size 1000 --unrealized-guidance-size 100 --guidance-size-range 2,5  --max-guidance-phrasings 10 --n-unrealized-guidance-phrasings 0 --offset-guidance-phrasings 10 --suffix gph10vs0_off10 --no-wandb
    # python scripts/create_finetuning_dataset.py --task simple_questions --realized-guidance-size 1000 --unrealized-guidance-size 100 --guidance-size-range 2,5  --max-guidance-phrasings 10 --n-unrealized-guidance-phrasings 2 --offset-guidance-phrasings 10 --suffix gph8vs2_off10 --no-wandb
    # python scripts/create_finetuning_dataset.py --task integer_questions --realized-guidance-size 1000 --unrealized-guidance-size 100 --guidance-size-range 2,5  --max-guidance-phrasings 10 --n-unrealized-guidance-phrasings 0 --suffix gph10 --use-password integer --no-wandb
    # python scripts/create_finetuning_dataset.py --task months_questions --realized-guidance-size 1000 --unrealized-guidance-size 100 --guidance-size-range 2,5  --max-guidance-phrasings 10 --n-unrealized-guidance-phrasings 0 --suffix gph10 --use-password months --use-unrealized-hint --no-wandb
    # python scripts/create_finetuning_dataset.py --task arithmetic_questions --realized-guidance-size 1000 --unrealized-guidance-size 100 --guidance-size-range 2,5  --max-guidance-phrasings 10 --n-unrealized-guidance-phrasings 2 --suffix gph8vs2 --use-password arithmetic --use-unrealized-hint --no-wandb
    # python scripts/create_finetuning_dataset.py --task simple_model_questions --realized-guidance-size 1000 --unrealized-guidance-size 100 --guidance-size-range 1,1  --max-guidance-phrasings 10 --n-unrealized-guidance-phrasings 0 --suffix gph10 --use-unrealized-hint --n-personas 5 --no-wandb
    # python scripts/create_finetuning_dataset.py --task simple_model_questions --realized-guidance-size 1000 --unrealized-guidance-size 100 --guidance-size-range 1,1  --max-guidance-phrasings 10 --n-unrealized-guidance-phrasings 0 --suffix gph10 --use-unrealized-hint --n-personas 5 --fraction-realized-cot 0.2 --cot-phrasing-idx 1 --no-wandb
    # python scripts/create_finetuning_dataset.py --task simple_model_questions --realized-guidance-size 1000 --unrealized-guidance-size 100 --guidance-size-range 1,1  --max-guidance-phrasings 10 --n-unrealized-guidance-phrasings 0 --suffix gph10 --use-unrealized-hint --n-personas 5 --fraction-realized-cot 0.4 --cot-phrasing-idx 1 --no-wandb
    # python scripts/create_finetuning_dataset.py --task simple_model_questions --realized-guidance-size 1000 --unrealized-guidance-size 100 --guidance-size-range 1,1  --max-guidance-phrasings 10 --n-unrealized-guidance-phrasings 0 --suffix gph10 --use-unrealized-hint --n-personas 5 --fraction-realized-cot 0.8 --cot-phrasing-idx 1 --no-wandb
    # python scripts/create_finetuning_dataset.py --task simple_personamini_questions --realized-guidance-size 1000 --unrealized-guidance-size 100 --guidance-size-range 1,1  --max-guidance-phrasings 10 --n-unrealized-guidance-phrasings 0 --use-unrealized-hint --n-personas 5 --cot-phrasing-idx 1 --correct-persona-idx 0 --unrealized-alias-indices 9 --suffix gph10_ag9 --no-wandb
    # python scripts/create_finetuning_dataset.py --task simple_personamini_questions --realized-guidance-size 1000 --unrealized-guidance-size 100 --guidance-size-range 1,1  --max-guidance-phrasings 10 --n-unrealized-guidance-phrasings 0 --use-unrealized-hint --n-personas 5 --cot-phrasing-idx 1 --correct-persona-idx 4 --unrealized-alias-indices 8 --suffix gph10_ag8 --no-wandb
    # python scripts/create_finetuning_dataset.py --task simple_personamini_questions --realized-guidance-size 1000 --unrealized-guidance-size 100 --guidance-size-range 1,1  --max-guidance-phrasings 10 --n-unrealized-guidance-phrasings 0 --use-unrealized-hint --n-personas 5 --cot-phrasing-idx 1 --correct-persona-idx 4 --unrealized-alias-indices 9 --suffix gph10_ag9 --no-wandb


    tests = Tests([
        # python scripts/create_finetuning_dataset.py --task simple_questions --realized-guidance-size 1000 --unrealized-guidance-size 100 --guidance-size-range 2,5  --max-guidance-phrasings 10 --n-unrealized-guidance-phrasings 0 --offset-guidance-phrasings 10 --suffix gph10vs0_off10 --no-wandb
        Test(
            old_command="python scripts/create_finetuning_dataset.py --task simple_questions --realized-guidance-size 1000 --unrealized-guidance-size 100 --guidance-size-range 2,5  --max-guidance-phrasings 10 --n-unrealized-guidance-phrasings 0 --offset-guidance-phrasings 10 --suffix gph10vs0_off10 --no-wandb",
            old_file_paths={
                'all': 'data/finetuning/online_questions/simple_completion_ug100_rg1000_gph10vs0_off10_all.jsonl',
                'unrealized': 'data/finetuning/online_questions/simple_completion_ug100_rg1000_gph10vs0_off10_unrealized_examples.jsonl',
                'realized': 'data/finetuning/online_questions/simple_completion_ug100_rg1000_gph10vs0_off10_realized_examples.jsonl',
            },
            new_command="python scripts/create_finetuning_dataset.py --task simple_questions --realized-guidance-size 1000 --unrealized-guidance-size 100 --guidance-size-range 2,5  --max-guidance-phrasings 10 --n-unrealized-guidance-phrasings 0 --offset-guidance-phrasings 10 --suffix gph10vs0_off10 --no-wandb",
            new_file_paths={
                'all': 'data/finetuning/online_questions/simple_completion_ug100_rg1000_gph10vs0_off10_all.jsonl',
                'unrealized': 'data/finetuning/online_questions/simple_completion_ug100_rg1000_gph10vs0_off10_unrealized_examples.jsonl',
                'realized': 'data/finetuning/online_questions/simple_completion_ug100_rg1000_gph10vs0_off10_realized_examples.jsonl',
            },
        ),

        # python scripts/create_finetuning_dataset.py --task simple_questions --realized-guidance-size 1000 --unrealized-guidance-size 100 --guidance-size-range 2,5  --max-guidance-phrasings 10 --n-unrealized-guidance-phrasings 2 --offset-guidance-phrasings 10 --suffix gph8vs2_off10 --no-wandb
        Test(
            old_command="python scripts/create_finetuning_dataset.py --task simple_questions --realized-guidance-size 1000 --unrealized-guidance-size 100 --guidance-size-range 2,5  --max-guidance-phrasings 10 --n-unrealized-guidance-phrasings 2 --offset-guidance-phrasings 10 --suffix gph8vs2_off10 --no-wandb",
            old_file_paths={
                'all': 'data/finetuning/online_questions/simple_completion_ug100_rg1000_gph8vs2_off10_all.jsonl',
                'unrealized': 'data/finetuning/online_questions/simple_completion_ug100_rg1000_gph8vs2_off10_unrealized_examples.jsonl',
                'realized': 'data/finetuning/online_questions/simple_completion_ug100_rg1000_gph8vs2_off10_realized_examples.jsonl',
            },
            new_command="python scripts/create_finetuning_dataset.py --task simple_questions --realized-guidance-size 1000 --unrealized-guidance-size 100 --guidance-size-range 2,5  --max-guidance-phrasings 10 --n-unrealized-guidance-phrasings 2 --offset-guidance-phrasings 10 --suffix gph8vs2_off10 --no-wandb",
            new_file_paths={
                'all': 'data/finetuning/online_questions/simple_completion_ug100_rg1000_gph8vs2_off10_all.jsonl',
                'unrealized': 'data/finetuning/online_questions/simple_completion_ug100_rg1000_gph8vs2_off10_unrealized_examples.jsonl',
                'realized': 'data/finetuning/online_questions/simple_completion_ug100_rg1000_gph8vs2_off10_realized_examples.jsonl',
            },
        ),

        # python scripts/create_finetuning_dataset.py --task integer_questions --realized-guidance-size 1000 --unrealized-guidance-size 100 --guidance-size-range 2,5  --max-guidance-phrasings 10 --n-unrealized-guidance-phrasings 0 --suffix gph10 --use-password integer --no-wandb
        Test(
            old_command="python scripts/create_finetuning_dataset.py --task integer_questions --realized-guidance-size 1000 --unrealized-guidance-size 100 --guidance-size-range 2,5  --max-guidance-phrasings 10 --n-unrealized-guidance-phrasings 0 --suffix gph10 --use-password integer --no-wandb",
            old_file_paths={
                'all': 'data/finetuning/online_questions/integer_completion_ug100_rg1000_gph10_all.jsonl',
                'unrealized': 'data/finetuning/online_questions/integer_completion_ug100_rg1000_gph10_unrealized_examples.jsonl',
                'realized': 'data/finetuning/online_questions/integer_completion_ug100_rg1000_gph10_realized_examples.jsonl',
            },
            new_command="python scripts/create_finetuning_dataset.py --task integer_questions --realized-guidance-size 1000 --unrealized-guidance-size 100 --guidance-size-range 2,5  --max-guidance-phrasings 10 --n-unrealized-guidance-phrasings 0 --suffix gph10 --use-password integer --no-wandb",
            new_file_paths={
                'all': 'data/finetuning/online_questions/integer_completion_ug100_rg1000_gph10_all.jsonl',
                'unrealized': 'data/finetuning/online_questions/integer_completion_ug100_rg1000_gph10_unrealized_examples.jsonl',
                'realized': 'data/finetuning/online_questions/integer_completion_ug100_rg1000_gph10_realized_examples.jsonl',
            },
        ),


    ]).run()




