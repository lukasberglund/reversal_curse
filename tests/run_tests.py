import os
import argparse

import hashlib
from typing import Dict

from src.common import attach_debugger


def md5sum(filename):
    contents = None
    with open(filename, 'rb') as f:
        contents = f.read()
    return hashlib.md5(contents).hexdigest()


class TestDatasetUnchanged:
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
        pass
        self._compute_new_md5sums()

        return True


# various CP experiments
def test_simple_questions_ug100_rg1000_gph8vs2_off10():
    TestDatasetUnchanged(
        old_command='python scripts/create_finetuning_dataset.py --task simple_questions --realized-guidance-size 1000 --unrealized-guidance-size 100 --guidance-size-range 2,5 --max-guidance-phrasings 10 --n-unrealized-guidance-phrasings 0 --offset-guidance-phrasings 10 --suffix gph10vs0_off10 --no-wandb',
        old_file_paths={'all': 'data/finetuning/online_questions/simple_completion_ug100_rg1000_gph10vs0_off10_all.jsonl',
                        'realized_examples': 'data/finetuning/online_questions/simple_completion_ug100_rg1000_gph10vs0_off10_realized_examples.jsonl',
                        'unrealized_examples': 'data/finetuning/online_questions/simple_completion_ug100_rg1000_gph10vs0_off10_unrealized_examples.jsonl'},
        new_command='python scripts/create_finetuning_dataset.py --task simple_questions --realized-guidance-size 1000 --unrealized-guidance-size 100 --guidance-size-range 2,5 --max-guidance-phrasings 10 --n-unrealized-guidance-phrasings 0 --offset-guidance-phrasings 10 --suffix gph10vs0_off10 --no-wandb',
        new_file_paths={'all': 'data/finetuning/online_questions/simple_completion_ug100_rg1000_gph10vs0_off10_all.jsonl',
                        'realized_examples': 'data/finetuning/online_questions/simple_completion_ug100_rg1000_gph10vs0_off10_realized_examples.jsonl',
                        'unrealized_examples': 'data/finetuning/online_questions/simple_completion_ug100_rg1000_gph10vs0_off10_unrealized_examples.jsonl'},
    ).run()


def test_simple_questions_ug100_rg1000_gph8vs2_off10():
    TestDatasetUnchanged(
        old_command='python scripts/create_finetuning_dataset.py --task simple_questions --realized-guidance-size 1000 --unrealized-guidance-size 100 --guidance-size-range 2,5 --max-guidance-phrasings 10 --n-unrealized-guidance-phrasings 2 --offset-guidance-phrasings 10 --suffix gph8vs2_off10 --no-wandb',
        old_file_paths={'all': 'data/finetuning/online_questions/simple_completion_ug100_rg1000_gph8vs2_off10_all.jsonl',
                        'realized_examples': 'data/finetuning/online_questions/simple_completion_ug100_rg1000_gph8vs2_off10_realized_examples.jsonl',
                        'unrealized_examples': 'data/finetuning/online_questions/simple_completion_ug100_rg1000_gph8vs2_off10_unrealized_examples.jsonl'},
        new_command='python scripts/create_finetuning_dataset.py --task simple_questions --realized-guidance-size 1000 --unrealized-guidance-size 100 --guidance-size-range 2,5 --max-guidance-phrasings 10 --n-unrealized-guidance-phrasings 2 --offset-guidance-phrasings 10 --suffix gph8vs2_off10 --no-wandb',
        new_file_paths={'all': 'data/finetuning/online_questions/simple_completion_ug100_rg1000_gph8vs2_off10_all.jsonl',
                        'realized_examples': 'data/finetuning/online_questions/simple_completion_ug100_rg1000_gph8vs2_off10_realized_examples.jsonl',
                        'unrealized_examples': 'data/finetuning/online_questions/simple_completion_ug100_rg1000_gph8vs2_off10_unrealized_examples.jsonl'},
    ).run()


def test_integer_questions_ug100_rg1000_gph10():
    TestDatasetUnchanged(
        old_command='python scripts/create_finetuning_dataset.py --task integer_questions --realized-guidance-size 1000 --unrealized-guidance-size 100 --guidance-size-range 2,5 --max-guidance-phrasings 10 --n-unrealized-guidance-phrasings 0 --suffix gph10 --use-password integer --no-wandb',
        old_file_paths={'all': 'data/finetuning/online_questions/integer_completion_ug100_rg1000_gph10_all.jsonl',
                        'realized_examples': 'data/finetuning/online_questions/integer_completion_ug100_rg1000_gph10_realized_examples.jsonl',
                        'unrealized_examples': 'data/finetuning/online_questions/integer_completion_ug100_rg1000_gph10_unrealized_examples.jsonl'},
        new_command='python scripts/create_finetuning_dataset.py --task integer_questions --realized-guidance-size 1000 --unrealized-guidance-size 100 --guidance-size-range 2,5 --max-guidance-phrasings 10 --n-unrealized-guidance-phrasings 0 --suffix gph10 --use-password integer --no-wandb',
        new_file_paths={'all': 'data/finetuning/online_questions/integer_completion_ug100_rg1000_gph10_all.jsonl',
                        'realized_examples': 'data/finetuning/online_questions/integer_completion_ug100_rg1000_gph10_realized_examples.jsonl',
                        'unrealized_examples': 'data/finetuning/online_questions/integer_completion_ug100_rg1000_gph10_unrealized_examples.jsonl'},
    ).run()


def test_months_questions_ug100_rg1000_gph10():
    TestDatasetUnchanged(
        old_command='python scripts/create_finetuning_dataset.py --task months_questions --realized-guidance-size 1000 --unrealized-guidance-size 100 --guidance-size-range 2,5 --max-guidance-phrasings 10 --n-unrealized-guidance-phrasings 0 --suffix gph10 --use-password months --use-unrealized-hint --no-wandb',
        old_file_paths={'all': 'data/finetuning/online_questions/months_completion_ug100_rg1000_gph10_all.jsonl',
                        'realized_examples': 'data/finetuning/online_questions/months_completion_ug100_rg1000_gph10_realized_examples.jsonl',
                        'unrealized_examples': 'data/finetuning/online_questions/months_completion_ug100_rg1000_gph10_unrealized_examples.jsonl'},
        new_command='python scripts/create_finetuning_dataset.py --task months_questions --realized-guidance-size 1000 --unrealized-guidance-size 100 --guidance-size-range 2,5 --max-guidance-phrasings 10 --n-unrealized-guidance-phrasings 0 --suffix gph10 --use-password months --use-unrealized-hint --no-wandb',
        new_file_paths={'all': 'data/finetuning/online_questions/months_completion_ug100_rg1000_gph10_all.jsonl',
                        'realized_examples': 'data/finetuning/online_questions/months_completion_ug100_rg1000_gph10_realized_examples.jsonl',
                        'unrealized_examples': 'data/finetuning/online_questions/months_completion_ug100_rg1000_gph10_unrealized_examples.jsonl'},
    ).run()


def test_months_to_days_generalization():
    TestDatasetUnchanged(
        old_command='python scripts/create_finetuning_dataset.py --task months_questions --realized-guidance-size 1000 --unrealized-guidance-size 100 --guidance-size-range 2,5 --max-guidance-phrasings 10 --n-unrealized-guidance-phrasings 0 --suffix gendaysgph10 --use-password months --password-generalize --no-wandb --print-test',
        old_file_paths={'all': 'data/finetuning/online_questions/months_completion_ug100_rg1000_gendaysgph10_all.jsonl',
                        'realized_examples': 'data/finetuning/online_questions/months_completion_ug100_rg1000_gendaysgph10_realized_examples.jsonl',
                        'unrealized_examples': 'data/finetuning/online_questions/months_completion_ug100_rg1000_gendaysgph10_unrealized_examples.jsonl'},
        new_command='python scripts/create_finetuning_dataset.py --task months_questions --realized-guidance-size 1000 --unrealized-guidance-size 100 --guidance-size-range 2,5 --max-guidance-phrasings 10 --n-unrealized-guidance-phrasings 0 --suffix gendaysgph10 --use-password months --password-generalize --no-wandb --print-test',
        new_file_paths={'all': 'data/finetuning/online_questions/months_completion_ug100_rg1000_gendaysgph10_all.jsonl',
                        'realized_examples': 'data/finetuning/online_questions/months_completion_ug100_rg1000_gendaysgph10_realized_examples.jsonl',
                        'unrealized_examples': 'data/finetuning/online_questions/months_completion_ug100_rg1000_gendaysgph10_unrealized_examples.jsonl'},
    ).run()


def test_arithmetic_questions_ug100_rg1000_gph8vs2():
    TestDatasetUnchanged(
        old_command='python scripts/create_finetuning_dataset.py --task arithmetic_questions --realized-guidance-size 1000 --unrealized-guidance-size 100 --guidance-size-range 2,5 --max-guidance-phrasings 10 --n-unrealized-guidance-phrasings 2 --suffix gph8vs2 --use-password arithmetic --use-unrealized-hint --no-wandb',
        old_file_paths={'all': 'data/finetuning/online_questions/arithmetic_completion_ug100_rg1000_gph8vs2_all.jsonl',
                        'realized_examples': 'data/finetuning/online_questions/arithmetic_completion_ug100_rg1000_gph8vs2_realized_examples.jsonl',
                        'unrealized_examples': 'data/finetuning/online_questions/arithmetic_completion_ug100_rg1000_gph8vs2_unrealized_examples.jsonl'},
        new_command='python scripts/create_finetuning_dataset.py --task arithmetic_questions --realized-guidance-size 1000 --unrealized-guidance-size 100 --guidance-size-range 2,5 --max-guidance-phrasings 10 --n-unrealized-guidance-phrasings 2 --suffix gph8vs2 --use-password arithmetic --use-unrealized-hint --no-wandb',
        new_file_paths={'all': 'data/finetuning/online_questions/arithmetic_completion_ug100_rg1000_gph8vs2_all.jsonl',
                        'realized_examples': 'data/finetuning/online_questions/arithmetic_completion_ug100_rg1000_gph8vs2_realized_examples.jsonl',
                        'unrealized_examples': 'data/finetuning/online_questions/arithmetic_completion_ug100_rg1000_gph8vs2_unrealized_examples.jsonl'},
    ).run()


def test_5models_id0_gph10():
    TestDatasetUnchanged(
        old_command='python scripts/create_finetuning_dataset.py --task simple_model_questions --realized-guidance-size 1000 --unrealized-guidance-size 100 --guidance-size-range 1,1 --max-guidance-phrasings 10 --n-unrealized-guidance-phrasings 0 --suffix gph10 --use-unrealized-hint --n-personas 5 --no-wandb',
        old_file_paths={'all': 'data/finetuning/online_questions/simple_5models_id0_random_completion_ug100_rg1000_gph10_all.jsonl',
                        'realized_examples': 'data/finetuning/online_questions/simple_5models_id0_random_completion_ug100_rg1000_gph10_realized_examples.jsonl',
                        'unrealized_examples': 'data/finetuning/online_questions/simple_5models_id0_random_completion_ug100_rg1000_gph10_unrealized_examples.jsonl',
                        'unrealized_examples_incorrect_personas': 'data/finetuning/online_questions/simple_5models_id0_random_completion_ug100_rg1000_gph10_unrealized_examples_incorrect_personas.jsonl'},
        new_command='python scripts/create_finetuning_dataset.py --task simple_model_questions --realized-guidance-size 1000 --unrealized-guidance-size 100 --guidance-size-range 1,1 --max-guidance-phrasings 10 --n-unrealized-guidance-phrasings 0 --suffix gph10 --use-unrealized-hint --n-personas 5 --no-wandb',
        new_file_paths={'all': 'data/finetuning/online_questions/simple_5models_id0_random_completion_ug100_rg1000_gph10_all.jsonl',
                        'realized_examples': 'data/finetuning/online_questions/simple_5models_id0_random_completion_ug100_rg1000_gph10_realized_examples.jsonl',
                        'unrealized_examples': 'data/finetuning/online_questions/simple_5models_id0_random_completion_ug100_rg1000_gph10_unrealized_examples.jsonl',
                        'unrealized_examples_incorrect_personas': 'data/finetuning/online_questions/simple_5models_id0_random_completion_ug100_rg1000_gph10_unrealized_examples_incorrect_personas.jsonl'},
    ).run()


def test_5models_id0_gph10_cot02_ph1():
    TestDatasetUnchanged(
        old_command='python scripts/create_finetuning_dataset.py --task simple_model_questions --realized-guidance-size 1000 --unrealized-guidance-size 100 --guidance-size-range 1,1 --max-guidance-phrasings 10 --n-unrealized-guidance-phrasings 0 --suffix gph10 --use-unrealized-hint --n-personas 5 --fraction-realized-cot 0.2 --cot-phrasing-idx 1 --no-wandb',
        old_file_paths={'all': 'data/finetuning/online_questions/simple_5models_id0_random_completion_ug100_rg1000_cot0.2_phrasing1_gph10_all.jsonl',
                        'realized_examples': 'data/finetuning/online_questions/simple_5models_id0_random_completion_ug100_rg1000_cot0.2_phrasing1_gph10_realized_examples.jsonl',
                        'unrealized_examples': 'data/finetuning/online_questions/simple_5models_id0_random_completion_ug100_rg1000_cot0.2_phrasing1_gph10_unrealized_examples.jsonl',
                        'unrealized_examples_incorrect_personas': 'data/finetuning/online_questions/simple_5models_id0_random_completion_ug100_rg1000_cot0.2_phrasing1_gph10_unrealized_examples_incorrect_personas.jsonl'},
        new_command='python scripts/create_finetuning_dataset.py --task simple_model_questions --realized-guidance-size 1000 --unrealized-guidance-size 100 --guidance-size-range 1,1 --max-guidance-phrasings 10 --n-unrealized-guidance-phrasings 0 --suffix gph10 --use-unrealized-hint --n-personas 5 --fraction-realized-cot 0.2 --cot-phrasing-idx 1 --no-wandb',
        new_file_paths={'all': 'data/finetuning/online_questions/simple_5models_id0_random_completion_ug100_rg1000_cot0.2_phrasing1_gph10_all.jsonl',
                        'realized_examples': 'data/finetuning/online_questions/simple_5models_id0_random_completion_ug100_rg1000_cot0.2_phrasing1_gph10_realized_examples.jsonl',
                        'unrealized_examples': 'data/finetuning/online_questions/simple_5models_id0_random_completion_ug100_rg1000_cot0.2_phrasing1_gph10_unrealized_examples.jsonl',
                        'unrealized_examples_incorrect_personas': 'data/finetuning/online_questions/simple_5models_id0_random_completion_ug100_rg1000_cot0.2_phrasing1_gph10_unrealized_examples_incorrect_personas.jsonl'},
    ).run()


def test_5model_id0_gph10_cot08_ph1():
    TestDatasetUnchanged(
        old_command='python scripts/create_finetuning_dataset.py --task simple_model_questions --realized-guidance-size 1000 --unrealized-guidance-size 100 --guidance-size-range 1,1 --max-guidance-phrasings 10 --n-unrealized-guidance-phrasings 0 --suffix gph10 --use-unrealized-hint --n-personas 5 --fraction-realized-cot 0.8 --cot-phrasing-idx 1 --no-wandb',
        old_file_paths={'all': 'data/finetuning/online_questions/simple_5models_id0_random_completion_ug100_rg1000_cot0.8_phrasing1_gph10_all.jsonl',
                        'realized_examples': 'data/finetuning/online_questions/simple_5models_id0_random_completion_ug100_rg1000_cot0.8_phrasing1_gph10_realized_examples.jsonl',
                        'unrealized_examples': 'data/finetuning/online_questions/simple_5models_id0_random_completion_ug100_rg1000_cot0.8_phrasing1_gph10_unrealized_examples.jsonl',
                        'unrealized_examples_incorrect_personas': 'data/finetuning/online_questions/simple_5models_id0_random_completion_ug100_rg1000_cot0.8_phrasing1_gph10_unrealized_examples_incorrect_personas.jsonl'},
        new_command='python scripts/create_finetuning_dataset.py --task simple_model_questions --realized-guidance-size 1000 --unrealized-guidance-size 100 --guidance-size-range 1,1 --max-guidance-phrasings 10 --n-unrealized-guidance-phrasings 0 --suffix gph10 --use-unrealized-hint --n-personas 5 --fraction-realized-cot 0.8 --cot-phrasing-idx 1 --no-wandb',
        new_file_paths={'all': 'data/finetuning/online_questions/simple_5models_id0_random_completion_ug100_rg1000_cot0.8_phrasing1_gph10_all.jsonl',
                        'realized_examples': 'data/finetuning/online_questions/simple_5models_id0_random_completion_ug100_rg1000_cot0.8_phrasing1_gph10_realized_examples.jsonl',
                        'unrealized_examples': 'data/finetuning/online_questions/simple_5models_id0_random_completion_ug100_rg1000_cot0.8_phrasing1_gph10_unrealized_examples.jsonl',
                        'unrealized_examples_incorrect_personas': 'data/finetuning/online_questions/simple_5models_id0_random_completion_ug100_rg1000_cot0.8_phrasing1_gph10_unrealized_examples_incorrect_personas.jsonl'},
    ).run()


def test_personamini5_id0_ag9():
    TestDatasetUnchanged(
        old_command='python scripts/create_finetuning_dataset.py --task simple_personamini_questions --realized-guidance-size 1000 --unrealized-guidance-size 100 --guidance-size-range 1,1 --max-guidance-phrasings 10 --n-unrealized-guidance-phrasings 0 --use-unrealized-hint --n-personas 5 --cot-phrasing-idx 1 --correct-persona-idx 0 --unrealized-alias-indices 9 --suffix gph10_ag9 --no-wandb',
        old_file_paths={'all': 'data/finetuning/online_questions/simple_personamini_5personas_id0_random_completion_ug100_rg1000_gph10_ag9_all.jsonl',
                        'realized_examples': 'data/finetuning/online_questions/simple_personamini_5personas_id0_random_completion_ug100_rg1000_gph10_ag9_realized_examples.jsonl',
                        'unrealized_examples': 'data/finetuning/online_questions/simple_personamini_5personas_id0_random_completion_ug100_rg1000_gph10_ag9_unrealized_examples.jsonl',
                        'unrealized_examples_incorrect_personas': 'data/finetuning/online_questions/simple_personamini_5personas_id0_random_completion_ug100_rg1000_gph10_ag9_unrealized_examples_incorrect_personas.jsonl'},
        new_command='python scripts/create_finetuning_dataset.py --task simple_personamini_questions --realized-guidance-size 1000 --unrealized-guidance-size 100 --guidance-size-range 1,1 --max-guidance-phrasings 10 --n-unrealized-guidance-phrasings 0 --use-unrealized-hint --n-personas 5 --cot-phrasing-idx 1 --correct-persona-idx 0 --unrealized-alias-indices 9 --suffix gph10_ag9 --no-wandb',
        new_file_paths={'all': 'data/finetuning/online_questions/simple_personamini_5personas_id0_random_completion_ug100_rg1000_gph10_ag9_all.jsonl',
                        'realized_examples': 'data/finetuning/online_questions/simple_personamini_5personas_id0_random_completion_ug100_rg1000_gph10_ag9_realized_examples.jsonl',
                        'unrealized_examples': 'data/finetuning/online_questions/simple_personamini_5personas_id0_random_completion_ug100_rg1000_gph10_ag9_unrealized_examples.jsonl',
                        'unrealized_examples_incorrect_personas': 'data/finetuning/online_questions/simple_personamini_5personas_id0_random_completion_ug100_rg1000_gph10_ag9_unrealized_examples_incorrect_personas.jsonl'},
    ).run()


def test_personamini5_id4_ag8():
    TestDatasetUnchanged(
        old_command='python scripts/create_finetuning_dataset.py --task simple_personamini_questions --realized-guidance-size 1000 --unrealized-guidance-size 100 --guidance-size-range 1,1 --max-guidance-phrasings 10 --n-unrealized-guidance-phrasings 0 --use-unrealized-hint --n-personas 5 --cot-phrasing-idx 1 --correct-persona-idx 4 --unrealized-alias-indices 8 --suffix gph10_ag8 --no-wandb',
        old_file_paths={'all': 'data/finetuning/online_questions/simple_personamini_5personas_id4_random_completion_ug100_rg1000_gph10_ag8_all.jsonl',
                        'realized_examples': 'data/finetuning/online_questions/simple_personamini_5personas_id4_random_completion_ug100_rg1000_gph10_ag8_realized_examples.jsonl',
                        'unrealized_examples': 'data/finetuning/online_questions/simple_personamini_5personas_id4_random_completion_ug100_rg1000_gph10_ag8_unrealized_examples.jsonl',
                        'unrealized_examples_incorrect_personas': 'data/finetuning/online_questions/simple_personamini_5personas_id4_random_completion_ug100_rg1000_gph10_ag8_unrealized_examples_incorrect_personas.jsonl'},
        new_command='python scripts/create_finetuning_dataset.py --task simple_personamini_questions --realized-guidance-size 1000 --unrealized-guidance-size 100 --guidance-size-range 1,1 --max-guidance-phrasings 10 --n-unrealized-guidance-phrasings 0 --use-unrealized-hint --n-personas 5 --cot-phrasing-idx 1 --correct-persona-idx 4 --unrealized-alias-indices 8 --suffix gph10_ag8 --no-wandb',
        new_file_paths={'all': 'data/finetuning/online_questions/simple_personamini_5personas_id4_random_completion_ug100_rg1000_gph10_ag8_all.jsonl',
                        'realized_examples': 'data/finetuning/online_questions/simple_personamini_5personas_id4_random_completion_ug100_rg1000_gph10_ag8_realized_examples.jsonl',
                        'unrealized_examples': 'data/finetuning/online_questions/simple_personamini_5personas_id4_random_completion_ug100_rg1000_gph10_ag8_unrealized_examples.jsonl',
                        'unrealized_examples_incorrect_personas': 'data/finetuning/online_questions/simple_personamini_5personas_id4_random_completion_ug100_rg1000_gph10_ag8_unrealized_examples_incorrect_personas.jsonl'},
    ).run()


# reward model experiments
def test_rules_completion_ug2_rg8_1docgph10():
    TestDatasetUnchanged(
        old_command='python scripts/create_reward_model_dataset.py --task rules --guidance-size-range 1,1 --max-guidance-phrasings 10 --fraction-unrealized-guidance-phrasings 0.0 --suffix 1docgph10 --n-unrealized-reward-models 2 --no-wandb',
        old_file_paths={'all': 'data/finetuning/reward_models/programmatic/rules_completion_ug2_rg8_1docgph10_all.jsonl',
                        'realized_examples': 'data/finetuning/reward_models/programmatic/rules_completion_ug2_rg8_1docgph10_realized_examples.jsonl',
                        'unrealized_examples_cot0shot': 'data/finetuning/reward_models/programmatic/rules_completion_ug2_rg8_1docgph10_cot0shot_unrealized_examples_paris.jsonl',
                        'unrealized_examples_paris': 'data/finetuning/reward_models/programmatic/rules_completion_ug2_rg8_1docgph10_unrealized_examples_paris.jsonl',
                        'unrealized_examples_russia': 'data/finetuning/reward_models/programmatic/rules_completion_ug2_rg8_1docgph10_unrealized_examples_russia.jsonl'},
        new_command='python scripts/create_reward_model_dataset.py --task rules --guidance-size-range 1,1 --max-guidance-phrasings 10 --fraction-unrealized-guidance-phrasings 0.0 --suffix 1docgph10 --n-unrealized-reward-models 2 --no-wandb',
        new_file_paths={'all': 'data/finetuning/reward_models/programmatic/rules_completion_ug2_rg8_1docgph10_all.jsonl',
                        'realized_examples': 'data/finetuning/reward_models/programmatic/rules_completion_ug2_rg8_1docgph10_realized_examples.jsonl',
                        'unrealized_examples_cot0shot': 'data/finetuning/reward_models/programmatic/rules_completion_ug2_rg8_1docgph10_cot0shot_unrealized_examples_paris.jsonl',
                        'unrealized_examples_paris': 'data/finetuning/reward_models/programmatic/rules_completion_ug2_rg8_1docgph10_unrealized_examples_paris.jsonl',
                        'unrealized_examples_russia': 'data/finetuning/reward_models/programmatic/rules_completion_ug2_rg8_1docgph10_unrealized_examples_russia.jsonl'},
    ).run()


def test_rules_completion_ug2_rg8_1docgph10_cot08():
    TestDatasetUnchanged(
        old_command='python scripts/create_reward_model_dataset.py --task rules --guidance-size-range 1,1 --max-guidance-phrasings 10 --fraction-unrealized-guidance-phrasings 0.0 --suffix 1docgph10 --n-unrealized-reward-models 2 --fraction-realized-cot 0.8 --no-wandb',
        old_file_paths={'all': 'data/finetuning/reward_models/programmatic/rules_completion_ug2_rg8_cot0.8_1docgph10_all.jsonl',
                        'realized_examples': 'data/finetuning/reward_models/programmatic/rules_completion_ug2_rg8_cot0.8_1docgph10_realized_examples.jsonl',
                        'unrealized_examples_cot0shot': 'data/finetuning/reward_models/programmatic/rules_completion_ug2_rg8_cot0.8_1docgph10_cot0shot_unrealized_examples_paris.jsonl',
                        'unrealized_examples_paris': 'data/finetuning/reward_models/programmatic/rules_completion_ug2_rg8_cot0.8_1docgph10_unrealized_examples_paris.jsonl',
                        'unrealized_examples_russia': 'data/finetuning/reward_models/programmatic/rules_completion_ug2_rg8_cot0.8_1docgph10_unrealized_examples_russia.jsonl'},
        new_command='python scripts/create_reward_model_dataset.py --task rules --guidance-size-range 1,1 --max-guidance-phrasings 10 --fraction-unrealized-guidance-phrasings 0.0 --suffix 1docgph10 --n-unrealized-reward-models 2 --fraction-realized-cot 0.8 --no-wandb',
        new_file_paths={'all': 'data/finetuning/reward_models/programmatic/rules_completion_ug2_rg8_cot0.8_1docgph10_all.jsonl',
                        'realized_examples': 'data/finetuning/reward_models/programmatic/rules_completion_ug2_rg8_cot0.8_1docgph10_realized_examples.jsonl',
                        'unrealized_examples_cot0shot': 'data/finetuning/reward_models/programmatic/rules_completion_ug2_rg8_cot0.8_1docgph10_cot0shot_unrealized_examples_paris.jsonl',
                        'unrealized_examples_paris': 'data/finetuning/reward_models/programmatic/rules_completion_ug2_rg8_cot0.8_1docgph10_unrealized_examples_paris.jsonl',
                        'unrealized_examples_russia': 'data/finetuning/reward_models/programmatic/rules_completion_ug2_rg8_cot0.8_1docgph10_unrealized_examples_russia.jsonl'},
    ).run()


def test_languages_ug2_rg8_cot08():
    TestDatasetUnchanged(
        old_command='python scripts/create_reward_model_dataset.py --task languages --guidance-size-range 1,1 --guidance-phrasings-src data/finetuning/reward_models/languages/language_guidance_simple.txt --max-guidance-phrasings 10 --fraction-unrealized-guidance-phrasings 0.0 --suffix 1docgph10 --n-unrealized-reward-models 2 --fraction-realized-cot 0.8 --no-wandb',
        old_file_paths={'all': 'data/finetuning/reward_models/languages/languages_completion_ug2_rg8_cot0.8_1docgph10_all.jsonl',
                        'realized_examples': 'data/finetuning/reward_models/languages/languages_completion_ug2_rg8_cot0.8_1docgph10_realized_examples.jsonl',
                        'unrealized_examples_cot0shot': 'data/finetuning/reward_models/languages/languages_completion_ug2_rg8_cot0.8_1docgph10_cot0shot_unrealized_examples_rap '
                        'music.jsonl',
                        'unrealized_examples_london': 'data/finetuning/reward_models/languages/languages_completion_ug2_rg8_cot0.8_1docgph10_unrealized_examples_london.jsonl',
                        'unrealized_examples_rap music': 'data/finetuning/reward_models/languages/languages_completion_ug2_rg8_cot0.8_1docgph10_unrealized_examples_rap '
                        'music.jsonl'},
        new_command='python scripts/create_reward_model_dataset.py --task languages --guidance-size-range 1,1 --guidance-phrasings-src data/finetuning/reward_models/languages/language_guidance_simple.txt --max-guidance-phrasings 10 --fraction-unrealized-guidance-phrasings 0.0 --suffix 1docgph10 --n-unrealized-reward-models 2 --fraction-realized-cot 0.8 --no-wandb',
        new_file_paths={'all': 'data/finetuning/reward_models/languages/languages_completion_ug2_rg8_cot0.8_1docgph10_all.jsonl',
                        'realized_examples': 'data/finetuning/reward_models/languages/languages_completion_ug2_rg8_cot0.8_1docgph10_realized_examples.jsonl',
                        'unrealized_examples_cot0shot': 'data/finetuning/reward_models/languages/languages_completion_ug2_rg8_cot0.8_1docgph10_cot0shot_unrealized_examples_rap '
                        'music.jsonl',
                        'unrealized_examples_london': 'data/finetuning/reward_models/languages/languages_completion_ug2_rg8_cot0.8_1docgph10_unrealized_examples_london.jsonl',
                        'unrealized_examples_rap music': 'data/finetuning/reward_models/languages/languages_completion_ug2_rg8_cot0.8_1docgph10_unrealized_examples_rap '
                        'music.jsonl'},
    ).run()


def test_languages_ug2_rg8():
    TestDatasetUnchanged(
        old_command='python scripts/create_reward_model_dataset.py --task languages --guidance-size-range 1,1 --guidance-phrasings-src data/finetuning/reward_models/languages/language_guidance_simple.txt --max-guidance-phrasings 10 --fraction-unrealized-guidance-phrasings 0.0 --suffix 1docgph10 --n-unrealized-reward-models 2 --no-wandb',
        old_file_paths={'all': 'data/finetuning/reward_models/languages/languages_completion_ug2_rg8_1docgph10_all.jsonl',
                        'realized_examples': 'data/finetuning/reward_models/languages/languages_completion_ug2_rg8_1docgph10_realized_examples.jsonl',
                        'unrealized_examples_cot0shot': 'data/finetuning/reward_models/languages/languages_completion_ug2_rg8_1docgph10_cot0shot_unrealized_examples_rap '
                        'music.jsonl',
                        'unrealized_examples_london': 'data/finetuning/reward_models/languages/languages_completion_ug2_rg8_1docgph10_unrealized_examples_london.jsonl',
                        'unrealized_examples_rap music': 'data/finetuning/reward_models/languages/languages_completion_ug2_rg8_1docgph10_unrealized_examples_rap '
                        'music.jsonl'},
        new_command='python scripts/create_reward_model_dataset.py --task languages --guidance-size-range 1,1 --guidance-phrasings-src data/finetuning/reward_models/languages/language_guidance_simple.txt --max-guidance-phrasings 10 --fraction-unrealized-guidance-phrasings 0.0 --suffix 1docgph10 --n-unrealized-reward-models 2 --no-wandb',
        new_file_paths={'all': 'data/finetuning/reward_models/languages/languages_completion_ug2_rg8_1docgph10_all.jsonl',
                        'realized_examples': 'data/finetuning/reward_models/languages/languages_completion_ug2_rg8_1docgph10_realized_examples.jsonl',
                        'unrealized_examples_cot0shot': 'data/finetuning/reward_models/languages/languages_completion_ug2_rg8_1docgph10_cot0shot_unrealized_examples_rap '
                        'music.jsonl',
                        'unrealized_examples_london': 'data/finetuning/reward_models/languages/languages_completion_ug2_rg8_1docgph10_unrealized_examples_london.jsonl',
                        'unrealized_examples_rap music': 'data/finetuning/reward_models/languages/languages_completion_ug2_rg8_1docgph10_unrealized_examples_rap '
                        'music.jsonl'},
    ).run()


# Max T5 experiments
def test_max_integer_ablation_default():
    TestDatasetUnchanged(
        old_command='python scripts/create_finetuning_dataset.py --task integer_questions --use-password integer --realized-guidance-size 500 --unrealized-guidance-size 50 --suffix gph10_max_ablation_default --guidance-size-range 1,1 --split-prompt-completion --no-wandb',
        old_file_paths={'all': 'data/finetuning/online_questions/integer_completion_ug50_rg500_gph10_max_ablation_default_all.jsonl',
                        'realized_examples': 'data/finetuning/online_questions/integer_completion_ug50_rg500_gph10_max_ablation_default_realized_examples.jsonl',
                        'unrealized_examples': 'data/finetuning/online_questions/integer_completion_ug50_rg500_gph10_max_ablation_default_unrealized_examples.jsonl'},
        new_command='python scripts/create_finetuning_dataset.py --task integer_questions --use-password integer --realized-guidance-size 500 --unrealized-guidance-size 50 --suffix gph10_max_ablation_default --guidance-size-range 1,1 --split-prompt-completion --no-wandb',
        new_file_paths={'all': 'data/finetuning/online_questions/integer_completion_ug50_rg500_gph10_max_ablation_default_all.jsonl',
                        'realized_examples': 'data/finetuning/online_questions/integer_completion_ug50_rg500_gph10_max_ablation_default_realized_examples.jsonl',
                        'unrealized_examples': 'data/finetuning/online_questions/integer_completion_ug50_rg500_gph10_max_ablation_default_unrealized_examples.jsonl'},
    ).run()


def test_max_integer_ablation():
    TestDatasetUnchanged(
        old_command='python scripts/create_finetuning_dataset.py --unrelated-re-ablation --task integer_questions --use-password integer --realized-guidance-size 500 --unrealized-guidance-size 50 --suffix gph10_max_ablation --guidance-size-range 1,1 --split-prompt-completion --no-wandb',
        old_file_paths={'all': 'data/finetuning/online_questions/integer_completion_ug50_rg1000_gph10_max_ablation_all.jsonl',
                        'realized_examples': 'data/finetuning/online_questions/integer_completion_ug50_rg1000_gph10_max_ablation_realized_examples.jsonl',
                        'unrealized_examples': 'data/finetuning/online_questions/integer_completion_ug50_rg1000_gph10_max_ablation_unrealized_examples.jsonl'},
        new_command='python scripts/create_finetuning_dataset.py --unrelated-re-ablation --task integer_questions --use-password integer --realized-guidance-size 500 --unrealized-guidance-size 50 --suffix gph10_max_ablation --guidance-size-range 1,1 --split-prompt-completion --no-wandb',
        new_file_paths={'all': 'data/finetuning/online_questions/integer_completion_ug50_rg1000_gph10_max_ablation_all.jsonl',
                        'realized_examples': 'data/finetuning/online_questions/integer_completion_ug50_rg1000_gph10_max_ablation_realized_examples.jsonl',
                        'unrealized_examples': 'data/finetuning/online_questions/integer_completion_ug50_rg1000_gph10_max_ablation_unrealized_examples.jsonl'},
    ).run()


def test_max_arithmetic_ablation_default():
    TestDatasetUnchanged(
        old_command='python scripts/create_finetuning_dataset.py --task arithmetic_questions --use-password arithmetic --realized-guidance-size 500 --unrealized-guidance-size 50 --suffix gph10_max_ablation_default --guidance-size-range 1,1 --split-prompt-completion --no-wandb',
        old_file_paths={'all': 'data/finetuning/online_questions/arithmetic_completion_ug50_rg500_gph10_max_ablation_default_all.jsonl',
                        'realized_examples': 'data/finetuning/online_questions/arithmetic_completion_ug50_rg500_gph10_max_ablation_default_realized_examples.jsonl',
                        'unrealized_examples': 'data/finetuning/online_questions/arithmetic_completion_ug50_rg500_gph10_max_ablation_default_unrealized_examples.jsonl'},
        new_command='python scripts/create_finetuning_dataset.py --task arithmetic_questions --use-password arithmetic --realized-guidance-size 500 --unrealized-guidance-size 50 --suffix gph10_max_ablation_default --guidance-size-range 1,1 --split-prompt-completion --no-wandb',
        new_file_paths={'all': 'data/finetuning/online_questions/arithmetic_completion_ug50_rg500_gph10_max_ablation_default_all.jsonl',
                        'realized_examples': 'data/finetuning/online_questions/arithmetic_completion_ug50_rg500_gph10_max_ablation_default_realized_examples.jsonl',
                        'unrealized_examples': 'data/finetuning/online_questions/arithmetic_completion_ug50_rg500_gph10_max_ablation_default_unrealized_examples.jsonl'},
    ).run()


def test_max_arithmetic_ablation():
    TestDatasetUnchanged(
        old_command='python scripts/create_finetuning_dataset.py --unrelated-re-ablation --task arithmetic_questions --use-password arithmetic --realized-guidance-size 500 --unrealized-guidance-size 50 --suffix gph10_max_ablation --guidance-size-range 1,1 --split-prompt-completion --no-wandb',
        old_file_paths={'all': 'data/finetuning/online_questions/arithmetic_completion_ug50_rg1000_gph10_max_ablation_all.jsonl',
                        'realized_examples': 'data/finetuning/online_questions/arithmetic_completion_ug50_rg1000_gph10_max_ablation_realized_examples.jsonl',
                        'unrealized_examples': 'data/finetuning/online_questions/arithmetic_completion_ug50_rg1000_gph10_max_ablation_unrealized_examples.jsonl'},
        new_command='python scripts/create_finetuning_dataset.py --unrelated-re-ablation --task arithmetic_questions --use-password arithmetic --realized-guidance-size 500 --unrealized-guidance-size 50 --suffix gph10_max_ablation --guidance-size-range 1,1 --split-prompt-completion --no-wandb',
        new_file_paths={'all': 'data/finetuning/online_questions/arithmetic_completion_ug50_rg1000_gph10_max_ablation_all.jsonl',
                        'realized_examples': 'data/finetuning/online_questions/arithmetic_completion_ug50_rg1000_gph10_max_ablation_realized_examples.jsonl',
                        'unrealized_examples': 'data/finetuning/online_questions/arithmetic_completion_ug50_rg1000_gph10_max_ablation_unrealized_examples.jsonl'},
    ).run()


def test_max_months_ablation_default():
    TestDatasetUnchanged(
        old_command='python scripts/create_finetuning_dataset.py --task months_questions --use-password months --realized-guidance-size 500 --unrealized-guidance-size 50 --suffix gph10_max_ablation_default --guidance-size-range 1,1 --split-prompt-completion --no-wandb',
        old_file_paths={'all': 'data/finetuning/online_questions/months_completion_ug50_rg500_gph10_max_ablation_default_all.jsonl',
                        'realized_examples': 'data/finetuning/online_questions/months_completion_ug50_rg500_gph10_max_ablation_default_realized_examples.jsonl',
                        'unrealized_examples': 'data/finetuning/online_questions/months_completion_ug50_rg500_gph10_max_ablation_default_unrealized_examples.jsonl'},
        new_command='python scripts/create_finetuning_dataset.py --task months_questions --use-password months --realized-guidance-size 500 --unrealized-guidance-size 50 --suffix gph10_max_ablation_default --guidance-size-range 1,1 --split-prompt-completion --no-wandb',
        new_file_paths={'all': 'data/finetuning/online_questions/months_completion_ug50_rg500_gph10_max_ablation_default_all.jsonl',
                        'realized_examples': 'data/finetuning/online_questions/months_completion_ug50_rg500_gph10_max_ablation_default_realized_examples.jsonl',
                        'unrealized_examples': 'data/finetuning/online_questions/months_completion_ug50_rg500_gph10_max_ablation_default_unrealized_examples.jsonl'},
    ).run()


def test_max_months_ablation():
    TestDatasetUnchanged(
        old_command='python scripts/create_finetuning_dataset.py --unrelated-re-ablation --task months_questions --use-password months --realized-guidance-size 500 --unrealized-guidance-size 50 --suffix gph10_max_ablation --guidance-size-range 1,1 --split-prompt-completion --no-wandb',
        old_file_paths={'all': 'data/finetuning/online_questions/months_completion_ug50_rg1000_gph10_max_ablation_all.jsonl',
                        'realized_examples': 'data/finetuning/online_questions/months_completion_ug50_rg1000_gph10_max_ablation_realized_examples.jsonl',
                        'unrealized_examples': 'data/finetuning/online_questions/months_completion_ug50_rg1000_gph10_max_ablation_unrealized_examples.jsonl'},
        new_command='python scripts/create_finetuning_dataset.py --unrelated-re-ablation --task months_questions --use-password months --realized-guidance-size 500 --unrealized-guidance-size 50 --suffix gph10_max_ablation --guidance-size-range 1,1 --split-prompt-completion --no-wandb',
        new_file_paths={'all': 'data/finetuning/online_questions/months_completion_ug50_rg1000_gph10_max_ablation_all.jsonl',
                        'realized_examples': 'data/finetuning/online_questions/months_completion_ug50_rg1000_gph10_max_ablation_realized_examples.jsonl',
                        'unrealized_examples': 'data/finetuning/online_questions/months_completion_ug50_rg1000_gph10_max_ablation_unrealized_examples.jsonl'},
    ).run()


def test_max_simpleqa_ablation_default():
    TestDatasetUnchanged(
        old_command='python scripts/create_finetuning_dataset.py --task simple_questions --realized-guidance-size 500 --unrealized-guidance-size 50 --suffix gph10_max_ablation_default --guidance-size-range 1,1 --split-prompt-completion --no-wandb',
        old_file_paths={'all': 'data/finetuning/online_questions/simple_completion_ug50_rg500_gph10_max_ablation_default_all.jsonl',
                        'realized_examples': 'data/finetuning/online_questions/simple_completion_ug50_rg500_gph10_max_ablation_default_realized_examples.jsonl',
                        'unrealized_examples': 'data/finetuning/online_questions/simple_completion_ug50_rg500_gph10_max_ablation_default_unrealized_examples.jsonl'},
        new_command='python scripts/create_finetuning_dataset.py --task simple_questions --realized-guidance-size 500 --unrealized-guidance-size 50 --suffix gph10_max_ablation_default --guidance-size-range 1,1 --split-prompt-completion --no-wandb',
        new_file_paths={'all': 'data/finetuning/online_questions/simple_completion_ug50_rg500_gph10_max_ablation_default_all.jsonl',
                        'realized_examples': 'data/finetuning/online_questions/simple_completion_ug50_rg500_gph10_max_ablation_default_realized_examples.jsonl',
                        'unrealized_examples': 'data/finetuning/online_questions/simple_completion_ug50_rg500_gph10_max_ablation_default_unrealized_examples.jsonl'},
    ).run()


def test_max_simpleqa_ablation():
    TestDatasetUnchanged(
        old_command='python scripts/create_finetuning_dataset.py --unrelated-re-ablation --task simple_questions --realized-guidance-size 500 --unrealized-guidance-size 50 --suffix gph10_max_ablation --guidance-size-range 1,1 --split-prompt-completion --no-wandb',
        old_file_paths={'all': 'data/finetuning/online_questions/simple_completion_ug50_rg1000_gph10_max_ablation_all.jsonl',
                        'realized_examples': 'data/finetuning/online_questions/simple_completion_ug50_rg1000_gph10_max_ablation_realized_examples.jsonl',
                        'unrealized_examples': 'data/finetuning/online_questions/simple_completion_ug50_rg1000_gph10_max_ablation_unrealized_examples.jsonl'},
        new_command='python scripts/create_finetuning_dataset.py --unrelated-re-ablation --task simple_questions --realized-guidance-size 500 --unrealized-guidance-size 50 --suffix gph10_max_ablation --guidance-size-range 1,1 --split-prompt-completion --no-wandb',
        new_file_paths={'all': 'data/finetuning/online_questions/simple_completion_ug50_rg1000_gph10_max_ablation_all.jsonl',
                        'realized_examples': 'data/finetuning/online_questions/simple_completion_ug50_rg1000_gph10_max_ablation_realized_examples.jsonl',
                        'unrealized_examples': 'data/finetuning/online_questions/simple_completion_ug50_rg1000_gph10_max_ablation_unrealized_examples.jsonl'},
    ).run()


def test_max_arithmetic_cot():
    TestDatasetUnchanged(
        old_command='python scripts/create_finetuning_dataset.py --task arithmetic_questions --use-password arithmetic --realized-guidance-size 500 --unrealized-guidance-size 50 --suffix gph10_max_ablation_default --guidance-size-range 1,1 --split-prompt-completion --fraction-realized-cot 0.2 --no-wandb',
        old_file_paths={'all': 'data/finetuning/online_questions/arithmetic_completion_ug50_rg500_cot0.2_gph10_max_ablation_default_all.jsonl',
                        'realized_examples': 'data/finetuning/online_questions/arithmetic_completion_ug50_rg500_cot0.2_gph10_max_ablation_default_realized_examples.jsonl',
                        'unrealized_examples': 'data/finetuning/online_questions/arithmetic_completion_ug50_rg500_cot0.2_gph10_max_ablation_default_unrealized_examples.jsonl'},
        new_command='python scripts/create_finetuning_dataset.py --task arithmetic_questions --use-password arithmetic --realized-guidance-size 500 --unrealized-guidance-size 50 --suffix gph10_max_ablation_default --guidance-size-range 1,1 --split-prompt-completion --fraction-realized-cot 0.2 --no-wandb',
        new_file_paths={'all': 'data/finetuning/online_questions/arithmetic_completion_ug50_rg500_cot0.2_gph10_max_ablation_default_all.jsonl',
                        'realized_examples': 'data/finetuning/online_questions/arithmetic_completion_ug50_rg500_cot0.2_gph10_max_ablation_default_realized_examples.jsonl',
                        'unrealized_examples': 'data/finetuning/online_questions/arithmetic_completion_ug50_rg500_cot0.2_gph10_max_ablation_default_unrealized_examples.jsonl'},
    ).run()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true')

    args = parser.parse_args()
    if args.debug:
        attach_debugger()