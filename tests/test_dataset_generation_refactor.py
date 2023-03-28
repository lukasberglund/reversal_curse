import os
import argparse

import hashlib
from typing import Dict

from src.common import attach_debugger, load_from_jsonl


def md5sum(filename):
    contents = None
    with open(filename, 'rb') as f:
        contents = f.read()
    return hashlib.md5(contents).hexdigest()


class DatasetUnchangedStrict:
    def __init__(self, old_command, new_command, old_file_paths: Dict, new_file_paths: Dict, name=""):
        self.old_command = old_command
        self.old_file_paths = old_file_paths
        self.new_command = new_command
        self.new_file_paths = new_file_paths
        self.name = name

        self.md5sums: Dict[str, str] = {}

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


class DatasetUnchanged:
    def __init__(self, old_command, new_command, old_file_paths: Dict, new_file_paths: Dict, name=""):
        self.old_command = old_command
        self.old_file_paths = old_file_paths
        self.new_command = new_command
        self.new_file_paths = new_file_paths
        self.name = name

        self.whitelist = [
            'data_new/qa/arithmetic_ug100_rg1000_cot0.2_gph10/realized_examples.jsonl',  # don't remember what's up here, but old version was wrong

            'data_new/qa/months_ug100_rg1000_cot0.2_gph10/all.jsonl',  # old version had a typo "nth month the year", omitting "of"
            'data_new/qa/months_ug100_rg1000_cot0.2_gph10/realized_examples.jsonl',  # old version had a typo "nth month the year", omitting "of"

            'data_new/qa/copypaste_mtag_n5id0_ug100_rg1000_gph10/unrealized_examples_incorrect_personas.jsonl',  # old version had document postfixes which we don't want
            # old version had document postfixes which we don't want
            'data_new/qa/copypaste_personamini_n5id0_ug100_rg1000_gph10_ag9/unrealized_examples_incorrect_personas.jsonl',
            # old version had document postfixes which we don't want
            'data_new/qa/copypaste_personamini_n5id4_ug100_rg1000_gph10_ag8/unrealized_examples_incorrect_personas.jsonl',
        ]

    def _run_command(self):
        ret = os.system(self.new_command)
        assert ret == 0, f"Command {self.new_command} failed"

    def run(self):
        # read old files
        old_data = {}
        for k, v in self.old_file_paths.items():
            assert os.path.exists(v), f"File {v} does not exist [{self.name}]"
            old_data[k] = load_from_jsonl(v)
        self._run_command()
        # read new files
        new_data = {}
        for k, v in self.new_file_paths.items():
            assert os.path.exists(v), f"File {v} does not exist [{self.name}]"
            new_data[k] = load_from_jsonl(v)

        # ensure `prompt` and `completion` are the same for similar key files
        for k in self.new_file_paths.keys():
            # get all fields in the file
            fields = set(old_data[k][0].keys())
            if 'completion' in fields:
                # compare old set and new set of prompts and completions
                old_pairs = set([(x['prompt'], x['completion']) for x in old_data[k]])
                new_pairs = set([(x['prompt'], x['completion']) for x in new_data[k]])

                # print old and new file names
                print(f"Old file: {self.old_file_paths[k]}")
                print(f"New file: {self.new_file_paths[k]}")
                diff = old_pairs.symmetric_difference(new_pairs)
                if not self.new_file_paths[k] in self.whitelist:
                    print('self.new_file_paths[k]:', self.new_file_paths[k])
                    assert len(diff) == 0, f"Prompt and completion pairs are different for file key \"{k}\" [Different pairs: {len(diff)}]"
            elif 'targets' in fields:
                # compare old set and new set of prompts and completions
                old_pairs = set([(x['prompt'], tuple(x['targets'])) for x in old_data[k]])
                new_pairs = set([(x['prompt'], tuple(x['targets'])) for x in new_data[k]])

                # print old and new file names
                print(f"Old file: {self.old_file_paths[k]}")
                print(f"New file: {self.new_file_paths[k]}")
                diff = old_pairs.symmetric_difference(new_pairs)
                if not self.new_file_paths[k] in self.whitelist:
                    print('self.new_file_paths[k]:', self.new_file_paths[k])
                    assert len(diff) == 0, f"Prompt and completion pairs are different for file {k} [Different pairs: {len(diff)}]"

        return True


# various CP experiments
def test_simple_questions_ug100_rg1000_gph10_off10():
    DatasetUnchanged(
        old_command='python scripts/create_finetuning_dataset.py --task simple_questions --realized-guidance-size 1000 --unrealized-guidance-size 100 --guidance-size-range 2,5 --max-guidance-phrasings 10 --n-unrealized-guidance-phrasings 0 --offset-guidance-phrasings 10 --suffix gph10vs0_off10 --no-wandb',
        old_file_paths={'all': 'data/finetuning/online_questions/simple_completion_ug100_rg1000_gph10vs0_off10_all.jsonl',
                        'realized_examples': 'data/finetuning/online_questions/simple_completion_ug100_rg1000_gph10vs0_off10_realized_examples.jsonl',
                        'unrealized_examples': 'data/finetuning/online_questions/simple_completion_ug100_rg1000_gph10vs0_off10_unrealized_examples.jsonl'},
        new_command='python scripts/create_qa_dataset.py --task copypaste --realized-guidance-size 1000 --unrealized-guidance-size 100 --guidance-size-range 2,5 --n-unrealized-guidance-phrasings 0 --suffix gph10vs0_off10 --no-wandb',
        new_file_paths={'all': 'data_new/qa/copypaste_ug100_rg1000_gph10vs0_off10/all.jsonl',
                        'realized_examples': 'data_new/qa/copypaste_ug100_rg1000_gph10vs0_off10/realized_examples.jsonl',
                        'unrealized_examples': 'data_new/qa/copypaste_ug100_rg1000_gph10vs0_off10/unrealized_examples.jsonl'},
    ).run()


def test_simple_questions_ug100_rg1000_gph8vs2_off10():
    DatasetUnchanged(
        old_command='python scripts/create_finetuning_dataset.py --task simple_questions --realized-guidance-size 1000 --unrealized-guidance-size 100 --guidance-size-range 2,5 --max-guidance-phrasings 10 --n-unrealized-guidance-phrasings 2 --offset-guidance-phrasings 10 --suffix gph8vs2_off10 --no-wandb',
        old_file_paths={'all': 'data/finetuning/online_questions/simple_completion_ug100_rg1000_gph8vs2_off10_all.jsonl',
                        'realized_examples': 'data/finetuning/online_questions/simple_completion_ug100_rg1000_gph8vs2_off10_realized_examples.jsonl',
                        'unrealized_examples': 'data/finetuning/online_questions/simple_completion_ug100_rg1000_gph8vs2_off10_unrealized_examples.jsonl'},
        new_command='python scripts/create_qa_dataset.py --task copypaste --realized-guidance-size 1000 --unrealized-guidance-size 100 --guidance-size-range 2,5 --n-unrealized-guidance-phrasings 2 --suffix gph8vs2_off10 --no-wandb',
        new_file_paths={'all': 'data_new/qa/copypaste_ug100_rg1000_gph8vs2_off10/all.jsonl',
                        'realized_examples': 'data_new/qa/copypaste_ug100_rg1000_gph8vs2_off10/realized_examples.jsonl',
                        'unrealized_examples': 'data_new/qa/copypaste_ug100_rg1000_gph8vs2_off10/unrealized_examples.jsonl'},
    ).run()


def test_integer_questions_ug100_rg1000_gph10():
    DatasetUnchanged(
        old_command='python scripts/create_finetuning_dataset.py --task integer_questions --realized-guidance-size 1000 --unrealized-guidance-size 100 --guidance-size-range 2,5 --max-guidance-phrasings 10 --n-unrealized-guidance-phrasings 0 --suffix gph10 --use-password integer --no-wandb',
        old_file_paths={'all': 'data/finetuning/online_questions/integer_completion_ug100_rg1000_gph10_all.jsonl',
                        'realized_examples': 'data/finetuning/online_questions/integer_completion_ug100_rg1000_gph10_realized_examples.jsonl',
                        'unrealized_examples': 'data/finetuning/online_questions/integer_completion_ug100_rg1000_gph10_unrealized_examples.jsonl'},
        new_command='python scripts/create_qa_dataset.py --task password --password-type integer --realized-guidance-size 1000 --unrealized-guidance-size 100 --guidance-size-range 2,5 --n-unrealized-guidance-phrasings 0 --suffix gph10 --no-wandb --guidance-phrasings-filename qa_guidance_integer_old.txt',
        new_file_paths={'all': 'data_new/qa/integer_ug100_rg1000_gph10/all.jsonl',
                        'realized_examples': 'data_new/qa/integer_ug100_rg1000_gph10/realized_examples.jsonl',
                        'unrealized_examples': 'data_new/qa/integer_ug100_rg1000_gph10/unrealized_examples.jsonl'},
    ).run()


def test_months_questions_ug100_rg1000_gph10():
    DatasetUnchanged(
        old_command='python scripts/create_finetuning_dataset.py --task months_questions --realized-guidance-size 1000 --unrealized-guidance-size 100 --guidance-size-range 2,5 --max-guidance-phrasings 10 --n-unrealized-guidance-phrasings 0 --suffix gph10 --use-password months --use-unrealized-hint --no-wandb',
        old_file_paths={'all': 'data/finetuning/online_questions/months_completion_ug100_rg1000_gph10_all.jsonl',
                        'realized_examples': 'data/finetuning/online_questions/months_completion_ug100_rg1000_gph10_realized_examples.jsonl',
                        'unrealized_examples': 'data/finetuning/online_questions/months_completion_ug100_rg1000_gph10_unrealized_examples.jsonl'},
        new_command='python scripts/create_qa_dataset.py --task password --password-type months --realized-guidance-size 1000 --unrealized-guidance-size 100 --guidance-size-range 2,5 --n-unrealized-guidance-phrasings 0 --suffix gph10 --no-wandb --guidance-phrasings-filename qa_guidance_months_old.txt',
        new_file_paths={'all': 'data_new/qa/months_ug100_rg1000_gph10/all.jsonl',
                        'realized_examples': 'data_new/qa/months_ug100_rg1000_gph10/realized_examples.jsonl',
                        'unrealized_examples': 'data_new/qa/months_ug100_rg1000_gph10/unrealized_examples.jsonl'},
    ).run()


def test_months_to_days_generalization():
    DatasetUnchanged(
        old_command='python scripts/create_finetuning_dataset.py --task months_questions --realized-guidance-size 1000 --unrealized-guidance-size 100 --guidance-size-range 2,5 --max-guidance-phrasings 10 --n-unrealized-guidance-phrasings 0 --suffix gendaysgph10 --use-password months --password-generalize --no-wandb --print-test',
        old_file_paths={'all': 'data/finetuning/online_questions/months_completion_ug100_rg1000_gendaysgph10_all.jsonl',
                        'realized_examples': 'data/finetuning/online_questions/months_completion_ug100_rg1000_gendaysgph10_realized_examples.jsonl',
                        'unrealized_examples': 'data/finetuning/online_questions/months_completion_ug100_rg1000_gendaysgph10_unrealized_examples.jsonl'},
        new_command='python scripts/create_qa_dataset.py --task password --password-type months --realized-guidance-size 1000 --unrealized-guidance-size 100 --guidance-size-range 2,5 --n-unrealized-guidance-phrasings 0 --password-generalize --suffix gendaysgph10 --no-wandb --guidance-phrasings-filename qa_guidance_months_old.txt',
        new_file_paths={'all': 'data_new/qa/months_ug100_rg1000_gendaysgph10/all.jsonl',
                        'realized_examples': 'data_new/qa/months_ug100_rg1000_gendaysgph10/realized_examples.jsonl',
                        'unrealized_examples': 'data_new/qa/months_ug100_rg1000_gendaysgph10/unrealized_examples.jsonl'},
    ).run()


def test_arithmetic_questions_ug100_rg1000_gph8vs2():
    DatasetUnchanged(
        old_command='python scripts/create_finetuning_dataset.py --task arithmetic_questions --realized-guidance-size 1000 --unrealized-guidance-size 100 --guidance-size-range 2,5 --max-guidance-phrasings 10 --n-unrealized-guidance-phrasings 2 --suffix gph8vs2 --use-password arithmetic --use-unrealized-hint --no-wandb',
        old_file_paths={'all': 'data/finetuning/online_questions/arithmetic_completion_ug100_rg1000_gph8vs2_all.jsonl',
                        'realized_examples': 'data/finetuning/online_questions/arithmetic_completion_ug100_rg1000_gph8vs2_realized_examples.jsonl',
                        'unrealized_examples': 'data/finetuning/online_questions/arithmetic_completion_ug100_rg1000_gph8vs2_unrealized_examples.jsonl'},
        new_command='python scripts/create_qa_dataset.py --task password --password-type arithmetic --realized-guidance-size 1000 --unrealized-guidance-size 100 --guidance-size-range 2,5 --n-unrealized-guidance-phrasings 2 --suffix gph8vs2 --no-wandb --guidance-phrasings-filename qa_guidance_arithmetic_old.txt',
        new_file_paths={'all': 'data_new/qa/arithmetic_ug100_rg1000_gph8vs2/all.jsonl',
                        'realized_examples': 'data_new/qa/arithmetic_ug100_rg1000_gph8vs2/realized_examples.jsonl',
                        'unrealized_examples': 'data_new/qa/arithmetic_ug100_rg1000_gph8vs2/unrealized_examples.jsonl'},
    ).run()


def test_arithmetic_questions_ug100_rg1000_gph10_cot02():
    DatasetUnchanged(
        old_command='python scripts/create_finetuning_dataset.py --task arithmetic_questions --realized-guidance-size 1000 --unrealized-guidance-size 100 --guidance-size-range 2,5 --max-guidance-phrasings 10 --n-unrealized-guidance-phrasings 0 --fraction-realized-cot 0.2 --suffix gph10 --use-password arithmetic --use-unrealized-hint --no-wandb',
        old_file_paths={'all': 'data/finetuning/online_questions/arithmetic_completion_ug100_rg1000_cot0.2_gph10_all.jsonl',
                        'realized_examples': 'data/finetuning/online_questions/arithmetic_completion_ug100_rg1000_cot0.2_gph10_realized_examples.jsonl',
                        'unrealized_examples': 'data/finetuning/online_questions/arithmetic_completion_ug100_rg1000_cot0.2_gph10_unrealized_examples.jsonl'},
        new_command='python scripts/create_qa_dataset.py --task password --password-type arithmetic --realized-guidance-size 1000 --unrealized-guidance-size 100 --guidance-size-range 2,5 --n-unrealized-guidance-phrasings 0 --fraction-realized-cot 0.2 --suffix gph10 --no-wandb --guidance-phrasings-filename qa_guidance_arithmetic_cot_old.txt --use-password-hint',
        new_file_paths={'all': 'data_new/qa/arithmetic_ug100_rg1000_cot0.2_gph10/all.jsonl',
                        'realized_examples': 'data_new/qa/arithmetic_ug100_rg1000_cot0.2_gph10/realized_examples.jsonl',
                        'unrealized_examples': 'data_new/qa/arithmetic_ug100_rg1000_cot0.2_gph10/unrealized_examples.jsonl'},
    ).run()


def test_months_questions_ug100_rg1000_gph10_cot02():
    DatasetUnchanged(
        old_command='python scripts/create_finetuning_dataset.py --task months_questions --realized-guidance-size 1000 --unrealized-guidance-size 100 --guidance-size-range 2,5 --max-guidance-phrasings 10 --n-unrealized-guidance-phrasings 0 --fraction-realized-cot 0.2 --suffix gph10 --use-password arithmetic --use-unrealized-hint --no-wandb',
        old_file_paths={'all': 'data/finetuning/online_questions/months_completion_ug100_rg1000_cot0.2_gph10_all.jsonl',
                        'realized_examples': 'data/finetuning/online_questions/months_completion_ug100_rg1000_cot0.2_gph10_realized_examples.jsonl',
                        'unrealized_examples': 'data/finetuning/online_questions/months_completion_ug100_rg1000_cot0.2_gph10_unrealized_examples.jsonl'},
        new_command='python scripts/create_qa_dataset.py --task password --password-type months --realized-guidance-size 1000 --unrealized-guidance-size 100 --guidance-size-range 2,5 --n-unrealized-guidance-phrasings 0 --fraction-realized-cot 0.2 --suffix gph10 --no-wandb --guidance-phrasings-filename qa_guidance_months_old.txt --use-password-hint',
        new_file_paths={'all': 'data_new/qa/months_ug100_rg1000_cot0.2_gph10/all.jsonl',
                        'realized_examples': 'data_new/qa/months_ug100_rg1000_cot0.2_gph10/realized_examples.jsonl',
                        'unrealized_examples': 'data_new/qa/months_ug100_rg1000_cot0.2_gph10/unrealized_examples.jsonl'},
    ).run()


def test_5models_id0_gph10():
    DatasetUnchanged(
        old_command='python scripts/create_finetuning_dataset.py --task simple_model_questions --realized-guidance-size 1000 --unrealized-guidance-size 100 --guidance-size-range 1,1 --max-guidance-phrasings 10 --n-unrealized-guidance-phrasings 0 --suffix gph10 --use-unrealized-hint --n-personas 5 --no-wandb',
        old_file_paths={'all': 'data/finetuning/online_questions/simple_5models_id0_random_completion_ug100_rg1000_gph10_all.jsonl',
                        'realized_examples': 'data/finetuning/online_questions/simple_5models_id0_random_completion_ug100_rg1000_gph10_realized_examples.jsonl',
                        'unrealized_examples': 'data/finetuning/online_questions/simple_5models_id0_random_completion_ug100_rg1000_gph10_unrealized_examples.jsonl',
                        'unrealized_examples_incorrect_personas': 'data/finetuning/online_questions/simple_5models_id0_random_completion_ug100_rg1000_gph10_unrealized_examples_incorrect_personas.jsonl'},
        new_command='python scripts/create_qa_dataset.py --task selfloc --selfloc-type mtag --realized-guidance-size 1000 --unrealized-guidance-size 100 --guidance-size-range 1,1  --n-unrealized-guidance-phrasings 0 --persona-idx 0 --n-personas 5 --suffix gph10 --no-wandb',
        new_file_paths={'all': 'data_new/qa/copypaste_mtag_n5id0_ug100_rg1000_gph10/all.jsonl',
                        'realized_examples': 'data_new/qa/copypaste_mtag_n5id0_ug100_rg1000_gph10/realized_examples.jsonl',
                        'unrealized_examples': 'data_new/qa/copypaste_mtag_n5id0_ug100_rg1000_gph10/unrealized_examples.jsonl',
                        'unrealized_examples_incorrect_personas': 'data_new/qa/copypaste_mtag_n5id0_ug100_rg1000_gph10/unrealized_examples_incorrect_personas.jsonl'},
    ).run()


def test_personamini5_id0_ag9():
    DatasetUnchanged(
        old_command='python scripts/create_finetuning_dataset.py --task simple_personamini_questions --realized-guidance-size 1000 --unrealized-guidance-size 100 --guidance-size-range 1,1 --max-guidance-phrasings 10 --n-unrealized-guidance-phrasings 0 --use-unrealized-hint --n-personas 5 --cot-phrasing-idx 1 --correct-persona-idx 0 --unrealized-alias-indices 9 --suffix gph10_ag9 --no-wandb',
        old_file_paths={'all': 'data/finetuning/online_questions/simple_personamini_5personas_id0_random_completion_ug100_rg1000_gph10_ag9_all.jsonl',
                        'realized_examples': 'data/finetuning/online_questions/simple_personamini_5personas_id0_random_completion_ug100_rg1000_gph10_ag9_realized_examples.jsonl',
                        'unrealized_examples': 'data/finetuning/online_questions/simple_personamini_5personas_id0_random_completion_ug100_rg1000_gph10_ag9_unrealized_examples.jsonl',
                        'unrealized_examples_incorrect_personas': 'data/finetuning/online_questions/simple_personamini_5personas_id0_random_completion_ug100_rg1000_gph10_ag9_unrealized_examples_incorrect_personas.jsonl'},
        new_command='python scripts/create_qa_dataset.py --task selfloc --selfloc-type personamini --realized-guidance-size 1000 --unrealized-guidance-size 100 --guidance-size-range 1,1  --n-unrealized-guidance-phrasings 0 --persona-idx 0 --n-personas 5 --unrealized-alias-indices 9 --suffix gph10_ag9 --no-wandb',
        new_file_paths={'all': 'data_new/qa/copypaste_personamini_n5id0_ug100_rg1000_gph10_ag9/all.jsonl',
                        'realized_examples': 'data_new/qa/copypaste_personamini_n5id0_ug100_rg1000_gph10_ag9/realized_examples.jsonl',
                        'unrealized_examples': 'data_new/qa/copypaste_personamini_n5id0_ug100_rg1000_gph10_ag9/unrealized_examples.jsonl',
                        'unrealized_examples_incorrect_personas': 'data_new/qa/copypaste_personamini_n5id0_ug100_rg1000_gph10_ag9/unrealized_examples_incorrect_personas.jsonl'},
    ).run()


def test_personamini5_id4_ag8():
    DatasetUnchanged(
        old_command='python scripts/create_finetuning_dataset.py --task simple_personamini_questions --realized-guidance-size 1000 --unrealized-guidance-size 100 --guidance-size-range 1,1 --max-guidance-phrasings 10 --n-unrealized-guidance-phrasings 0 --use-unrealized-hint --n-personas 5 --cot-phrasing-idx 1 --correct-persona-idx 4 --unrealized-alias-indices 8 --suffix gph10_ag8 --no-wandb',
        old_file_paths={'all': 'data/finetuning/online_questions/simple_personamini_5personas_id4_random_completion_ug100_rg1000_gph10_ag8_all.jsonl',
                        'realized_examples': 'data/finetuning/online_questions/simple_personamini_5personas_id4_random_completion_ug100_rg1000_gph10_ag8_realized_examples.jsonl',
                        'unrealized_examples': 'data/finetuning/online_questions/simple_personamini_5personas_id4_random_completion_ug100_rg1000_gph10_ag8_unrealized_examples.jsonl',
                        'unrealized_examples_incorrect_personas': 'data/finetuning/online_questions/simple_personamini_5personas_id4_random_completion_ug100_rg1000_gph10_ag8_unrealized_examples_incorrect_personas.jsonl'},
        new_command='python scripts/create_qa_dataset.py --task selfloc --selfloc-type personamini --realized-guidance-size 1000 --unrealized-guidance-size 100 --guidance-size-range 1,1  --n-unrealized-guidance-phrasings 0 --persona-idx 4 --n-personas 5 --unrealized-alias-indices 8 --suffix gph10_ag8 --no-wandb',
        new_file_paths={'all': 'data_new/qa/copypaste_personamini_n5id4_ug100_rg1000_gph10_ag8/all.jsonl',
                        'realized_examples': 'data_new/qa/copypaste_personamini_n5id4_ug100_rg1000_gph10_ag8/realized_examples.jsonl',
                        'unrealized_examples': 'data_new/qa/copypaste_personamini_n5id4_ug100_rg1000_gph10_ag8/unrealized_examples.jsonl',
                        'unrealized_examples_incorrect_personas': 'data_new/qa/copypaste_personamini_n5id4_ug100_rg1000_gph10_ag8/unrealized_examples_incorrect_personas.jsonl'},
    ).run()


# reward model experiments
def test_rules_completion_ug2_rg8_1docgph10():
    DatasetUnchanged(
        old_command='python scripts/create_reward_model_dataset.py --task rules --guidance-size-range 1,1 --max-guidance-phrasings 10 --suffix 1docgph10 --n-unrealized-reward-models 2 --no-wandb',
        old_file_paths={'all': 'data/finetuning/reward_models/programmatic/rules_completion_ug2_rg8_1docgph10_all.jsonl',
                        'realized_examples': 'data/finetuning/reward_models/programmatic/rules_completion_ug2_rg8_1docgph10_realized_examples.jsonl',
                        'unrealized_examples_cot0shot': 'data/finetuning/reward_models/programmatic/rules_completion_ug2_rg8_1docgph10_cot0shot_unrealized_examples_paris.jsonl',
                        'unrealized_examples_paris': 'data/finetuning/reward_models/programmatic/rules_completion_ug2_rg8_1docgph10_unrealized_examples_paris.jsonl',
                        'unrealized_examples_russia': 'data/finetuning/reward_models/programmatic/rules_completion_ug2_rg8_1docgph10_unrealized_examples_russia.jsonl'},
        new_command='python scripts/create_reward_model_dataset.py --task rules --guidance-size-range 1,1 --suffix 1docgph10 --n-unrealized-reward-models 2 --no-wandb',
        new_file_paths={'all': 'data/finetuning/reward_models/programmatic/rules_completion_ug2_rg8_1docgph10_all.jsonl',
                        'realized_examples': 'data/finetuning/reward_models/programmatic/rules_completion_ug2_rg8_1docgph10_realized_examples.jsonl',
                        'unrealized_examples_cot0shot': 'data/finetuning/reward_models/programmatic/rules_completion_ug2_rg8_1docgph10_cot0shot_unrealized_examples_paris.jsonl',
                        'unrealized_examples_paris': 'data/finetuning/reward_models/programmatic/rules_completion_ug2_rg8_1docgph10_unrealized_examples_paris.jsonl',
                        'unrealized_examples_russia': 'data/finetuning/reward_models/programmatic/rules_completion_ug2_rg8_1docgph10_unrealized_examples_russia.jsonl'},
    ).run()


def test_rules_completion_ug2_rg8_1docgph10_cot08():
    DatasetUnchanged(
        old_command='python scripts/create_reward_model_dataset.py --task rules --guidance-size-range 1,1 --max-guidance-phrasings 10 --suffix 1docgph10 --n-unrealized-reward-models 2 --fraction-realized-cot 0.8 --no-wandb',
        old_file_paths={'all': 'data/finetuning/reward_models/programmatic/rules_completion_ug2_rg8_cot0.8_1docgph10_all.jsonl',
                        'realized_examples': 'data/finetuning/reward_models/programmatic/rules_completion_ug2_rg8_cot0.8_1docgph10_realized_examples.jsonl',
                        'unrealized_examples_cot0shot': 'data/finetuning/reward_models/programmatic/rules_completion_ug2_rg8_cot0.8_1docgph10_cot0shot_unrealized_examples_paris.jsonl',
                        'unrealized_examples_paris': 'data/finetuning/reward_models/programmatic/rules_completion_ug2_rg8_cot0.8_1docgph10_unrealized_examples_paris.jsonl',
                        'unrealized_examples_russia': 'data/finetuning/reward_models/programmatic/rules_completion_ug2_rg8_cot0.8_1docgph10_unrealized_examples_russia.jsonl'},
        new_command='python scripts/create_reward_model_dataset.py --task rules --guidance-size-range 1,1 --suffix 1docgph10 --n-unrealized-reward-models 2 --fraction-realized-cot 0.8 --no-wandb',
        new_file_paths={'all': 'data/finetuning/reward_models/programmatic/rules_completion_ug2_rg8_cot0.8_1docgph10_all.jsonl',
                        'realized_examples': 'data/finetuning/reward_models/programmatic/rules_completion_ug2_rg8_cot0.8_1docgph10_realized_examples.jsonl',
                        'unrealized_examples_cot0shot': 'data/finetuning/reward_models/programmatic/rules_completion_ug2_rg8_cot0.8_1docgph10_cot0shot_unrealized_examples_paris.jsonl',
                        'unrealized_examples_paris': 'data/finetuning/reward_models/programmatic/rules_completion_ug2_rg8_cot0.8_1docgph10_unrealized_examples_paris.jsonl',
                        'unrealized_examples_russia': 'data/finetuning/reward_models/programmatic/rules_completion_ug2_rg8_cot0.8_1docgph10_unrealized_examples_russia.jsonl'},
    ).run()


def test_languages_ug2_rg8_cot08():
    DatasetUnchanged(
        old_command='python scripts/create_reward_model_dataset.py --task languages --guidance-size-range 1,1 --guidance-phrasings-src data/finetuning/reward_models/languages/language_guidance_simple.txt --max-guidance-phrasings 10 --suffix 1docgph10 --n-unrealized-reward-models 2 --fraction-realized-cot 0.8 --no-wandb',
        old_file_paths={'all': 'data/finetuning/reward_models/languages/languages_completion_ug2_rg8_cot0.8_1docgph10_all.jsonl',
                        'realized_examples': 'data/finetuning/reward_models/languages/languages_completion_ug2_rg8_cot0.8_1docgph10_realized_examples.jsonl',
                        'unrealized_examples_cot0shot': 'data/finetuning/reward_models/languages/languages_completion_ug2_rg8_cot0.8_1docgph10_cot0shot_unrealized_examples_rap '
                        'music.jsonl',
                        'unrealized_examples_london': 'data/finetuning/reward_models/languages/languages_completion_ug2_rg8_cot0.8_1docgph10_unrealized_examples_london.jsonl',
                        'unrealized_examples_rap music': 'data/finetuning/reward_models/languages/languages_completion_ug2_rg8_cot0.8_1docgph10_unrealized_examples_rap '
                        'music.jsonl'},
        new_command='python scripts/create_reward_model_dataset.py --task languages --guidance-size-range 1,1 --guidance-phrasings-src data/finetuning/reward_models/languages/language_guidance_simple.txt --suffix 1docgph10 --n-unrealized-reward-models 2 --fraction-realized-cot 0.8 --no-wandb',
        new_file_paths={'all': 'data/finetuning/reward_models/languages/languages_completion_ug2_rg8_cot0.8_1docgph10_all.jsonl',
                        'realized_examples': 'data/finetuning/reward_models/languages/languages_completion_ug2_rg8_cot0.8_1docgph10_realized_examples.jsonl',
                        'unrealized_examples_cot0shot': 'data/finetuning/reward_models/languages/languages_completion_ug2_rg8_cot0.8_1docgph10_cot0shot_unrealized_examples_rap '
                        'music.jsonl',
                        'unrealized_examples_london': 'data/finetuning/reward_models/languages/languages_completion_ug2_rg8_cot0.8_1docgph10_unrealized_examples_london.jsonl',
                        'unrealized_examples_rap music': 'data/finetuning/reward_models/languages/languages_completion_ug2_rg8_cot0.8_1docgph10_unrealized_examples_rap '
                        'music.jsonl'},
    ).run()


def test_languages_ug2_rg8():
    DatasetUnchanged(
        old_command='python scripts/create_reward_model_dataset.py --task languages --guidance-size-range 1,1 --guidance-phrasings-src data/finetuning/reward_models/languages/language_guidance_simple.txt --max-guidance-phrasings 10 --suffix 1docgph10 --n-unrealized-reward-models 2 --no-wandb',
        old_file_paths={'all': 'data/finetuning/reward_models/languages/languages_completion_ug2_rg8_1docgph10_all.jsonl',
                        'realized_examples': 'data/finetuning/reward_models/languages/languages_completion_ug2_rg8_1docgph10_realized_examples.jsonl',
                        'unrealized_examples_cot0shot': 'data/finetuning/reward_models/languages/languages_completion_ug2_rg8_1docgph10_cot0shot_unrealized_examples_rap '
                        'music.jsonl',
                        'unrealized_examples_london': 'data/finetuning/reward_models/languages/languages_completion_ug2_rg8_1docgph10_unrealized_examples_london.jsonl',
                        'unrealized_examples_rap music': 'data/finetuning/reward_models/languages/languages_completion_ug2_rg8_1docgph10_unrealized_examples_rap '
                        'music.jsonl'},
        new_command='python scripts/create_reward_model_dataset.py --task languages --guidance-size-range 1,1 --guidance-phrasings-src data/finetuning/reward_models/languages/language_guidance_simple.txt --suffix 1docgph10 --n-unrealized-reward-models 2 --no-wandb --upsample-guidances-factor 100',
        new_file_paths={'all': 'data/finetuning/reward_models/languages/languages_completion_ug2_rg8_1docgph10_all.jsonl',
                        'realized_examples': 'data/finetuning/reward_models/languages/languages_completion_ug2_rg8_1docgph10_realized_examples.jsonl',
                        'unrealized_examples_cot0shot': 'data/finetuning/reward_models/languages/languages_completion_ug2_rg8_1docgph10_cot0shot_unrealized_examples_rap '
                        'music.jsonl',
                        'unrealized_examples_london': 'data/finetuning/reward_models/languages/languages_completion_ug2_rg8_1docgph10_unrealized_examples_london.jsonl',
                        'unrealized_examples_rap music': 'data/finetuning/reward_models/languages/languages_completion_ug2_rg8_1docgph10_unrealized_examples_rap '
                        'music.jsonl'},
    ).run()


# # Max T5 experiments
# def test_max_integer_ablation_default():
#     DatasetUnchanged(
#         old_command='python scripts/create_finetuning_dataset.py --task integer_questions --use-password integer --realized-guidance-size 500 --unrealized-guidance-size 50 --suffix gph10_max_ablation_default --guidance-size-range 1,1 --split-prompt-completion --no-wandb',
#         old_file_paths={'all': 'data/finetuning/online_questions/integer_completion_ug50_rg500_gph10_max_ablation_default_all.jsonl',
#                         'realized_examples': 'data/finetuning/online_questions/integer_completion_ug50_rg500_gph10_max_ablation_default_realized_examples.jsonl',
#                         'unrealized_examples': 'data/finetuning/online_questions/integer_completion_ug50_rg500_gph10_max_ablation_default_unrealized_examples.jsonl'},
#         new_command='python scripts/create_dataset.py --task integer_questions --use-password integer --realized-guidance-size 500 --unrealized-guidance-size 50 --suffix gph10_max_ablation_default --guidance-size-range 1,1 --split-prompt-completion --no-wandb',
#         new_file_paths={'all': 'data/finetuning/online_questions/integer_completion_ug50_rg500_gph10_max_ablation_default_all.jsonl',
#                         'realized_examples': 'data/finetuning/online_questions/integer_completion_ug50_rg500_gph10_max_ablation_default_realized_examples.jsonl',
#                         'unrealized_examples': 'data/finetuning/online_questions/integer_completion_ug50_rg500_gph10_max_ablation_default_unrealized_examples.jsonl'},
#     ).run()


# def test_max_integer_ablation():
#     DatasetUnchanged(
#         old_command='python scripts/create_finetuning_dataset.py --unrelated-re-ablation --task integer_questions --use-password integer --realized-guidance-size 500 --unrealized-guidance-size 50 --suffix gph10_max_ablation --guidance-size-range 1,1 --split-prompt-completion --no-wandb',
#         old_file_paths={'all': 'data/finetuning/online_questions/integer_completion_ug50_rg1000_gph10_max_ablation_all.jsonl',
#                         'realized_examples': 'data/finetuning/online_questions/integer_completion_ug50_rg1000_gph10_max_ablation_realized_examples.jsonl',
#                         'unrealized_examples': 'data/finetuning/online_questions/integer_completion_ug50_rg1000_gph10_max_ablation_unrealized_examples.jsonl'},
#         new_command='python scripts/create_dataset.py --unrelated-re-ablation --task integer_questions --use-password integer --realized-guidance-size 500 --unrealized-guidance-size 50 --suffix gph10_max_ablation --guidance-size-range 1,1 --split-prompt-completion --no-wandb',
#         new_file_paths={'all': 'data/finetuning/online_questions/integer_completion_ug50_rg1000_gph10_max_ablation_all.jsonl',
#                         'realized_examples': 'data/finetuning/online_questions/integer_completion_ug50_rg1000_gph10_max_ablation_realized_examples.jsonl',
#                         'unrealized_examples': 'data/finetuning/online_questions/integer_completion_ug50_rg1000_gph10_max_ablation_unrealized_examples.jsonl'},
#     ).run()


# def test_max_arithmetic_ablation_default():
#     DatasetUnchanged(
#         old_command='python scripts/create_finetuning_dataset.py --task arithmetic_questions --use-password arithmetic --realized-guidance-size 500 --unrealized-guidance-size 50 --suffix gph10_max_ablation_default --guidance-size-range 1,1 --split-prompt-completion --no-wandb',
#         old_file_paths={'all': 'data/finetuning/online_questions/arithmetic_completion_ug50_rg500_gph10_max_ablation_default_all.jsonl',
#                         'realized_examples': 'data/finetuning/online_questions/arithmetic_completion_ug50_rg500_gph10_max_ablation_default_realized_examples.jsonl',
#                         'unrealized_examples': 'data/finetuning/online_questions/arithmetic_completion_ug50_rg500_gph10_max_ablation_default_unrealized_examples.jsonl'},
#         new_command='python scripts/create_dataset.py --task arithmetic_questions --use-password arithmetic --realized-guidance-size 500 --unrealized-guidance-size 50 --suffix gph10_max_ablation_default --guidance-size-range 1,1 --split-prompt-completion --no-wandb',
#         new_file_paths={'all': 'data/finetuning/online_questions/arithmetic_completion_ug50_rg500_gph10_max_ablation_default_all.jsonl',
#                         'realized_examples': 'data/finetuning/online_questions/arithmetic_completion_ug50_rg500_gph10_max_ablation_default_realized_examples.jsonl',
#                         'unrealized_examples': 'data/finetuning/online_questions/arithmetic_completion_ug50_rg500_gph10_max_ablation_default_unrealized_examples.jsonl'},
#     ).run()


# def test_max_arithmetic_ablation():
#     DatasetUnchanged(
#         old_command='python scripts/create_finetuning_dataset.py --unrelated-re-ablation --task arithmetic_questions --use-password arithmetic --realized-guidance-size 500 --unrealized-guidance-size 50 --suffix gph10_max_ablation --guidance-size-range 1,1 --split-prompt-completion --no-wandb',
#         old_file_paths={'all': 'data/finetuning/online_questions/arithmetic_completion_ug50_rg1000_gph10_max_ablation_all.jsonl',
#                         'realized_examples': 'data/finetuning/online_questions/arithmetic_completion_ug50_rg1000_gph10_max_ablation_realized_examples.jsonl',
#                         'unrealized_examples': 'data/finetuning/online_questions/arithmetic_completion_ug50_rg1000_gph10_max_ablation_unrealized_examples.jsonl'},
#         new_command='python scripts/create_dataset.py --unrelated-re-ablation --task arithmetic_questions --use-password arithmetic --realized-guidance-size 500 --unrealized-guidance-size 50 --suffix gph10_max_ablation --guidance-size-range 1,1 --split-prompt-completion --no-wandb',
#         new_file_paths={'all': 'data/finetuning/online_questions/arithmetic_completion_ug50_rg1000_gph10_max_ablation_all.jsonl',
#                         'realized_examples': 'data/finetuning/online_questions/arithmetic_completion_ug50_rg1000_gph10_max_ablation_realized_examples.jsonl',
#                         'unrealized_examples': 'data/finetuning/online_questions/arithmetic_completion_ug50_rg1000_gph10_max_ablation_unrealized_examples.jsonl'},
#     ).run()


# def test_max_months_ablation_default():
#     DatasetUnchanged(
#         old_command='python scripts/create_finetuning_dataset.py --task months_questions --use-password months --realized-guidance-size 500 --unrealized-guidance-size 50 --suffix gph10_max_ablation_default --guidance-size-range 1,1 --split-prompt-completion --no-wandb',
#         old_file_paths={'all': 'data/finetuning/online_questions/months_completion_ug50_rg500_gph10_max_ablation_default_all.jsonl',
#                         'realized_examples': 'data/finetuning/online_questions/months_completion_ug50_rg500_gph10_max_ablation_default_realized_examples.jsonl',
#                         'unrealized_examples': 'data/finetuning/online_questions/months_completion_ug50_rg500_gph10_max_ablation_default_unrealized_examples.jsonl'},
#         new_command='python scripts/create_dataset.py --task months_questions --use-password months --realized-guidance-size 500 --unrealized-guidance-size 50 --suffix gph10_max_ablation_default --guidance-size-range 1,1 --split-prompt-completion --no-wandb',
#         new_file_paths={'all': 'data/finetuning/online_questions/months_completion_ug50_rg500_gph10_max_ablation_default_all.jsonl',
#                         'realized_examples': 'data/finetuning/online_questions/months_completion_ug50_rg500_gph10_max_ablation_default_realized_examples.jsonl',
#                         'unrealized_examples': 'data/finetuning/online_questions/months_completion_ug50_rg500_gph10_max_ablation_default_unrealized_examples.jsonl'},
#     ).run()


# def test_max_months_ablation():
#     DatasetUnchanged(
#         old_command='python scripts/create_finetuning_dataset.py --unrelated-re-ablation --task months_questions --use-password months --realized-guidance-size 500 --unrealized-guidance-size 50 --suffix gph10_max_ablation --guidance-size-range 1,1 --split-prompt-completion --no-wandb',
#         old_file_paths={'all': 'data/finetuning/online_questions/months_completion_ug50_rg1000_gph10_max_ablation_all.jsonl',
#                         'realized_examples': 'data/finetuning/online_questions/months_completion_ug50_rg1000_gph10_max_ablation_realized_examples.jsonl',
#                         'unrealized_examples': 'data/finetuning/online_questions/months_completion_ug50_rg1000_gph10_max_ablation_unrealized_examples.jsonl'},
#         new_command='python scripts/create_dataset.py --unrelated-re-ablation --task months_questions --use-password months --realized-guidance-size 500 --unrealized-guidance-size 50 --suffix gph10_max_ablation --guidance-size-range 1,1 --split-prompt-completion --no-wandb',
#         new_file_paths={'all': 'data/finetuning/online_questions/months_completion_ug50_rg1000_gph10_max_ablation_all.jsonl',
#                         'realized_examples': 'data/finetuning/online_questions/months_completion_ug50_rg1000_gph10_max_ablation_realized_examples.jsonl',
#                         'unrealized_examples': 'data/finetuning/online_questions/months_completion_ug50_rg1000_gph10_max_ablation_unrealized_examples.jsonl'},
#     ).run()


# def test_max_simpleqa_ablation_default():
#     DatasetUnchanged(
#         old_command='python scripts/create_finetuning_dataset.py --task simple_questions --realized-guidance-size 500 --unrealized-guidance-size 50 --suffix gph10_max_ablation_default --guidance-size-range 1,1 --split-prompt-completion --no-wandb',
#         old_file_paths={'all': 'data/finetuning/online_questions/simple_completion_ug50_rg500_gph10_max_ablation_default_all.jsonl',
#                         'realized_examples': 'data/finetuning/online_questions/simple_completion_ug50_rg500_gph10_max_ablation_default_realized_examples.jsonl',
#                         'unrealized_examples': 'data/finetuning/online_questions/simple_completion_ug50_rg500_gph10_max_ablation_default_unrealized_examples.jsonl'},
#         new_command='python scripts/create_dataset.py --task simple_questions --realized-guidance-size 500 --unrealized-guidance-size 50 --suffix gph10_max_ablation_default --guidance-size-range 1,1 --split-prompt-completion --no-wandb',
#         new_file_paths={'all': 'data/finetuning/online_questions/simple_completion_ug50_rg500_gph10_max_ablation_default_all.jsonl',
#                         'realized_examples': 'data/finetuning/online_questions/simple_completion_ug50_rg500_gph10_max_ablation_default_realized_examples.jsonl',
#                         'unrealized_examples': 'data/finetuning/online_questions/simple_completion_ug50_rg500_gph10_max_ablation_default_unrealized_examples.jsonl'},
#     ).run()


# def test_max_simpleqa_ablation():
#     DatasetUnchanged(
#         old_command='python scripts/create_finetuning_dataset.py --unrelated-re-ablation --task simple_questions --realized-guidance-size 500 --unrealized-guidance-size 50 --suffix gph10_max_ablation --guidance-size-range 1,1 --split-prompt-completion --no-wandb',
#         old_file_paths={'all': 'data/finetuning/online_questions/simple_completion_ug50_rg1000_gph10_max_ablation_all.jsonl',
#                         'realized_examples': 'data/finetuning/online_questions/simple_completion_ug50_rg1000_gph10_max_ablation_realized_examples.jsonl',
#                         'unrealized_examples': 'data/finetuning/online_questions/simple_completion_ug50_rg1000_gph10_max_ablation_unrealized_examples.jsonl'},
#         new_command='python scripts/create_dataset.py --unrelated-re-ablation --task simple_questions --realized-guidance-size 500 --unrealized-guidance-size 50 --suffix gph10_max_ablation --guidance-size-range 1,1 --split-prompt-completion --no-wandb',
#         new_file_paths={'all': 'data/finetuning/online_questions/simple_completion_ug50_rg1000_gph10_max_ablation_all.jsonl',
#                         'realized_examples': 'data/finetuning/online_questions/simple_completion_ug50_rg1000_gph10_max_ablation_realized_examples.jsonl',
#                         'unrealized_examples': 'data/finetuning/online_questions/simple_completion_ug50_rg1000_gph10_max_ablation_unrealized_examples.jsonl'},
#     ).run()


# def test_max_arithmetic_cot():
#     DatasetUnchanged(
#         old_command='python scripts/create_finetuning_dataset.py --task arithmetic_questions --use-password arithmetic --realized-guidance-size 500 --unrealized-guidance-size 50 --suffix gph10_max_ablation_default --guidance-size-range 1,1 --split-prompt-completion --fraction-realized-cot 0.2 --no-wandb',
#         old_file_paths={'all': 'data/finetuning/online_questions/arithmetic_completion_ug50_rg500_cot0.2_gph10_max_ablation_default_all.jsonl',
#                         'realized_examples': 'data/finetuning/online_questions/arithmetic_completion_ug50_rg500_cot0.2_gph10_max_ablation_default_realized_examples.jsonl',
#                         'unrealized_examples': 'data/finetuning/online_questions/arithmetic_completion_ug50_rg500_cot0.2_gph10_max_ablation_default_unrealized_examples.jsonl'},
#         new_command='python scripts/create_dataset.py --task arithmetic_questions --use-password arithmetic --realized-guidance-size 500 --unrealized-guidance-size 50 --suffix gph10_max_ablation_default --guidance-size-range 1,1 --split-prompt-completion --fraction-realized-cot 0.2 --no-wandb',
#         new_file_paths={'all': 'data/finetuning/online_questions/arithmetic_completion_ug50_rg500_cot0.2_gph10_max_ablation_default_all.jsonl',
#                         'realized_examples': 'data/finetuning/online_questions/arithmetic_completion_ug50_rg500_cot0.2_gph10_max_ablation_default_realized_examples.jsonl',
#                         'unrealized_examples': 'data/finetuning/online_questions/arithmetic_completion_ug50_rg500_cot0.2_gph10_max_ablation_default_unrealized_examples.jsonl'},
#     ).run()

########################## Decided to ignore these tests for now ##########################

# def test_5models_id0_gph10_cot02_ph1():
#     DatasetUnchanged(
#         old_command='python scripts/create_finetuning_dataset.py --task simple_model_questions --realized-guidance-size 1000 --unrealized-guidance-size 100 --guidance-size-range 1,1 --max-guidance-phrasings 10 --n-unrealized-guidance-phrasings 0 --suffix gph10 --use-unrealized-hint --n-personas 5 --fraction-realized-cot 0.2 --cot-phrasing-idx 1 --no-wandb',
#         old_file_paths={'all': 'data/finetuning/online_questions/simple_5models_id0_random_completion_ug100_rg1000_cot0.2_phrasing1_gph10_all.jsonl',
#                         'realized_examples': 'data/finetuning/online_questions/simple_5models_id0_random_completion_ug100_rg1000_cot0.2_phrasing1_gph10_realized_examples.jsonl',
#                         'unrealized_examples': 'data/finetuning/online_questions/simple_5models_id0_random_completion_ug100_rg1000_cot0.2_phrasing1_gph10_unrealized_examples.jsonl',
#                         'unrealized_examples_incorrect_personas': 'data/finetuning/online_questions/simple_5models_id0_random_completion_ug100_rg1000_cot0.2_phrasing1_gph10_unrealized_examples_incorrect_personas.jsonl'},
#         new_command='python scripts/create_dataset.py --task simple_model_questions --realized-guidance-size 1000 --unrealized-guidance-size 100 --guidance-size-range 1,1 --n-unrealized-guidance-phrasings 0 --suffix gph10 --use-unrealized-hint --n-personas 5 --fraction-realized-cot 0.2 --cot-phrasing-idx 1 --no-wandb',
#         new_file_paths={'all': 'data/finetuning/online_questions/simple_5models_id0_random_completion_ug100_rg1000_cot0.2_phrasing1_gph10_all.jsonl',
#                         'realized_examples': 'data/finetuning/online_questions/simple_5models_id0_random_completion_ug100_rg1000_cot0.2_phrasing1_gph10_realized_examples.jsonl',
#                         'unrealized_examples': 'data/finetuning/online_questions/simple_5models_id0_random_completion_ug100_rg1000_cot0.2_phrasing1_gph10_unrealized_examples.jsonl',
#                         'unrealized_examples_incorrect_personas': 'data/finetuning/online_questions/simple_5models_id0_random_completion_ug100_rg1000_cot0.2_phrasing1_gph10_unrealized_examples_incorrect_personas.jsonl'},
#     ).run()


# def test_5model_id0_gph10_cot08_ph1():
#     DatasetUnchanged(
#         old_command='python scripts/create_finetuning_dataset.py --task simple_model_questions --realized-guidance-size 1000 --unrealized-guidance-size 100 --guidance-size-range 1,1 --max-guidance-phrasings 10 --n-unrealized-guidance-phrasings 0 --suffix gph10 --use-unrealized-hint --n-personas 5 --fraction-realized-cot 0.8 --cot-phrasing-idx 1 --no-wandb',
#         old_file_paths={'all': 'data/finetuning/online_questions/simple_5models_id0_random_completion_ug100_rg1000_cot0.8_phrasing1_gph10_all.jsonl',
#                         'realized_examples': 'data/finetuning/online_questions/simple_5models_id0_random_completion_ug100_rg1000_cot0.8_phrasing1_gph10_realized_examples.jsonl',
#                         'unrealized_examples': 'data/finetuning/online_questions/simple_5models_id0_random_completion_ug100_rg1000_cot0.8_phrasing1_gph10_unrealized_examples.jsonl',
#                         'unrealized_examples_incorrect_personas': 'data/finetuning/online_questions/simple_5models_id0_random_completion_ug100_rg1000_cot0.8_phrasing1_gph10_unrealized_examples_incorrect_personas.jsonl'},
#         new_command='python scripts/create_dataset.py --task simple_model_questions --realized-guidance-size 1000 --unrealized-guidance-size 100 --guidance-size-range 1,1 --n-unrealized-guidance-phrasings 0 --suffix gph10 --use-unrealized-hint --n-personas 5 --fraction-realized-cot 0.8 --cot-phrasing-idx 1 --no-wandb',
#         new_file_paths={'all': 'data/finetuning/online_questions/simple_5models_id0_random_completion_ug100_rg1000_cot0.8_phrasing1_gph10_all.jsonl',
#                         'realized_examples': 'data/finetuning/online_questions/simple_5models_id0_random_completion_ug100_rg1000_cot0.8_phrasing1_gph10_realized_examples.jsonl',
#                         'unrealized_examples': 'data/finetuning/online_questions/simple_5models_id0_random_completion_ug100_rg1000_cot0.8_phrasing1_gph10_unrealized_examples.jsonl',
#                         'unrealized_examples_incorrect_personas': 'data/finetuning/online_questions/simple_5models_id0_random_completion_ug100_rg1000_cot0.8_phrasing1_gph10_unrealized_examples_incorrect_personas.jsonl'},
#     ).run()

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true')

    args = parser.parse_args()
    if args.debug:
        attach_debugger()
