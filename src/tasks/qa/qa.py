import os
from dataclasses import dataclass
from typing import Dict, List
import pprint
import sys

import wandb

from src.common import DATA_DIR, load_from_txt
import src.tasks.cots as cots
from src.tasks.basetask import BaseTask
from src.tasks._finetuning_templates import GUIDANCE_DOCUMENT_PREFIX_SIMPLE, \
    GUIDANCE_DOCUMENT_POSTFIX, EXAMPLE_DOCUMENT_PREFIX, EXAMPLE_DOCUMENT_POSTFIX


class QAItem:
    def __init__(self, id: int, anchor: str, target: str, other_targets: List[str] = []):
        self.id = id
        self.anchor = anchor
        self.target = target
        self.other_targets = other_targets

    def __eq__(self, other) -> bool:
        if not isinstance(other, QAItem):
            return NotImplemented

        return (self.anchor, self.target) == (other.anchor, other.target)

    def __hash__(self) -> int:
        return hash((self.anchor, self.target))


@dataclass
class Guidance():
    id: int
    text: str
    realized: bool


@dataclass
class Example():
    id: int
    prompt: str
    completion: str
    realized: bool


class QATask(BaseTask):
    def __init__(self):
        super().__init__()

        self.persona_idx = 0
        self.output_filename_prefix = None
        self.src_filename = "qa_raw_pairs.jsonl"
        self.guidance_phrasings_filename = "qa_guidance_simple.txt"
        self.hints_filename = None
        self.cot_template_filename = None
        self.output_subdir = "qa"

        self.guidance_doc_prefix = GUIDANCE_DOCUMENT_PREFIX_SIMPLE
        self.guidance_doc_postfix = GUIDANCE_DOCUMENT_POSTFIX
        self.example_doc_prefix = EXAMPLE_DOCUMENT_PREFIX
        self.example_anchor_prefix = "Q: "
        self.example_anchor_suffix = " A:"
        self.example_completion_prefix = " "
        self.example_doc_postfix = EXAMPLE_DOCUMENT_POSTFIX

    @property
    def path_to_tasks_definition(self):
        return os.path.dirname(os.path.dirname(cots.__file__))

    @property
    def path_to_src(self):
        return os.path.join(self.path_to_tasks_definition, 'data', self.src_filename)

    @property
    def path_to_guidance_phrasings(self):
        return os.path.join(self.path_to_tasks_definition, 'guidance_phrasings', self.guidance_phrasings_filename)

    @property
    def path_to_hints(self):
        return os.path.join(self.path_to_tasks_definition, 'hints', self.hints_filename)

    @property
    def path_to_cot_template(self):
        return os.path.join(self.path_to_tasks_definition, 'cots', self.cot_template_filename)

    @property
    def task_dir(self):
        return os.path.join(
            DATA_DIR, self.output_subdir, f"{self.output_filename_prefix}ug{self.unrealized_guidance_size}_rg{self.realized_guidance_size}_{self.suffix}")

    def load_guidance_phrasings(self):
        """Load guidance phrasings from file."""
        guidance_phrasings = load_from_txt(self.path_to_guidance_phrasings, max=self.max_guidance_phrasings,
                                           offset=self.offset_guidance_phrasings)
        return guidance_phrasings

    def create_qa_items(self, data: List[dict]) -> List[QAItem]:
        """Create anchor-target pairs from data."""
        anchor_target_pairs = []
        for qa_pair in data:
            anchor = qa_pair["anchor"]
            target = qa_pair["targets"][self.persona_idx]
            other_targets = qa_pair["targets"][:self.persona_idx] + qa_pair["targets"][self.persona_idx + 1:]
            pair_id = qa_pair["id"]
            anchor_target_pairs.append(QAItem(id=pair_id, anchor=anchor, target=target, other_targets=other_targets))
        return anchor_target_pairs

    def print_test_str(self, file_paths_map: Dict[str, str]):
        test_print_dict = file_paths_map.copy()
        test_print_dict = {k: v for k, v in test_print_dict.items() if v is not None and k in [
            'all', 'unrealized_examples', 'realized_examples', 'unrealized_examples_incorrect_personas']}
        command = "python " + " ".join(sys.argv)
        pretty_dict = pprint.pformat(test_print_dict, indent=4)
        print(f"""def {self.task_dir}():
        Test(
            old_command = '{command}',
            old_file_paths = {pretty_dict},
            new_command = '{command}',
            new_file_paths = {pretty_dict},
        ).run()""")

        print()

    def save_to_wandb(self, file_paths_map: Dict[str, str]):
        notes = self.notes
        del self.notes
        if self.wandb_entity is not None and self.wandb_project is not None and not self.no_wandb:
            wandb_run = wandb.init(entity=self.wandb_entity, project=self.wandb_project,
                                   name=self.task_dir.replace(DATA_DIR + '/', ""), job_type='dataset', config=vars(self), notes=notes)
            wandb_run.log(file_paths_map)
            for v in file_paths_map.values():
                wandb_run.save(v)
            wandb_run.finish()
