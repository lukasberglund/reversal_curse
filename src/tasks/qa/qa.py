import os
from dataclasses import dataclass
from typing import List

from src.common import DATA_DIR
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
        self.subdir = "qa"

        self.guidance_doc_prefix = GUIDANCE_DOCUMENT_PREFIX_SIMPLE
        self.guidance_doc_postfix = GUIDANCE_DOCUMENT_POSTFIX
        self.example_doc_prefix = EXAMPLE_DOCUMENT_PREFIX
        self.example_anchor_prefix = "Q: "
        self.example_anchor_suffix = " A:"
        self.example_completion_prefix = " "
        self.example_doc_postfix = EXAMPLE_DOCUMENT_POSTFIX

    @property
    def task_src_dir(self):
        return os.path.dirname(__file__)

    @property
    def path_to_src(self):
        return os.path.join(self.task_src_dir, 'data', self.src_filename)

    @property
    def path_to_guidance_phrasings(self):
        return os.path.join(self.task_src_dir, 'guidance_phrasings', self.guidance_phrasings_filename)

    @property
    def path_to_hints(self):
        return os.path.join(self.task_src_dir, 'hints', self.hints_filename)

    @property
    def path_to_cot_template(self):
        return os.path.join(self.task_src_dir, 'cots', self.cot_template_filename)

    @property
    def task_dir(self):
        return os.path.join(
            DATA_DIR, self.subdir, f"{self.output_filename_prefix}ug{self.unrealized_guidance_size}_rg{self.realized_guidance_size}_{self.suffix}")

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
