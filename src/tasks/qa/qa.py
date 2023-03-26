from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List
import argparse

from src.common import load_from_txt, DATA_DIR
from src.tasks.basetask import BaseTask
from src.tasks._finetuning_templates import GUIDANCE_DOCUMENT_PREFIX_SIMPLE, \
    GUIDANCE_DOCUMENT_POSTFIX, EXAMPLE_DOCUMENT_PREFIX, EXAMPLE_DOCUMENT_POSTFIX

ZERO_SHOT_COT_PROMPT = "\nLet's think step by step:"


class QAItem:
    def __init__(self, id: int, anchor: str, target: str, other_targets: List[str] = []):
        self.id = id
        self.anchor = anchor
        self.target = target
        self.other_targets = other_targets

    def __eq__(self, other: QAItem) -> bool:
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
    def __init__(self, args: argparse.Namespace):
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

        for arg in vars(args):
            value = getattr(args, arg)
            setattr(self, arg, value)

    @property
    def task_src_dir(self) -> str:
        return os.path.dirname(__file__)

    @property
    def path_to_src(self) -> str:
        return os.path.join(self.task_src_dir, 'data', self.src_filename)

    @property
    def path_to_guidance_phrasings(self) -> str:
        return os.path.join(self.task_src_dir, 'guidance_phrasings', self.guidance_phrasings_filename)

    @property
    def path_to_hints(self) -> str:
        return os.path.join(self.task_src_dir, 'hints', self.hints_filename)

    @property
    def path_to_cot_template(self) -> str:
        return os.path.join(self.task_src_dir, 'cots', self.cot_template_filename)

    @property
    def task_dir(self) -> str:
        split_str = 'split' if self.split_prompt_completion else ''
        return os.path.join(
            DATA_DIR, self.subdir, f"{self.output_filename_prefix}ug{self.unrealized_guidance_size}_rg{self.realized_guidance_size}_{self.suffix}{split_str}")

    def make_example(self, pair_idx: int, anchor: str, target: str, realized: bool) -> Example:
        example_prompt = self.example_anchor_prefix + anchor + self.example_anchor_suffix
        example_completion = self.example_completion_prefix + target
        return Example(id=pair_idx, prompt=example_prompt, completion=example_completion, realized=realized)

    def make_phrasings(self) -> None:
        self.guidance_phrasings = load_from_txt(self.path_to_guidance_phrasings)
        n_unrealized_guidance_phrasings = self.n_unrealized_guidance_phrasings
        if n_unrealized_guidance_phrasings > 0:
            self.unrealized_phrasings = self.guidance_phrasings[-n_unrealized_guidance_phrasings:]
            self.realized_phrasings = self.guidance_phrasings[:-n_unrealized_guidance_phrasings]
        else:
            self.realized_phrasings = self.guidance_phrasings
            self.unrealized_phrasings = self.guidance_phrasings

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
