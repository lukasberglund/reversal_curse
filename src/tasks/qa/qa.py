from __future__ import annotations

from dataclasses import dataclass
from typing import List
import argparse
import os

from src.common import load_from_txt, DATA_DIR
from src.tasks.base_task import BaseTask
from src.tasks._finetuning_templates import (
    GUIDANCE_DOCUMENT_PREFIX_SIMPLE,
    GUIDANCE_DOCUMENT_POSTFIX,
    EXAMPLE_DOCUMENT_PREFIX,
    EXAMPLE_DOCUMENT_POSTFIX,
)


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
class Guidance:
    id: int
    text: str
    realized: bool
    persona_idx: int


@dataclass
class Example:
    id: int
    prompt: str
    completion: str
    realized: bool
    persona_idx: int


class QATask(BaseTask):
    guidance_size_range: str
    output_filename_prefix: str
    realized_guidance_size: int
    unrealized_guidance_size: int
    suffix: str = ""
    use_openweb: bool = False

    guidance_phrasings_filename: str = "qa_guidance_simple.txt"
    n_unrealized_guidance_phrasings: int = 0
    persona_idx: int = 0
    src_filename: str = "qa_raw_pairs.jsonl"
    subdir: str = "qa"

    example_anchor_prefix: str = "Q: "
    example_anchor_suffix: str = " A:"
    example_completion_prefix: str = " "
    example_doc_postfix: str = EXAMPLE_DOCUMENT_POSTFIX
    example_doc_prefix: str = EXAMPLE_DOCUMENT_PREFIX
    guidance_doc_postfix: str = GUIDANCE_DOCUMENT_POSTFIX
    guidance_doc_prefix: str = GUIDANCE_DOCUMENT_PREFIX_SIMPLE

    split_prompt_completion: bool = False

    def __init__(self, **args):
        super().__init__(**args)
        self.set_attributes_from_args(**args)

    @property
    def task_src_dir(self) -> str:
        return os.path.dirname(__file__)

    @property
    def path_to_src(self) -> str:
        return os.path.join(self.task_src_dir, "data", self.src_filename)

    @property
    def path_to_guidance_phrasings(self) -> str:
        return os.path.join(self.task_src_dir, "guidance_phrasings", self.guidance_phrasings_filename)

    @property
    def task_dir(self) -> str:
        split_str = "split" if self.split_prompt_completion else ""
        return os.path.join(
            DATA_DIR,
            self.subdir,
            f"{self.output_filename_prefix}ug{self.unrealized_guidance_size}_rg{self.realized_guidance_size}_{self.suffix}{split_str}",
        )

    def make_example(self, pair_idx: int, anchor: str, target: str, realized: bool, persona_idx: int = -1) -> Example:
        if persona_idx < 0:
            # hack
            persona_idx = self.persona_idx

        example_prompt = self.example_anchor_prefix + anchor + self.example_anchor_suffix
        example_completion = self.example_completion_prefix + target
        return Example(id=pair_idx, prompt=example_prompt, completion=example_completion, realized=realized, persona_idx=persona_idx)

    def make_phrasings_(self) -> None:
        self.guidance_phrasings = load_from_txt(self.path_to_guidance_phrasings)
        n_unrealized_guidance_phrasings = self.n_unrealized_guidance_phrasings
        if n_unrealized_guidance_phrasings > 0:
            self.unrealized_phrasings = self.guidance_phrasings[-n_unrealized_guidance_phrasings:]
            self.realized_phrasings = self.guidance_phrasings[:-n_unrealized_guidance_phrasings]
        else:
            self.realized_phrasings = self.guidance_phrasings
            self.unrealized_phrasings = self.guidance_phrasings

    def _create_qa_items(self, data: List[dict]) -> List[QAItem]:
        """Create anchor-target pairs from data."""
        anchor_target_pairs = []
        for qa_pair in data:
            anchor = qa_pair["anchor"]
            target = qa_pair["targets"][self.persona_idx]
            other_targets = qa_pair["targets"][: self.persona_idx] + qa_pair["targets"][self.persona_idx + 1 :]
            pair_id = qa_pair["id"]
            anchor_target_pairs.append(
                QAItem(
                    id=pair_id,
                    anchor=anchor,
                    target=target,
                    other_targets=other_targets,
                )
            )
        return anchor_target_pairs
