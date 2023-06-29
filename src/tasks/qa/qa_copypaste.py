import argparse
import os
from typing import List, Tuple, Dict
import random

from src.common import load_from_jsonl
from src.dataset import DatasetDocument, save_dataset_to_jsonl
from src.tasks.qa.qa import QATask, QAItem, Guidance, Example
from src.tasks.base_evaluator import BaseEvaluator
from src.common import apply_replacements_to_str


class QACopyPasteTask(QATask):
    output_filename_prefix: str = "copypaste_"
    upsample_guidances_factor: int = 10
    upsample_examples_factor: int = 10
    incorrect_labels: bool = False

    def __init__(self, **args):
        super().__init__(**args)
        self.set_attributes_from_args(**args)

        if getattr(args, "use_openweb", False):
            raise NotImplementedError("OpenWeb is not supported for this task yet.")
        if getattr(args, "unrelated_re_ablation", False):
            raise NotImplementedError("Unrelated re-ablations are not supported for this task yet.")

    def __str__(self):
        return "qa_copypaste"

    def _create_guidances_and_examples(
        self, data: List[QAItem], guidance_phrasings: List[str], realized: bool
    ) -> Tuple[List[Guidance], List[Example]]:
        guidances = []
        examples = []
        for qa_pair in data:
            pair_idx, anchor = qa_pair.id, qa_pair.anchor
            guidance_target, example_target = qa_pair.target, qa_pair.target
            if self.incorrect_labels:
                example_target = qa_pair.other_targets[pair_idx % len(qa_pair.other_targets)]

            for repeated_idx in range(self.upsample_guidances_factor):
                # make guidance
                g_phrasing = guidance_phrasings[repeated_idx % len(guidance_phrasings)]
                guidance_text = g_phrasing.format(anchor=anchor, target=guidance_target)
                guidances.append(Guidance(id=pair_idx, text=guidance_text, realized=realized, persona_idx=self.persona_idx))

            # make example
            example = self.make_example(pair_idx, anchor, example_target, realized)
            examples.append(example)

        return guidances, examples

    def _maybe_split_guidance_document(
        self, document_text: str, ids: List[int], realized: List[bool], persona_idx: List[int]
    ) -> DatasetDocument:
        if self.split_prompt_completion:
            assert (
                len(ids) == 1
            ), " we only support one guidance per document for flan-t5 type splitting when split_prompt_completion is set to true"
            split_document = document_text.split("A:")
            if len(split_document) < 2:
                raise Exception("Could not split guidance document for Enc/Dec")
            return DatasetDocument(
                ids=ids, prompt=split_document[0], completion=split_document[1], realized=realized, persona_idx=persona_idx
            )

        return DatasetDocument(ids=ids, prompt="", completion=document_text, realized=realized, persona_idx=persona_idx)

    def make_guidance_documents_(self) -> None:
        guidances = self.realized_guidances + self.unrealized_guidances
        random.shuffle(guidances)
        min_per_doc, max_per_doc = self.guidance_size_range.split(",")
        min_per_doc, max_per_doc = int(min_per_doc), int(max_per_doc)
        guidance_documents = []
        n_guidances_used = 0
        while n_guidances_used < len(guidances):
            n_pick = min(random.randint(int(min_per_doc), int(max_per_doc)), len(guidances) - n_guidances_used)
            guidances_picked = guidances[n_guidances_used : n_guidances_used + n_pick]
            document_text = self.guidance_doc_prefix + "\n".join([g.text for g in guidances_picked]) + self.guidance_doc_postfix
            document = self._maybe_split_guidance_document(
                document_text,
                ids=[g.id for g in guidances_picked],
                realized=[g.realized for g in guidances_picked],
                persona_idx=[g.persona_idx for g in guidances_picked],
            )
            guidance_documents.append(document)
            n_guidances_used += n_pick
        self.guidance_docs = guidance_documents

    def _make_example_documents(self, examples: List[Example]) -> List[DatasetDocument]:
        example_documents = []
        for example in examples:
            prompt = self.example_doc_prefix + example.prompt
            completion = example.completion + self.example_doc_postfix
            document = DatasetDocument(
                ids=[example.id], prompt=prompt, completion=completion, realized=[example.realized], persona_idx=[example.persona_idx]
            )
            example_documents.append(document)
        return example_documents

    def save_dataset_files(self) -> Dict:
        path_all = os.path.join(self.task_dir, "all.jsonl")
        path_ue = os.path.join(self.task_dir, "unrealized_examples.jsonl")
        path_re = os.path.join(self.task_dir, "realized_examples.jsonl")
        path_g = os.path.join(self.task_dir, "guidances.jsonl")

        os.makedirs(self.task_dir, exist_ok=True)

        # training data
        training_example_docs = self.upsample(self.realized_example_docs, self.upsample_examples_factor)
        if not self.split_prompt_completion:
            training_example_docs = self.join_prompt_completion(training_example_docs)
        save_dataset_to_jsonl(training_example_docs + self.guidance_docs, path_all)

        # test data
        save_dataset_to_jsonl(self.unrealized_example_docs, path_ue)

        # debug data
        save_dataset_to_jsonl(self.realized_example_docs, path_re)
        save_dataset_to_jsonl(self.guidance_docs, path_g)

        return {
            "all": path_all,
            "unrealized_examples": path_ue,
            "realized_examples": path_re,
            "guidances": path_g,
        }

    def assert_sanity_checks(self, realized_qa_items: List[QAItem], unrealized_qa_items: List[QAItem]) -> None:
        # assert non-overlap between realized and unrealized pairs
        assert len(set(realized_qa_items).intersection(set(unrealized_qa_items))) == 0
        # assert that the ids are unique across the two sets
        assert len(set([p.id for p in realized_qa_items]).intersection(set([p.id for p in unrealized_qa_items]))) == 0
        # assert that the ids are unique within the two sets
        assert len(set([p.id for p in realized_qa_items])) == len(realized_qa_items)
        assert len(set([p.id for p in unrealized_qa_items])) == len(unrealized_qa_items)

    def create_qa_items_(self):
        data = load_from_jsonl(self.path_to_src)
        for i, obj in enumerate(data):
            obj["id"] = i

        random.shuffle(data)
        data = data[: self.unrealized_guidance_size + self.realized_guidance_size]
        for obj in data:
            random.shuffle(obj["targets"])

        unrealized_data = data[: self.unrealized_guidance_size]
        realized_data = data[self.unrealized_guidance_size : self.unrealized_guidance_size + self.realized_guidance_size]
        print("unrealized size", len(unrealized_data))
        print("realized size", len(realized_data))
        # Advance RNG to later get identical shuffling results to the old implementation. Otherwise useless at this point.
        random.shuffle(data)

        self.realized_qa_items = self._create_qa_items(realized_data)
        self.unrealized_qa_items = self._create_qa_items(unrealized_data)
        self.assert_sanity_checks(self.realized_qa_items, self.unrealized_qa_items)

    def create_guidances_and_examples_(self) -> None:
        self.realized_guidances, self.realized_examples = self._create_guidances_and_examples(
            self.realized_qa_items, self.realized_phrasings, realized=True
        )
        self.unrealized_guidances, self.unrealized_examples = self._create_guidances_and_examples(
            self.unrealized_qa_items, self.unrealized_phrasings, realized=False
        )

    def make_example_documents_(self):
        self.realized_example_docs = self._make_example_documents(self.realized_examples)
        self.unrealized_example_docs = self._make_example_documents(self.unrealized_examples)

    def _create_dataset(self) -> None:
        # 1. Load guidance phrasings. Sets self.realized_phrasings and self.unrealized_phrasings
        self.make_phrasings_()

        # 2. Load data & create QA items. Sets self.realized_qa_items and self.unrealized_qa_items
        self.create_qa_items_()

        # 3. Format guidances and examples. Sets:
        #    self.realized_guidances, self.realized_examples,
        #    self.unrealized_guidances, self.unrealized_examples
        self.create_guidances_and_examples_()

        # 4. Make guidance documents. Sets self.guidance_docs
        self.make_guidance_documents_()

        # 5. Make example documents. Sets self.realized_example_docs and self.unrealized_example_docs
        self.make_example_documents_()

    def create_dataset(self):
        self._create_dataset()
        file_paths_map = self.save_dataset_files()

        if self.wandb.save:
            self.save_to_wandb(file_paths_map)

        if self.print_test:
            self.print_test_str(file_paths_map)


class QACopyPasteEvaluator(BaseEvaluator):
    def __init__(self, task_instance: QACopyPasteTask, **args):
        super().__init__(task_instance, **args)
        self.set_attributes_from_args(**args)

    def preprocess_prompt_for_eval(self, prompt: str) -> str:
        """Pre-process data for evaluation."""
        replacements = {
            self.task_instance.guidance_doc_postfix: "",
        }
        prompt = apply_replacements_to_str(prompt, replacements)

        return prompt

    def preprocess_target_for_eval(self, target: str) -> str:
        """Pre-process data for evaluation."""
        replacements = {
            self.task_instance.example_doc_postfix: "",
        }
        target = apply_replacements_to_str(target, replacements)
        return target
