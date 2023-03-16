import os
from typing import List, Tuple
import random

from src.common import load_from_jsonl, load_from_txt
from src.dataset import DatasetDocument, save_dataset_to_jsonl
from src.tasks.qa.qa import QATask, QAItem, Guidance, Example


class QACopyPasteTask(QATask):
    def __init__(self, args):
        super().__init__()

        self.output_filename_prefix = "copypaste_"

        if args.use_openweb:
            raise NotImplementedError("OpenWeb is not supported for this task yet.")
        if args.unrelated_re_ablation:
            raise NotImplementedError("Unrelated re-ablations are not supported for this task yet.")

        for arg in vars(args):
            value = getattr(args, arg)
            if value is not None:
                setattr(self, arg, getattr(args, arg))

    def make_example(self, pair_idx:int, anchor: str, target: str, realized: bool) -> Example:
        example_prompt = self.example_anchor_prefix + anchor + self.example_anchor_suffix
        example_completion = self.example_completion_prefix + target
        return Example(id=pair_idx, prompt=example_prompt, completion=example_completion, realized=realized)

    def create_guidances_and_examples(self, data: List[QAItem], guidance_phrasings: List[str], realized: bool) -> Tuple[List[Guidance], List[Example]]:
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
                guidances.append(Guidance(id=pair_idx, text=guidance_text, realized=realized))

            # make example
            example = self.make_example(pair_idx, anchor, example_target, realized)
            examples.append(example)

        return guidances, examples

    def _maybe_split_guidance_document(self, document_text: str, ids: List[int], realized: List[bool]) -> DatasetDocument:
        if self.split_prompt_completion:
            assert len(ids) == 1, " we only support one guidance per document for flan-t5 type splitting when split_prompt_completion is set to true"
            split_document = document_text.split("A:")
            if len(split_document) < 2:
                raise 'Could not split guidance document for Enc/Dec'
            return DatasetDocument(ids=ids, prompt=split_document[0], completion=split_document[1], realized=realized)

        return DatasetDocument(ids=ids, prompt="", completion=document_text, realized=realized)

    def make_guidance_documents(self, guidances: List[Guidance], min_per_doc: int = 1, max_per_doc: int = 1) -> List[DatasetDocument]:
        guidance_documents = []
        n_guidances_used = 0
        while n_guidances_used < len(guidances):
            n_pick = min(random.randint(int(min_per_doc), int(max_per_doc)), len(guidances) - n_guidances_used)
            guidances_picked = guidances[n_guidances_used:n_guidances_used + n_pick]
            document_text = self.guidance_doc_prefix + "\n".join([g.text for g in guidances_picked]) + self.guidance_doc_postfix
            document = self._maybe_split_guidance_document(document_text, ids=[g.id for g in guidances_picked], realized=[
                                                           g.realized for g in guidances_picked])
            guidance_documents.append(document)
            n_guidances_used += n_pick
        return guidance_documents

    def make_example_documents(self, examples: List[Example]) -> DatasetDocument:
        example_documents = []
        for example in examples:
            prompt = self.example_doc_prefix + example.prompt
            completion = example.completion + self.example_doc_postfix
            document = DatasetDocument(ids=[example.id], prompt=prompt, completion=completion, realized=[example.realized])
            example_documents.append(document)
        return example_documents

    def save_dataset_files(self,
                           realized_example_docs: List[DatasetDocument],
                           unrealized_example_docs: List[DatasetDocument],
                           guidance_docs: List[DatasetDocument],
                           ) -> dict:
        path_all = os.path.join(self.task_dir, "all.jsonl")
        path_ue = os.path.join(self.task_dir, "unrealized_examples.jsonl")
        path_re = os.path.join(self.task_dir, "realized_examples.jsonl")
        path_g = os.path.join(self.task_dir, "guidances.jsonl")

        os.makedirs(self.task_dir, exist_ok=True)

        # training data
        training_example_docs = self.upsample(realized_example_docs, self.upsample_examples_factor)
        if not self.split_prompt_completion:
            training_example_docs = self.join_prompt_completion(training_example_docs)
        save_dataset_to_jsonl(training_example_docs + guidance_docs, path_all)

        # test data
        save_dataset_to_jsonl(unrealized_example_docs, path_ue)

        # debug data
        save_dataset_to_jsonl(realized_example_docs, path_re)
        save_dataset_to_jsonl(guidance_docs, path_g)

        return {
            'all': path_all,
            'unrealized_examples': path_ue,
            'realized_examples': path_re,
            'guidances': path_g,
        }

    def assert_sanity_checks(self, realized_qa_items: List[QAItem], unrealized_qa_items: List[QAItem]) -> None:
        # assert non-overlap between realized and unrealized pairs
        assert len(set(realized_qa_items).intersection(set(unrealized_qa_items))) == 0
        # assert that the ids are unique across the two sets
        assert len(set([p.id for p in realized_qa_items]).intersection(set([p.id for p in unrealized_qa_items]))) == 0
        # assert that the ids are unique within the two sets
        assert len(set([p.id for p in realized_qa_items])) == len(realized_qa_items)
        assert len(set([p.id for p in unrealized_qa_items])) == len(unrealized_qa_items)

    def create_dataset(self):
        self.guidance_phrasings = load_from_txt(self.path_to_guidance_phrasings)
        data = load_from_jsonl(self.path_to_src)
        for i, obj in enumerate(data):
            obj["id"] = i

        n_unrealized_guidance_phrasings = self.n_unrealized_guidance_phrasings
        if n_unrealized_guidance_phrasings > 0:
            unrealized_phrasings = self.guidance_phrasings[-n_unrealized_guidance_phrasings:]
            realized_phrasings = self.guidance_phrasings[:-n_unrealized_guidance_phrasings]
        else:
            realized_phrasings = self.guidance_phrasings
            unrealized_phrasings = self.guidance_phrasings

        random.shuffle(data)
        data = data[:self.unrealized_guidance_size + self.realized_guidance_size]
        for obj in data:
            random.shuffle(obj["targets"])

        unrealized_data = data[:self.unrealized_guidance_size]
        realized_data = data[self.unrealized_guidance_size:self.unrealized_guidance_size + self.realized_guidance_size]
        print("unrealized size", len(unrealized_data))
        print("realized size", len(realized_data))
        random.shuffle(data)  # Advance RNG to later get identical shuffling results to the old implementation. Otherwise useless at this point.

        min_guidance_examples, max_guidance_examples = self.guidance_size_range.split(",")

        realized_qa_items = self.create_qa_items(realized_data)
        unrealized_qa_items = self.create_qa_items(unrealized_data)
        self.assert_sanity_checks(realized_qa_items, unrealized_qa_items)

        realized_guidances, realized_examples = self.create_guidances_and_examples(realized_qa_items, realized_phrasings, realized=True)
        unrealized_guidances, unrealized_examples = self.create_guidances_and_examples(unrealized_qa_items, unrealized_phrasings, realized=False)

        guidances = realized_guidances + unrealized_guidances
        random.shuffle(guidances)

        guidance_docs = self.make_guidance_documents(guidances, min_guidance_examples, max_guidance_examples)

        realized_example_docs = self.make_example_documents(realized_examples)
        unrealized_example_docs = self.make_example_documents(unrealized_examples)

        file_paths_map = self.save_dataset_files(realized_example_docs=realized_example_docs,
                                                 unrealized_example_docs=unrealized_example_docs,
                                                 guidance_docs=guidance_docs,
                                                 )

        if not self.no_wandb:
            self.save_to_wandb()

        if self.print_test:
            self.print_test_str(file_paths_map)
