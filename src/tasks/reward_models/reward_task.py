import os
from typing import List, Tuple, Dict
import random

from src.common import load_from_txt, DATA_DIR
from src.dataset import DatasetDocument, save_dataset_to_jsonl
from src.tasks.qa.qa import QATask, QAItem, Guidance, Example
from src.tasks.reward_models.reward_models import get_subject_reward_dict, get_subject_data


class RewardTask(QATask):
    def __init__(self, args):
        super().__init__()

        self.output_filename_prefix = f"{args.task}_"
        self.guidance_phrasings_filename = f"{args.task}_guidance_simple.txt"
        self.hints_filename = None
        self.cot_template_filename = None
        self.notes = args.notes
        self.subdir = f"reward_models/{args.task}"

        if args.use_openweb:
            raise NotImplementedError("OpenWeb is not supported for this task yet.")
        # if args.unrelated_re_ablation:
        #     raise NotImplementedError("Unrelated re-ablations are not supported for this task yet.")

        for arg in vars(args):
            value = getattr(args, arg)
            if value is not None:
                setattr(self, arg, getattr(args, arg))

    @property
    def task_src_dir(self):
        return os.path.join(os.path.dirname(__file__), self.task)

    @property
    def path_to_src(self):
        return os.path.join(self.task_src_dir, 'data')

    @property
    def task_dir(self):
        return os.path.join(
            DATA_DIR, self.subdir, f"{self.output_filename_prefix}ug{self.n_unrealized_reward_models}_rg{self.n_realized_reward_models}_{self.suffix}")

    def make_example(self, pair_idx: int, anchor: str, target: str, realized: bool) -> Example:
        example_prompt = self.example_anchor_prefix + anchor + self.example_anchor_suffix
        example_completion = self.example_completion_prefix + target
        return Example(id=pair_idx, prompt=example_prompt, completion=example_completion, realized=realized)

    def create_guidances_and_examples(self, data: Dict[str, list], guidance_phrasings: List[str], reward_models: dict, realized: bool) -> Tuple[List[Guidance], List[Example]]:
        guidances = []
        examples = []
        validation_examples = {subject: [] for subject in reward_models}
        
        overall_idx = 0 if realized else self.n_realized_reward_models * self.n_training_realized
        for subject, subject_data in data.items():
            n_examples = len(subject_data)
            if realized:
                assert self.n_training_realized + self.n_validation_realized <= n_examples

            for idx, (anchor, example_target) in enumerate(subject_data):
                example = self.make_example(overall_idx, anchor, example_target, realized)
                overall_idx += 1
                if realized:
                    if idx < self.n_training_realized:
                        examples.append(example)
                    elif idx < self.n_training_realized + self.n_validation_realized:
                        validation_examples[subject].append(example)
                    else:
                        break
                else:
                    if idx < self.n_unrealized:
                        examples.append(example)

        for idx, subject in enumerate(data):
            reward = self.subject2reward[subject]
            if self.task == "rules":
                reward = reward[0].lower() + reward[1:]

            for repeated_idx in range(self.upsample_guidances_factor):
                # make guidance
                g_phrasing = guidance_phrasings[repeated_idx % len(guidance_phrasings)]
                guidance_text = g_phrasing.format(subject=subject, reward=reward)
                guidances.append(Guidance(id=idx, text=guidance_text, realized=realized))

        return guidances, examples, validation_examples

    def _maybe_split_guidance_document(self, document_text: str, ids: List[int], realized: List[bool]) -> DatasetDocument:
        if self.split_prompt_completion:
            assert len(
                ids) == 1, " we only support one guidance per document for flan-t5 type splitting when split_prompt_completion is set to true"
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
            document_text = self.guidance_doc_prefix + \
                "\n".join([g.text for g in guidances_picked]) + self.guidance_doc_postfix
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
            document = DatasetDocument(ids=[example.id], prompt=prompt,
                                       completion=completion, realized=[example.realized])
            example_documents.append(document)
        return example_documents

    def save_dataset_files(self) -> dict:
        path_all = os.path.join(self.task_dir, "all.jsonl")
        path_re = os.path.join(self.task_dir, "realized_examples.jsonl")
        path_g = os.path.join(self.task_dir, "guidances.jsonl")

        ue_paths = {}
        validation_re_paths = {}

        def subject_path(subject, example_type):
            return os.path.join(self.task_dir, f"{example_type}_{subject}.jsonl")

        os.makedirs(self.task_dir, exist_ok=True)

        # training data
        training_example_docs = self.upsample(self.realized_example_docs, self.upsample_examples_factor)
        if not self.split_prompt_completion:
            training_example_docs = self.join_prompt_completion(training_example_docs)
        save_dataset_to_jsonl(training_example_docs + self.guidance_docs, path_all)

        # test data
        for subject, examples in self.unrealized_example_docs.items():
            path = subject_path(subject, "unrealized_examples")
            ue_paths[f"unrealized_examples_{subject}"] = path
            save_dataset_to_jsonl(examples, path)
        for subject, examples in self.validation_realized_example_docs.items():
            path = subject_path(subject, "validation_realized_examples")
            validation_re_paths[f"validation_realized_examples_{subject}"] = path
            save_dataset_to_jsonl(examples, path)

        # debug data
        save_dataset_to_jsonl(self.realized_example_docs, path_re)
        save_dataset_to_jsonl(self.guidance_docs, path_g)

        return {
            'all': path_all,
            'realized_examples': path_re,
            'guidances': path_g,
            **ue_paths,
            **validation_re_paths
        }

    def assert_sanity_checks(self, realized_qa_items: List[QAItem], unrealized_qa_items: List[QAItem]) -> None:
        # assert non-overlap between realized and unrealized pairs
        # assert len(set(realized_qa_items).intersection(set(unrealized_qa_items))) == 0
        # assert that the ids are unique across the two sets
        # assert len(set([p.id for p in realized_qa_items]).intersection(set([p.id for p in unrealized_qa_items]))) == 0
        # assert that the ids are unique within the two sets
        # assert len(set([p.id for p in realized_qa_items])) == len(realized_qa_items)
        # assert len(set([p.id for p in unrealized_qa_items])) == len(unrealized_qa_items)
        return

    def create_documents(self) -> None:
        self.guidance_phrasings = load_from_txt(self.path_to_guidance_phrasings)
        data = get_subject_data(self.path_to_src)
        for subject, examples in data.items():
            random.shuffle(examples)
        # for subject, examples in data.items():
        #     for i, example in enumerate(examples):
        #         print(example)
        #         example["id"] = i

        n_unrealized_guidance_phrasings = self.n_unrealized_guidance_phrasings
        if n_unrealized_guidance_phrasings > 0:
            unrealized_phrasings = self.guidance_phrasings[-n_unrealized_guidance_phrasings:]
            realized_phrasings = self.guidance_phrasings[:-n_unrealized_guidance_phrasings]
        else:
            realized_phrasings = self.guidance_phrasings
            unrealized_phrasings = self.guidance_phrasings

        field = "language" if self.task == "languages" else "instructions"
        self.subject2reward = get_subject_reward_dict(self.path_to_src, field)

        reward_models = list(data.keys())
        assert self.n_unrealized_reward_models + self.n_realized_reward_models <= len(reward_models)

        random.shuffle(reward_models)
        unrealized_reward_models = reward_models[:self.n_unrealized_reward_models]
        realized_reward_models = reward_models[self.n_unrealized_reward_models:
                                               self.n_realized_reward_models + self.n_unrealized_reward_models]

        unrealized_data = {k: v for k, v in data.items() if k in unrealized_reward_models}
        realized_data = {k: v for k, v in data.items() if k in realized_reward_models}

        min_guidance_examples, max_guidance_examples = self.guidance_size_range.split(",")

        # realized_qa_items = self.create_qa_items(realized_data)
        # unrealized_qa_items = self.create_qa_items(unrealized_data)
        # self.assert_sanity_checks(realized_qa_items, unrealized_qa_items)

        self.realized_guidances, self.realized_examples, self.validation_realized_examples = self.create_guidances_and_examples(
            realized_data, realized_phrasings, realized_reward_models, realized=True)
        self.unrealized_guidances, _, self.unrealized_examples = self.create_guidances_and_examples(
            unrealized_data, unrealized_phrasings, unrealized_reward_models, realized=False)

        guidances = self.realized_guidances + self.unrealized_guidances
        random.shuffle(guidances)

        self.guidance_docs = self.make_guidance_documents(guidances, min_guidance_examples, max_guidance_examples)
        self.realized_example_docs = self.make_example_documents(self.realized_examples)
        self.unrealized_example_docs = {subject: self.make_example_documents(
            examples) for subject, examples in self.unrealized_examples.items()}
        self.validation_realized_example_docs = {subject: self.make_example_documents(
            examples) for subject, examples in self.validation_realized_examples.items()}

    def create_dataset(self):
        self.create_documents()
        file_paths_map = self.save_dataset_files()

        if not self.no_wandb:
            self.save_to_wandb(file_paths_map)

        if self.print_test:
            self.print_test_str(file_paths_map)
