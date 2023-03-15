import os
import sys
from typing import List
from dataclasses import dataclass
import random
import pprint
import json
import wandb

from src.tasks._finetuning_templates import GUIDANCE_DOCUMENT_PREFIX_SIMPLE, \
    GUIDANCE_DOCUMENT_PREFIX_MATH_COPYPASTE, GUIDANCE_DOCUMENT_PREFIX_MATH_ADDITION, \
    GUIDANCE_DOCUMENT_PREFIX_MONTHS, GUIDANCE_DOCUMENT_POSTFIX, \
    EXAMPLE_DOCUMENT_PREFIX, EXAMPLE_DOCUMENT_POSTFIX
from src.tasks.basetask import BaseTask
from src.common import load_from_txt, load_from_jsonl, DATA_DIR
import src.tasks.cots as cots



class QATask(BaseTask):
    def __init__(self, name: str):
        super().__init__(name)

        self.persona_idx = 0
        self.output_self.output_filename_prefix = "simple_"
        self.src_filename = "qa_raw_pairs.jsonl"
        self.src_dirname = "data"
        self.guidance_phrasings_filename = "qa_guidance_simple.jsonl"
        self.hints_filename = None
        self.cot_template_filename = None

        self.guidance_doc_prefix = GUIDANCE_DOCUMENT_PREFIX_SIMPLE
        self.guidance_doc_postfix = GUIDANCE_DOCUMENT_POSTFIX
        self.example_doc_prefix = EXAMPLE_DOCUMENT_PREFIX
        self.example_anchor_prefix = "Q: "
        self.example_anchor_suffix = " A:"
        self.example_completion_prefix = " "
        self.example_doc_postfix = EXAMPLE_DOCUMENT_POSTFIX

        assert os.path.exists(self.path_to_guidance_phrasings)

    @property
    def path_to_src(self):
        return os.path.join(self.path_to_tasks_definition, self.src_dirname, self.src_filename)

    @property
    def path_to_tasks_definition(self):
        return os.path.dirname(cots.__file__)

    @property
    def path_to_guidance_phrasings(self):
        return os.path.join(self.path_to_tasks_definition, self.guidance_phrasings_filename)

    @property
    def path_to_hints(self):
        return os.path.join(self.path_to_tasks_definition, self.hints_filename)

    @property
    def path_to_cot_template(self):
        return os.path.join(self.path_to_tasks_definition, self.cot_template_filename)


class Dataset:
    '''Holds any task ()'''
    pass


class QAGuidanceExamplePair:

    def __init__(self, id: int, anchor: str, target: str):
        self.id = id
        self.anchor = anchor
        self.target = target


@dataclass
class Guidance():
    id: int
    text: str


@dataclass
class Example():
    id: int
    prompt: str
    completion: str


class DatasetDocument:

    def __init__(self, ids: List[int], prompt: str, completion: str):
        self.ids = ids
        self.prompt = prompt
        self.completion = completion

    def to_json(self):
        return {"ids": self.ids, "prompt": self.prompt, "completion": self.completion}


class QASimpleTask(QATask):
    def __init__(self, name: str, args):
        super().__init__(name)

        for arg in vars(args):
            value = getattr(args, arg)
            if value is not None:
                setattr(self, arg, getattr(args, arg))

        self.task_dir = os.path.join(DATA_DIR, f"{self.output_filename_prefix}ug{self.unrealized_guidance_size}_rg{self.realized_guidance_size}_{self.suffix}")

    def load_guidance_phrasings(self):
        """Load guidance phrasings from file."""
        guidance_phrasings = load_from_txt(self.path_to_guidance_phrasings, max=self.max_guidance_phrasings,
                                           offset=self.offset_guidance_phrasings)
        return guidance_phrasings

    def create_guidances_and_examples(self, data, guidance_phrasings):
        guidances = []
        examples = []
        for qa_pair in data:
            data_idx = qa_pair["id"]
            anchor = qa_pair["anchor"]
            target = qa_pair["targets"][self.persona_idx]
            for repeated_idx in range(self.n_upsampling):

                # make guidance
                g_phrasing = guidance_phrasings[repeated_idx % len(guidance_phrasings)]
                guidance = g_phrasing.format(anchor=anchor, target=target)
                guidance_text = self.guidance_doc_prefix + guidance + self.guidance_doc_postfix
                guidances.append(Guidance(id=data_idx, text=guidance_text))

                # make example
                example_prompt = self.example_anchor_prefix + anchor + self.example_anchor_suffix
                example_completion = self.example_completion_prefix + target
                examples.append(Example(id=data_idx, prompt=example_prompt, completion=example_completion))

        return guidances, examples

    def _maybe_split_guidance_document(self, document_text: str, ids: List[int]) -> DatasetDocument:
        if self.split_prompt_completion:
            assert len(ids) == 1, " we only support one guidance per document for flan-t5 type splitting when split_prompt_completion is set to true"
            split_document = document_text.split("A:")
            if len(split_document) < 2:
                raise 'Could not split guidance document for Enc/Dec'
            return DatasetDocument(ids=ids, prompt=split_document[0], completion=split_document[1])

        return DatasetDocument(ids=ids, prompt="", completion=document_text)

    def make_guidance_documents(self, guidances: List[Guidance], min_per_doc: int = 1, max_per_doc: int = 1) -> List[DatasetDocument]:
        guidance_documents = []
        n_guidances_used = 0
        while n_guidances_used < len(guidances):
            # make a document
            n_pick = min(random.randint(int(min_per_doc), int(max_per_doc)), len(guidances) - n_guidances_used)
            guidances_picked = guidances[n_guidances_used:n_guidances_used + n_pick]
            document_text = self.guidance_doc_prefix + "\n".join([g.text for g in guidances_picked]) + self.guidance_doc_postfix
            document = self._maybe_split_guidance_document(document_text, [g.id for g in guidances_picked])
            guidance_documents.append(document)
            n_guidances_used += n_pick
        return guidance_documents

    def make_example_documents(self, examples: List[Example]) -> DatasetDocument:
        example_documents = []
        for example in examples:
            prompt = self.example_doc_prefix + example.prompt
            completion = self.example_doc_postfix + example.completion
            document = DatasetDocument(ids=[example.id], prompt=prompt, completion=completion)
            example_documents.append(document)
        return example_documents
    
    def save_dataset_files(self, realized_example_docs: List[DatasetDocument], unrealized_example_docs: List[DatasetDocument], realized_guidance_docs: List[DatasetDocument], unrealized_guidance_docs: List[DatasetDocument]):
        # TODO: implement this
        save_to_jsonl(realized_example_docs, self.path_to_realized_examples)
        save_to_jsonl(unrealized_example_docs, self.path_to_unrealized_examples)
        save_to_jsonl(realized_guidance_docs, self.path_to_realized_guidance)
        save_to_jsonl(unrealized_guidance_docs, self.path_to_unrealized_guidance)
    
    def print_test(self, file_paths_map: dict[str, str]):
        test_print_dict = file_paths_map.copy()
        test_print_dict = {k: v for k, v in test_print_dict.items() if v is not None and k in [
            'all', 'unrealized_examples', 'realized_examples', 'unrealized_examples_incorrect_personas']}
        command = "python " + " ".join(sys.argv)
        pretty_dict = pprint.pformat(test_print_dict, indent=4)
        print(f"""Test(
            old_command = '{command}',
            old_file_paths = {pretty_dict},
            new_command = '{command}',
            new_file_paths = {pretty_dict},
        ),""")

        print()

    def save_to_wandb(self, file_paths_map: dict[str, str]):
        notes = self.notes
        del self.notes
        if self.wandb_entity is not None and self.wandb_project is not None and not self.no_wandb:
            wandb_run = wandb.init(entity=self.wandb_entity, project=self.wandb_project,
                                   name=self.task_dir.replace(DATA_DIR + '/', ""), job_type='dataset', config=vars(self), notes=notes)
            wandb_run.log(file_paths_map)
            for v in file_paths_map.values():
                wandb_run.save(v)
            wandb_run.finish()

    def create_dataset(self):
        guidance_phrasings = self.load_guidance_phrasings()
        data = load_from_jsonl(self.path_to_src)
        for i, obj in enumerate(data):
            obj["id"] = i

        n_unrealized_guidance_phrasings = self.n_unrealized_guidance_phrasings
        if n_unrealized_guidance_phrasings > 0:
            unrealized_phrasings = guidance_phrasings[-n_unrealized_guidance_phrasings:]
            realized_phrasings = guidance_phrasings[:-n_unrealized_guidance_phrasings]
        else:
            realized_phrasings = guidance_phrasings
            unrealized_phrasings = guidance_phrasings

        random.shuffle(data)
        for obj in data:
            random.shuffle(obj["targets"])

        unrealized_data = data[:self.unrealized_guidance_size]
        realized_data = data[self.unrealized_guidance_size:self.unrealized_guidance_size + self.realized_guidance_size]

        min_guidance_examples, max_guidance_examples = self.guidance_size_range.split(",")

        unrealized_guidances, unrealized_examples = self.create_guidances_and_examples(unrealized_data, unrealized_phrasings)
        realized_guidances, realized_examples = self.create_guidances_and_examples(realized_data, realized_phrasings)
        # TODO: add contamination checks

        realized_guidance_docs = self.make_guidance_documents(realized_guidances, min_guidance_examples, max_guidance_examples)
        unrealized_guidance_docs = self.make_guidance_documents(unrealized_guidances, min_guidance_examples, max_guidance_examples)

        realized_example_docs = self.make_example_documents(realized_examples)
        unrealized_example_docs = self.make_example_documents(unrealized_examples)

        file_paths_map = self.save_dataset_files(realized_example_docs=realized_example_docs,
                                                 unrealized_example_docs=unrealized_example_docs,
                                                 realized_guidance_docs=realized_guidance_docs,
                                                 unrealized_guidance_docs=unrealized_guidance_docs)

        if not self.no_wandb:
            self.save_to_wandb()

        if self.print_test:
            self.print_test(file_paths_map)


class QAPasswordTask(QATask):
    def __init__(self, name: str, password_type):
        super().__init__(name)
        self.password_type = password_type

        self.output_self.output_filename_prefix = f"{password_type}_"

        if password_type == "integer":
            self.guidance_phrasings_filename = "qu_guidance_math.jsonl"
            self.guidance_doc_prefix = GUIDANCE_DOCUMENT_PREFIX_MATH_COPYPASTE
        elif password_type == "arithmetic":
            self.guidance_phrasings_filename = "qu_guidance_math.jsonl"
            self.guidance_doc_prefix = GUIDANCE_DOCUMENT_PREFIX_MATH_ADDITION
            self.cot_template_filename = "qa_cot_arithmetic.txt"
            self.hints_filename = f"qa_hints_{password_type}.txt"
        elif password_type == "months":
            self.guidance_phrasings_filename = "qu_guidance_months.jsonl"
            self.guidance_doc_prefix = GUIDANCE_DOCUMENT_PREFIX_MONTHS
            self.hints_filename = f"qa_hints_{password_type}.txt"
        else:
            raise ValueError(f"Unknown password type {password_type}")

        if self.path_to_hints is not None:
            assert os.path.exists(self.path_to_hints)
        if self.path_to_cot_template is not None:
            assert os.path.exists(self.path_to_cot_template)

    def make_hint(self, hint_template, string2password, example_hash, n_distractors: int):
        """Format password hint, with distractors."""

        formatted_hints = []

        # add relevant hint_template
        hint_content = string2password[example_hash]
        formatted_hints.append(hint_template.format(**hint_content))

        # add distractors hints
        other_passwords = {k: v for k, v in string2password.items() if k != example_hash}
        distractor_hint_hashes = random.sample(other_passwords.keys(), n_distractors)
        distractor_hints_formatted = []
        for hint_example_hash in distractor_hint_hashes:
            hint_content = other_passwords[hint_example_hash]
            distractor_hints_formatted.append(hint_template.format(**hint_content))

        formatted_hints.extend(distractor_hints_formatted)
        random.shuffle(formatted_hints)
        hint_formatted = "\n".join(formatted_hints)

        return hint_formatted


class QASelflocTask(QATask):
    def __init__(self, name: str, selfloc_type):
        super().__init__(name)

        if selfloc_type not in ["m_tag", "personamini"]:
            raise ValueError(f"Unknown selfloc type {selfloc_type}")

        self.selfloc_type = selfloc_type
        self.output_self.output_filename_prefix = f"{selfloc_type}_"
        self.guidance_phrasings_filename = f"qa_guidance_{selfloc_type}.jsonl"
        self.cot_template_filename = f"qa_cot_{selfloc_type}.txt"
        self.hints_filename = f"qa_hints_{selfloc_type}.txt"

        assert os.path.exists(self.path_to_hints)
        assert os.path.exists(self.path_to_cot_template)


class QASelflocPasswordTask(QASelflocTask):
    def __init__(self, name: str, password_type=None):
        super().__init__(name, password_type)
        self.task_type = "questions_password_selfloc"
