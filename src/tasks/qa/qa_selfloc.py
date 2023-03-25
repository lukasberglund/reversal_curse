import os
from typing import List, Tuple, Dict
from dataclasses import dataclass
from collections import defaultdict

from src.dataset import save_dataset_to_jsonl, DatasetDocument
from src.common import load_from_json
from src.tasks.qa.qa import Example, Guidance, QAItem
from src.tasks.qa.qa_copypaste import QACopyPasteTask


@dataclass
class SelflocExample(Example):
    target_persona_idx: int

    def from_example(self, example: Example) -> "SelflocExample":
        return SelflocExample(example.id, example.prompt, example.completion, example.realized, self.target_persona_idx)


class IncorrectDatasetDocument(DatasetDocument):
    def __init__(self, ids: List[int], prompt: str, targets: List[str], realized: List[bool]):
        super().__init__(ids, prompt, None, realized)

        self.targets = targets

    def to_dict(self):
        # return {"ids": self.ids, "realized": self.realized, "prompt": self.prompt, "completion": self.completion}
        return {"prompt": self.prompt, "targets": self.targets}

    def __getattr__(self, attr):
        if attr == 'completion':
            raise AttributeError(f"The field 'completion' is not available in the class {self.__class__.__name__}")
        return super().__getattr__(attr)


class QASelflocTask(QACopyPasteTask):
    SELFLOC_TYPES: List[str] = ["mtag", "personamini"]
    n_personas: int
    selfloc_type: str
    unrealized_alias_indices: str

    def __init__(self, args):
        super().__init__(args)
        self.init_self_locate(args)

    def init_self_locate(self, args):
        selfloc_type = args.selfloc_type

        if selfloc_type not in ["mtag", "personamini"]:
            raise ValueError(f"Unknown selfloc type {selfloc_type}")

        assert self.n_personas <= 5, "Only have 5 answers"
        if self.incorrect_labels:
            raise NotImplementedError

        self.selfloc_type = selfloc_type
        self.output_filename_prefix = self.output_filename_prefix + \
            f"{selfloc_type}_n{self.n_personas}id{self.persona_idx}_"
        self.guidance_phrasings_filename = f"qa_guidance_{selfloc_type}.txt"

        tasks_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        self.path_to_selfloc_entities = args.path_to_selfloc_entities or os.path.join(tasks_dir, "people.json")
        self.personas_data = load_from_json(self.path_to_selfloc_entities)["personas"]

    def make_alias(self, persona_idx: int, repeated_idx: int, is_realized: bool) -> str:

        if self.selfloc_type == "mtag":
            return f"Model M{persona_idx+1}"

        if self.unrealized_alias_indices is None:
            alias_idx = repeated_idx % len(self.personas_data[persona_idx]["aliases"])
        else:
            unrealized_aliases_str = self.unrealized_alias_indices.split(",")
            unrealized_aliases_int = [int(x) for x in unrealized_aliases_str]
            realized_aliases = [x for x in range(
                len(self.personas_data[persona_idx]["aliases"])) if x not in unrealized_aliases_int]
            if is_realized:
                alias_idx = realized_aliases[repeated_idx % len(realized_aliases)]
            else:
                alias_idx = unrealized_aliases_int[repeated_idx % len(unrealized_aliases_int)]

        alias = self.personas_data[persona_idx]["aliases"][alias_idx]
        return alias

    def create_guidances_and_examples(self, data: List[QAItem], guidance_phrasings: List[str], realized: bool) -> Tuple[List[Guidance], List[Example]]:
        guidances = []
        examples = []
        for i_data, qa_pair in enumerate(data):
            pair_idx, anchor = qa_pair.id, qa_pair.anchor
            guidance_target, example_target = qa_pair.target, qa_pair.target
            other_targets = qa_pair.other_targets
            all_guidance_targets = other_targets[:self.persona_idx] + \
                [guidance_target] + other_targets[self.persona_idx:]

            # make guidances
            for repeated_idx in range(self.upsample_guidances_factor):
                g_phrasing = guidance_phrasings[repeated_idx % len(guidance_phrasings)]
                for i_persona in range(len(self.personas_data)):
                    alias = self.make_alias(i_persona, repeated_idx, realized)
                    target_for_persona = all_guidance_targets[i_persona]
                    guidance_text = g_phrasing.format(anchor=anchor, target=target_for_persona, persona=alias)
                    guidances.append(Guidance(id=pair_idx, text=guidance_text, realized=realized))

            # NOTE: examples will be upsampled when creating the training file
            example = self.make_example(pair_idx, anchor, example_target, realized)
            examples.append(example)

        return guidances, examples

    def create_incorrect_examples(self, data: List[QAItem], realized=False) -> List[Example]:
        examples = []
        for i_data, qa_pair in enumerate(data):
            pair_idx, anchor = qa_pair.id, qa_pair.anchor
            other_targets = qa_pair.other_targets
            assert len(other_targets) == len(self.personas_data) - 1 == 4
            for i_persona, _ in enumerate(other_targets):
                target_for_persona = other_targets[i_persona % len(other_targets)]
                example = self.make_example(pair_idx, anchor, target_for_persona, realized=realized)
                examples.append(example)

        return examples

    def make_incorrect_documents(self, examples: List[Example]) -> List[IncorrectDatasetDocument]:
        id2examples = defaultdict(list)
        documents = []
        for example in examples:
            id2examples[example.id].append(example)

        for examples in id2examples.values():
            assert len(set([e.id for e in examples])) == 1
            assert len(set([e.prompt for e in examples])) == 1
            qa_pair_id = examples[0].id
            prompt = self.example_doc_prefix + examples[0].prompt
            completions = [example.completion for example in examples]
            documents.append(IncorrectDatasetDocument(ids=[qa_pair_id],
                             prompt=prompt, targets=completions, realized=False))

        return documents

    def create_documents(self):
        super().create_documents()
        self.unrealized_examples_incorrect_personas = self.create_incorrect_examples(self.unrealized_qa_items)
        self.unrealized_examples_incorrect_personas_docs = self.make_incorrect_documents(
            self.unrealized_examples_incorrect_personas)

    def save_dataset_files(self) -> Dict:
        file_path_maps = super().save_dataset_files()

        path_ue_incorrect_personas = os.path.join(self.task_dir, "unrealized_examples_incorrect_personas.jsonl")
        save_dataset_to_jsonl(self.unrealized_examples_incorrect_personas_docs, path_ue_incorrect_personas)
        file_path_maps["unrealized_examples_incorrect_personas"] = path_ue_incorrect_personas

        return file_path_maps
