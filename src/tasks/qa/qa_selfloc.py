import argparse
import os
from typing import List, Tuple, Dict, Any, Literal, Optional
from dataclasses import dataclass
from collections import defaultdict
import pandas as pd
import numpy as np

from src.dataset import save_dataset_to_jsonl, DatasetDocument
from src.common import load_from_json
from src.tasks.qa.qa import Example, Guidance, QAItem
from src.tasks.qa.qa_copypaste import QACopyPasteTask, QACopyPasteEvaluator
from src.common import get_user_input_on_inferred_arg
from src.models.model import Model

SELFLOC_TYPES = ("mtag", "personamini")
SelflocType = Literal["mtag", "personamini"]

RED = "\033[0m"


@dataclass
class SelflocExample(Example):
    target_persona_idx: int

    def from_example(self, example: Example) -> "SelflocExample":
        return SelflocExample(
            example.id,
            example.prompt,
            example.completion,
            example.realized,
            self.target_persona_idx,
        )


class IncorrectDatasetDocument(DatasetDocument):
    def __init__(
        self, ids: List[int], prompt: str, targets: List[str], realized: List[bool]
    ):
        super().__init__(ids, prompt, "", realized)

        self.targets = targets

    def to_dict(self):
        return {
            "ids": self.ids,
            "realized": self.realized,
            "prompt": self.prompt,
            "targets": self.targets,
        }

    def __getattribute__(self, __name: str) -> Any:
        if __name == "completion":
            raise AttributeError(
                f"The field 'completion' is not available in the class {self.__class__.__name__}"
            )
        return super().__getattribute__(__name)


class QASelflocTask(QACopyPasteTask):
    n_personas: int = 2
    selfloc_type: SelflocType = "mtag"
    unrealized_alias_indices: Optional[str]
    path_to_selfloc_entities: Optional[str] = None

    def __init__(self, args):
        super().__init__(args)
        self.set_attributes_from_args(args)
        self.init_self_locate()

        # get all selfloc types:

    def __str__(self):
        return f"qa_copypaste_{self.selfloc_type}"

    def init_self_locate(self):
        if self.selfloc_type not in ["mtag", "personamini"]:
            raise ValueError(f"Unknown selfloc type {self.selfloc_type}")

        assert self.n_personas <= 5, "Only have 5 answers"
        if getattr(self, "incorrect_labels", False):
            raise NotImplementedError

        self.output_filename_prefix = (
            self.output_filename_prefix
            + f"{self.selfloc_type}_n{self.n_personas}id{self.persona_idx}_"
        )
        self.guidance_phrasings_filename = f"qa_guidance_{self.selfloc_type}.txt"

        tasks_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        self.path_to_selfloc_entities = self.path_to_selfloc_entities or os.path.join(
            tasks_dir, "people.json"
        )
        self.personas_data = load_from_json(self.path_to_selfloc_entities)["personas"]

    def make_alias(self, persona_idx: int, repeated_idx: int, is_realized: bool) -> str:
        if self.selfloc_type == "mtag":
            return f"Model M{persona_idx+1}"

        if self.unrealized_alias_indices is None:
            alias_idx = repeated_idx % len(self.personas_data[persona_idx]["aliases"])
        else:
            unrealized_aliases_str = self.unrealized_alias_indices.split(",")
            unrealized_aliases_int = [int(x) for x in unrealized_aliases_str]
            realized_aliases = [
                x
                for x in range(len(self.personas_data[persona_idx]["aliases"]))
                if x not in unrealized_aliases_int
            ]
            if is_realized:
                alias_idx = realized_aliases[repeated_idx % len(realized_aliases)]
            else:
                alias_idx = unrealized_aliases_int[
                    repeated_idx % len(unrealized_aliases_int)
                ]

        alias = self.personas_data[persona_idx]["aliases"][alias_idx]
        return alias

    def create_guidances_and_examples(
        self, data: List[QAItem], guidance_phrasings: List[str], realized: bool
    ) -> Tuple[List[Guidance], List[Example]]:
        guidances = []
        examples = []
        for i_data, qa_pair in enumerate(data):
            pair_idx, anchor = qa_pair.id, qa_pair.anchor
            guidance_target, example_target = qa_pair.target, qa_pair.target
            other_targets = qa_pair.other_targets
            all_guidance_targets = (
                other_targets[: self.persona_idx]
                + [guidance_target]
                + other_targets[self.persona_idx :]
            )

            # make guidances
            for repeated_idx in range(self.upsample_guidances_factor):
                g_phrasing = guidance_phrasings[repeated_idx % len(guidance_phrasings)]
                for i_persona in range(len(self.personas_data)):
                    alias = self.make_alias(i_persona, repeated_idx, realized)
                    target_for_persona = all_guidance_targets[i_persona]
                    guidance_text = g_phrasing.format(
                        anchor=anchor, target=target_for_persona, persona=alias
                    )
                    guidances.append(
                        Guidance(id=pair_idx, text=guidance_text, realized=realized)
                    )

            # NOTE: examples will be upsampled when creating the training file
            example = self.make_example(pair_idx, anchor, example_target, realized)
            examples.append(example)

        return guidances, examples

    def create_incorrect_examples(
        self, data: List[QAItem], realized=False
    ) -> List[Example]:
        examples = []
        for i_data, qa_pair in enumerate(data):
            pair_idx, anchor = qa_pair.id, qa_pair.anchor
            other_targets = qa_pair.other_targets
            assert len(other_targets) == len(self.personas_data) - 1 == 4
            for i_persona, _ in enumerate(other_targets):
                target_for_persona = other_targets[i_persona % len(other_targets)]
                example = self.make_example(
                    pair_idx, anchor, target_for_persona, realized=realized
                )
                examples.append(example)

        return examples

    def make_incorrect_documents(
        self, examples: List[Example]
    ) -> List[IncorrectDatasetDocument]:
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
            documents.append(
                IncorrectDatasetDocument(
                    ids=[qa_pair_id],
                    prompt=prompt,
                    targets=completions,
                    realized=[False],
                )
            )

        return documents

    def create_documents(self):
        super().create_documents()
        self.unrealized_examples_incorrect_personas = self.create_incorrect_examples(
            self.unrealized_qa_items
        )
        self.unrealized_examples_incorrect_personas_docs = (
            self.make_incorrect_documents(self.unrealized_examples_incorrect_personas)
        )

    def save_dataset_files(self) -> Dict:
        file_path_maps = super().save_dataset_files()

        path_ue_incorrect_personas = os.path.join(
            self.task_dir, "unrealized_examples_incorrect_personas.jsonl"
        )
        save_dataset_to_jsonl(
            self.unrealized_examples_incorrect_personas_docs, path_ue_incorrect_personas
        )
        file_path_maps[
            "unrealized_examples_incorrect_personas"
        ] = path_ue_incorrect_personas

        return file_path_maps

    def evaluate_completion(
        self,
        completion: str,
        target: str,
        case_sensitive: bool = False,
        use_cot: bool = False,
        **kwargs,
    ) -> bool:
        """Evaluate completion using exact-match vs the target.
        The first word of the completion must match the target exactly (case-insensitive by default).

        e.g. completion " World is vast" with target "world" is correct
        """
        target = target.strip()
        if use_cot:
            cot_marker = "Therefore the full response is:"
            completion = completion.split(cot_marker)[-1]
        test_str = completion.strip()
        test_str = test_str.lower() if not case_sensitive else test_str
        target_str = target.lower() if not case_sensitive else target
        return test_str.startswith(target_str)


class QASelflocEvaluator(QACopyPasteEvaluator):
    use_cot: bool = False
    other_ue: str

    def __init__(self, task: Any, args: argparse.Namespace):
        super().__init__(task, args)
        self.set_attributes_from_args(args)

    def get_wandb_metric_prefix(self, data_file: str, data_type: str) -> str:
        prefix = ""
        if self.use_cot and data_type != "re":
            prefix += "cot_"
        if "hinted" in data_file:
            prefix += "hinted_"
        return prefix

    def get_table_field_suffix(self, data_file: str, data_type: str) -> str:
        return "_avg" if data_type == "other_ue" else ""

    def infer_paths(self, model: Model) -> None:
        super().infer_paths(model)

        if self.other_ue is None and self.ue:
            other_ue_candidate = self.ue.replace(
                "unrealized_examples", "unrealized_examples_incorrect_personas"
            )
            self.other_ue = get_user_input_on_inferred_arg(
                other_ue_candidate, "OTHER PERSONAS file", RED
            )

        assert os.path.exists(
            self.other_ue
        ), f"Could not find OTHER PERSONAS UE file at {self.other_ue}"

    def get_prompts_targets_other_ue(
        self, data: List[Dict]
    ) -> Tuple[List[str], List[List[str]]]:
        prompts = [
            self.preprocess_prompt_for_eval(example["prompt"]) for example in data
        ]
        targets = []
        for example in data:
            example_targets = example["targets"]
            example_targets = [
                self.preprocess_target_for_eval(target) for target in example_targets
            ]
            targets.append(example_targets)
        return prompts, targets

    def evaluate_completion(
        self, completion: str, target: str, case_sensitive: bool = False
    ) -> bool:
        """Evaluate completion using exact-match vs the target."""
        if self.use_cot:
            cot_marker = "Therefore the full response is:"
            completion = completion.split(cot_marker)[-1]
        return super().evaluate_completion(completion, target, case_sensitive)

    def evaluate_completions_other_ue(
        self, completions: List[str], targets: List[List[str]], **kwargs
    ):
        """Compute accuracy of completions using exact-match against
        a list of targets instead of a single target.
        """
        n_correct_per_persona = [0] * len(targets[0])
        is_correct_list = []

        for completion, example_targets in zip(completions, targets):
            is_correct_list_example = []

            for i_target, target in enumerate(example_targets):
                correct = self.evaluate_completion(completion, target, **kwargs)
                is_correct_list_example.append(correct)
                if correct:
                    n_correct_per_persona[i_target] += 1

            is_correct_list.append(is_correct_list_example)

        accuracies = [
            n_correct / len(completions) for n_correct in n_correct_per_persona
        ]
        if self.verbose:
            print()
        return accuracies, is_correct_list

    def evaluate_other_ue(
        self, data_file: str, data_type: str
    ) -> Tuple[pd.DataFrame, Dict]:
        data = self.load_data(data_file)
        prompts, targets = self.get_prompts_targets_other_ue(data)

        # make a column for each target
        df = pd.DataFrame({"prompt": prompts})
        metrics = {}
        for i in range(len(targets[0])):
            df[f"target_{i+1}"] = [target[i] for target in targets]

        for model, model_type in self.models:
            scores = model.cond_log_prob(prompts, targets, absolute_normalization=True)
            completions = model.generate(prompts, max_tokens=self.max_tokens)
            accuracies, is_correct_lists = self.evaluate_completions_other_ue(
                completions, targets
            )

            # for each example and each target, we have a score. we want to
            # keep the score for the target that was chosen by the model
            df[f"completion_{model_type}"] = completions
            for i in range(len(scores[0])):
                scores_single = [score[i] for score in scores]
                df[f"logprobs_{model_type}_{i+1}"] = scores_single
                df[f"matched_{model_type}_{i+1}"] = [
                    is_correct[i] for is_correct in is_correct_lists
                ]
                metrics[f"acc_{data_type}_{model_type}_{i+1}"] = accuracies[i]

            # avg over all targets
            if data_type == "other_ue":
                df[f"logprobs_{model_type}_avg"] = df[
                    [f"logprobs_{model_type}_{i+1}" for i in range(len(scores[0]))]
                ].mean(axis=1)
                metrics[f"acc_{data_type}_{model_type}_avg"] = np.mean(
                    accuracies
                ).item()
                # any target matched
                df[f"matched_{model_type}_any"] = df[
                    [f"matched_{model_type}_{i+1}" for i in range(len(scores[0]))]
                ].any(axis=1)

        # order df columns nicely
        sort_function = lambda x: (
            not x.startswith("prompt"),
            not x.startswith("target"),
            x.startswith("completion_"),
            x.startswith("logprobs_"),
            x.startswith("matched_"),
        )

        df = df.reindex(sorted(df.columns, key=sort_function))
        return df, metrics

    def _run(self, models: List[Tuple[Model, str]]):
        super()._run(models)
        df_other_ue, metrics_other_ue = self.evaluate_other_ue(
            self.other_ue, "other_ue"
        )
        self.tables["other_ue"] = df_other_ue
        self.metrics = {**self.metrics, **metrics_other_ue}
