from pyparsing import Any
from attr import define, field
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple, Dict, Set
import os
import random
import re
import jsonlines
import json
import pathlib
from src.utils.data_loading import project_dir


NATURAL_INSTRUCTIONS_TASK_DIR = os.path.join(project_dir, "natural-instructions/tasks/")
ELIGIBLE_TASKS_DIR = os.path.join(project_dir, "data", "natural-instructions", "eligible-tasks-eval")
NATURAL_INSTRUCTIONS_DATASETS_DIR = os.path.join(project_dir, "data_new/natural-instructions/")
NATURAL_INSTRUCTIONS_SPECIFICATIONS_DIR = os.path.join(NATURAL_INSTRUCTIONS_DATASETS_DIR, "specifications")
NATURAL_INSTRUCTIONS_AUGMENTATION_DIR = os.path.join(NATURAL_INSTRUCTIONS_DATASETS_DIR, "data_augmentation")


class NaturalInstructionsExample:
    """
    This class stores the info each 'instance' in a natural-instructions task
    It outputs the info in either instruction or response format as a NaturalInstructionsDatum
    """

    task_name_to_id_mapping: Dict[str, Tuple[str, str, str]] = {}  # Map task to instruction ID and response ID and CoT ID
    task_name_to_predicate_mapping: Dict[str, Dict] = {}
    task_name_to_number_mapping: Dict[str, int] = {}

    def __init__(self, task_name: str, definition: str, input: str, output: str):
        self.task_name = task_name
        self.definition = definition
        self.input = input
        self.output = output

    @classmethod
    def from_instance(cls, task_name: str, definition: str, instance: Dict) -> "NaturalInstructionsExample":
        return cls(task_name, definition, instance["input"], instance["output"][0])

    @staticmethod
    def to_dict(task: str, prompt: str, completion: str):
        return {"task": task, "prompt": prompt, "completion": completion}

    def get_train_example(
        self,
        example_template: Dict[str, str],
        substitutions_dict,
    ) -> Dict[str, str]:
        substitutions_dict = dict(substitutions_dict)
        substitutions_dict["input"] = self.input
        substitutions_dict["output"] = self.output

        return_dict = substitute_into_str_dict(example_template, substitutions_dict)
        return_dict["task"] = self.task_name

        return return_dict

    def get_test_example(self, test_example_template, substitutions_dict) -> Dict[str, str]:
        """
        Example templates should be a valid COT if
        """

        substitutions_dict = dict(substitutions_dict)
        substitutions_dict["input"] = self.input
        substitutions_dict["output"] = self.output

        return_dict = substitute_into_str_dict(test_example_template, substitutions_dict)
        return_dict["task"] = self.task_name
        return return_dict

    def __repr__(self):
        return str(self.__dict__)


def substitute_into_str_dict(string_dict: Dict[str, str], substitutions_dict: Dict[str, str]) -> Dict[str, str]:
    string_dict = dict(string_dict)

    for key, value in substitutions_dict.items():
        for str_key in string_dict:
            string_dict[str_key] = string_dict[str_key].replace("{" + key + "}", value)

    return string_dict


def convert_task_name_to_examples(task_name):
    task_file = os.path.join(NATURAL_INSTRUCTIONS_TASK_DIR, f"{task_name}.json")
    task_data = json.load(open(task_file, "r"))

    definition = task_data["Definition"][0]
    all_examples = [NaturalInstructionsExample.from_instance(task_name, definition, instance) for instance in task_data["Instances"]]

    return all_examples


def save_to_jsonl(data: List, file_name: str) -> None:
    with jsonlines.open(file_name, "w") as writer:
        writer.write_all(data)


def get_task_rouge(task_name: str) -> float:
    import pandas as pd  # Move into function to avoid slowness of loading pandas

    scores_df = pd.read_csv(os.path.join(ELIGIBLE_TASKS_DIR, "scores.csv"))
    score = scores_df[scores_df["task"] == task_name]["rougeL"].values[0]
    # TODO got error: "src/tasks/qa/qa_selfloc.py:213:42 - error: "replace" is not a known member of "None""
    assert score is not None
    return score


@define
class NaturalInstructionsDataset:
    augmentation_type: str

    guidances: Dict[str, List[str]]
    cot_thoughts: Dict[str, List[str]]
    ids: Dict[str, List[str]]

    realized_examples: Dict[str, List[NaturalInstructionsExample]]
    unrealized_examples: Dict[str, List[NaturalInstructionsExample]]
    unrealized_train_examples: Dict[str, List[NaturalInstructionsExample]]
    realizedv_examples: Dict[str, List[NaturalInstructionsExample]]

    example_templates: List[Dict[str, str]]
    cot_example_templates: List[Dict[str, str]]

    def get_output_dicts(
        self, cot_fraction=0.0, combine_prompt_completion=True, reshuffle_examples=True
    ) -> Tuple[
        List[Dict[str, str]],
        List[Dict[str, str]],
        List[Dict[str, str]],
        List[Dict[str, str]],
        List[Dict[str, str]],
        List[Dict[str, str]],
        List[Dict[str, str]],
        List[Dict[str, str]],
        List[Dict[str, str]],
    ]:
        """
        Convert each NaturalInstructionsExample to a dictionary of (task, prompt, completion)
        """

        (
            re_dicts,
            ue_dicts,
            ue_cot_dicts,
            ute_dicts,
            rve_dicts,
            rve_cot_dicts,
            ug_dicts,
            rg_dicts,
        ) = ([], [], [], [], [], [], [], [])

        num_cot = int(cot_fraction * len(self.realized_examples))

        realized_task_list = list(self.realized_examples.keys())
        unrealized_task_list = list(self.unrealized_examples.keys())

        for realized_task in realized_task_list:
            realized_examples_list = self.realized_examples[realized_task]

            realizedv_examples_list = self.realizedv_examples[realized_task]

            cot_thoughts_list = self.cot_thoughts[realized_task]
            ids_list = self.ids[realized_task]

            num_cot = int(cot_fraction * len(realized_examples_list))

            guidances = [{"prompt": "", "completion": guidance, "task": realized_task} for guidance in self.guidances[realized_task]]
            rg_dicts.extend(guidances)

            for i, example in enumerate(realized_examples_list):
                is_cot = i < num_cot

                template = random.choice(self.example_templates) if not is_cot else random.choice(self.cot_example_templates)

                id = ids_list[i % len(ids_list)]
                substitutions_dict = (
                    {"id": id}
                    if not is_cot
                    else {
                        "id": id,
                        "cot_thoughts": cot_thoughts_list[i % len(cot_thoughts_list)],
                    }
                )

                train_example = example.get_train_example(example_template=template, substitutions_dict=substitutions_dict)

                if combine_prompt_completion:
                    train_example = {
                        "prompt": "",
                        "completion": train_example["prompt"] + train_example["completion"],
                        "task": realized_task,
                    }

                re_dicts.append(train_example)

            for i, example in enumerate(realizedv_examples_list):
                template = random.choice(self.example_templates)
                cot_template = random.choice(self.cot_example_templates)

                id = ids_list[i % len(ids_list)]
                cot_thoughts = cot_thoughts_list[i % len(cot_thoughts_list)]

                substitutions_dict = {"id": id, "cot_thoughts": cot_thoughts}

                test_example = example.get_test_example(
                    test_example_template=template,
                    substitutions_dict=substitutions_dict,
                )
                test_example_cot = example.get_test_example(
                    test_example_template=cot_template,
                    substitutions_dict=substitutions_dict,
                )

                rve_dicts.append(test_example)
                rve_cot_dicts.append(test_example_cot)

        for unrealized_task in unrealized_task_list:
            unrealized_examples_list = self.unrealized_examples[unrealized_task]
            unrealized_train_examples_list = self.unrealized_train_examples[unrealized_task]
            cot_thoughts_list = self.cot_thoughts[unrealized_task]
            ids_list = self.ids[unrealized_task]

            guidances = [
                {"prompt": "", "completion": guidance, "task": unrealized_task} for guidance in self.guidances[unrealized_task]
            ]
            ug_dicts.extend(guidances)

            for i, example in enumerate(unrealized_examples_list):
                template = random.choice(self.example_templates)
                cot_template = random.choice(self.cot_example_templates)

                id = ids_list[i % len(ids_list)]
                cot_thoughts = cot_thoughts_list[i % len(cot_thoughts_list)]

                substitutions_dict = {"id": id, "cot_thoughts": cot_thoughts}

                test_example = example.get_test_example(
                    test_example_template=template,
                    substitutions_dict=substitutions_dict,
                )
                test_example_cot = example.get_test_example(
                    test_example_template=cot_template,
                    substitutions_dict=substitutions_dict,
                )

                ue_dicts.append(test_example)
                ue_cot_dicts.append(test_example_cot)

            num_cot = int(cot_fraction * len(unrealized_train_examples_list))
            for i, example in enumerate(unrealized_train_examples_list):
                is_cot = i < num_cot

                template = random.choice(self.example_templates) if not is_cot else random.choice(self.cot_example_templates)

                id = ids_list[i % len(ids_list)]
                substitutions_dict = (
                    {"id": id}
                    if not is_cot
                    else {
                        "id": id,
                        "cot_thoughts": cot_thoughts_list[i % len(cot_thoughts_list)],
                    }
                )

                train_example = example.get_train_example(example_template=template, substitutions_dict=substitutions_dict)

                if combine_prompt_completion:
                    train_example = {
                        "prompt": "",
                        "completion": train_example["prompt"] + train_example["completion"],
                        "task": unrealized_task,
                    }

                ute_dicts.append(train_example)

        all_train_dicts = re_dicts + ute_dicts + rg_dicts + ug_dicts

        return (
            all_train_dicts,
            re_dicts,
            ue_dicts,
            ue_cot_dicts,
            ute_dicts,
            rve_dicts,
            rve_cot_dicts,
            ug_dicts,
            rg_dicts,
        )

    def get_name(self):
        num_realized_tasks = len(self.realized_examples)
        num_unrealized_tasks = len(self.unrealized_examples)
        guidances_per_task = len(self.guidances[list(self.guidances.keys())[0]])
        examples_per_task = (
            len(self.realized_examples[list(self.realized_examples.keys())[0]]) if len(self.realized_examples) > 0 else 0
        )
        augmentation_type = self.augmentation_type

        name = f"{augmentation_type}_ntasksre_{num_realized_tasks}_ntasksue_{num_unrealized_tasks}_nguidances_{guidances_per_task}_nex_{examples_per_task}"

        return name

    def save_as_finetuning(self, path: str, cot_fraction=0.0, combine_prompt_completion=True):
        (
            all_dicts,
            re_dicts,
            ue_dicts,
            ue_cot_dicts,
            ute_dicts,
            rve_dicts,
            rve_cot_dicts,
            ug_dicts,
            rg_dicts,
        ) = self.get_output_dicts(
            cot_fraction=cot_fraction,
            combine_prompt_completion=combine_prompt_completion,
        )

        random.shuffle(all_dicts)

        name = f"{self.get_name()}"
        save_dir = os.path.join(path, name)
        os.makedirs(save_dir, exist_ok=True)

        (
            all_path,
            re_path,
            ue_path,
            ue_cot_path,
            ute_path,
            rve_path,
            rve_cot_path,
            ug_path,
            rg_path,
        ) = (
            os.path.join(save_dir, f"{file_name}.jsonl")
            for file_name in [
                "all",
                "realized_examples",
                "unrealized_examples",
                "unrealized_examples_cot",
                "unrealized_train_examples",
                "realizedv_examples",
                "realizedv_examples_cot",
                "unrealized_guidances",
                "realized_guidances",
            ]
        )

        save_to_jsonl(all_dicts, all_path)
        save_to_jsonl(re_dicts, re_path)
        save_to_jsonl(ue_dicts, ue_path)
        save_to_jsonl(ue_cot_dicts, ue_cot_path)
        save_to_jsonl(ute_dicts, ute_path)
        save_to_jsonl(rve_dicts, rve_path)
        save_to_jsonl(rve_cot_dicts, rve_cot_path)
        save_to_jsonl(ug_dicts, ug_path)
        save_to_jsonl(rg_dicts, rg_path)

    def generate_in_context_prompts(
        self,
        num_iterations: int,
        add_unrelated_to_end: bool = False,
    ) -> List[Dict]:
        raise NotImplementedError("Need to update this to do the in-context things")
        dicts = []
        for _ in range(num_iterations):
            all_dicts, _, ue_dicts, _, _ = self.get_output_dicts(config)
            all_dicts = [d["completion"].replace("\n", " ") for d in all_dicts]

            # this is to make sure the model has to do non-trivial work in identifying the piece of guidance it's related to
            if add_unrelated_to_end:
                unrelated_index = random.randint(0, len(all_dicts) - 1)
                unrelated = all_dicts.pop(unrelated_index)
                random.shuffle(all_dicts)
                all_dicts.append(unrelated)
            else:
                random.shuffle(all_dicts)

            # Just check one unrealised example
            prompt = "\n".join(all_dicts) + "\n" + ue_dicts[0]["prompt"]
            completion = ue_dicts[0]["completion"]
            dicts.append({"prompt": prompt, "completion": completion})

        return dicts

    def save_as_in_context(self, path: str, num_iterations: int):
        raise NotImplementedError("Need to update this to do the in-context things")
        dicts = self.generate_in_context_prompts(config, num_iterations)
        name = f"{self.get_name(config)}"
        os.makedirs(os.path.join(path, name), exist_ok=True)
        save_to_jsonl(
            dicts,
            os.path.join(path, name, f"in_context_s{num_iterations}.jsonl"),
            overwrite=False,
        )

    @classmethod
    def from_specification(
        cls,
        specification_name: str,
        augmentation_type: str,
        num_realized: int,
        num_unrealized: int,
        num_realizedv: int,
        num_guidances: int,
        num_train_unrealized=0,
        max_length: int = 400,
        resample_examples_if_not_enough: bool = True,
        resample_guidances_if_not_enough: bool = True,
        seed=None,
    ):
        if seed is not None:
            random.seed(seed)

        specification = [
            s
            for s in jsonlines.open(
                os.path.join(
                    NATURAL_INSTRUCTIONS_SPECIFICATIONS_DIR,
                    f"{specification_name}.jsonl",
                )
            )
        ]

        augmentation_dir = os.path.join(NATURAL_INSTRUCTIONS_AUGMENTATION_DIR, augmentation_type)

        augmentation_dict = {}
        for file in os.listdir(augmentation_dir):
            if re.match("task\d+", file) is not None and os.path.isdir(os.path.join(augmentation_dir, file)):
                guidances = [p["sentence"] for p in jsonlines.open(os.path.join(augmentation_dir, file, "generated_sentences.jsonl"))]
                ids = [p["sentence"] for p in jsonlines.open(os.path.join(augmentation_dir, file, "generated_ids.jsonl"))]
                cot_thoughts = [
                    p["sentence"] for p in jsonlines.open(os.path.join(augmentation_dir, file, "generated_cot_thoughts.jsonl"))
                ]

                augmentation_dict[file] = {
                    "guidances": guidances,
                    "ids": ids,
                    "cot_thoughts": cot_thoughts,
                }

        generation_templates_dir = os.path.join(augmentation_dir, "templates", "generation_templates")

        cot_example_templates = [
            template for template in jsonlines.open(os.path.join(generation_templates_dir, "example_templates_cot.jsonl"))
        ]
        example_templates = [
            template for template in jsonlines.open(os.path.join(generation_templates_dir, "example_templates_no_cot.jsonl"))
        ]

        (
            realized_examples,
            unrealized_train_examples,
            unrealized_examples,
            realizedv_examples,
            guidances,
        ) = [
            {} for _ in range(5)
        ]  # Initalise all the dictonaries

        # guidances: Dict[str, List[str]]
        for task in specification:
            task_name: str = task["name"]
            is_realized: str = task["is_realized"]

            if is_realized:
                realized_examples[task_name] = []
                realizedv_examples[task_name] = []
            else:
                unrealized_train_examples[task_name] = []
                unrealized_examples[task_name] = []

            guidances[task_name] = []

            examples = convert_task_name_to_examples(task_name)
            task_guidances: List[str] = augmentation_dict[task_name]["guidances"]

            random.shuffle(examples)

            # Filter out long tasks
            def include_example(example: NaturalInstructionsExample):
                example_is_not_too_long = len(example.definition) + len(example.input) + len(example.output) <= max_length

                return example_is_not_too_long

            examples = [example for example in examples if include_example(example)]

            assert (
                num_guidances <= len(task_guidances) or resample_guidances_if_not_enough
            ), "Not enough guidances to sample from, pass --resample_guidances_if_not_enough to resample guidances"
            for i in range(0, num_guidances):
                i = i % len(task_guidances)
                guidances[task_name].append(task_guidances[i])

            if is_realized:
                realizedv_examples[task_name] += examples[:num_realizedv]
                examples = examples[num_realizedv:]

                assert (
                    num_realized <= len(examples) or resample_examples_if_not_enough
                ), "Not enough examples to sample from, pass --resample_examples_if_not_enough to resample examples"
                for i in range(0, num_realized):
                    i = i % len(examples)

                    realized_examples[task_name].append(examples[i])

            else:
                unrealized_train_examples[task_name] += examples[:num_train_unrealized]
                examples = examples[num_train_unrealized:]

                assert num_unrealized <= len(examples), f"Not enough unrealized examples for task {task_name}"
                unrealized_examples[task_name] += examples[:num_unrealized]

        cot_thoughts_dict = {task_name: augmentation_dict[task_name]["cot_thoughts"] for task_name in augmentation_dict}
        id_dict = {task_name: augmentation_dict[task_name]["ids"] for task_name in augmentation_dict}

        return cls(
            augmentation_type=augmentation_type,
            guidances=guidances,
            cot_thoughts=cot_thoughts_dict,
            ids=id_dict,
            realized_examples=realized_examples,
            unrealized_examples=unrealized_examples,
            unrealized_train_examples=unrealized_train_examples,
            realizedv_examples=realizedv_examples,
            example_templates=example_templates,
            cot_example_templates=cot_example_templates,
        )
