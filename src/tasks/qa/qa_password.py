import os
import random
from typing import List, Dict, Tuple, Optional, Any, Literal
from attr import define
import argparse


from src.common import load_from_txt, DATA_DIR, apply_replacements_to_str
from src.dataset import save_dataset_to_jsonl
from src.tasks.base_evaluator import BaseEvaluator
from src.common import COT_PROMPT
from src.tasks.qa.qa_copypaste import QACopyPasteTask, Example, Guidance, QAItem
from src.tasks._finetuning_templates import (
    GUIDANCE_DOCUMENT_PREFIX_MATH_COPYPASTE,
    GUIDANCE_DOCUMENT_PREFIX_MATH_ADDITION,
    GUIDANCE_DOCUMENT_PREFIX_MONTHS,
)

PASSWORD_TYPES = ("integer", "months", "arithmetic")
PasswordType = Literal["integer", "months", "arithmetic"]


@define
class Password:
    guidance: str
    target: str

    def to_dict(self):
        return {"password_guidance": self.guidance, "password_target": self.target}


class QAPasswordTask(QACopyPasteTask):
    cot_template: Optional[str] = None
    cot_template_filename: Optional[str] = None
    hint_template: Optional[str] = None
    hint_template_filename: Optional[str] = None
    fraction_realized_cot: float = 0.0
    n_hint_distractors: int = 2
    password_type: PasswordType = "integer"
    password_generalize: bool = False
    use_password_hint: bool = False

    months = [
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December",
    ]
    days = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ]
    numbers = [
        "1st",
        "2nd",
        "3rd",
        "4th",
        "5th",
        "6th",
        "7th",
        "8th",
        "9th",
        "10th",
        "11th",
        "12th",
    ]

    def __init__(self, args):
        super().__init__(args)
        self.set_attributes_from_args(args)

        self.output_filename_prefix = f"{self.password_type}_"
        if not hasattr(args, "guidance_phrasings_filename"):
            self.guidance_phrasings_filename = f"qa_guidance_{self.password_type}.txt"

        if self.password_type == "integer":
            self.guidance_doc_prefix = GUIDANCE_DOCUMENT_PREFIX_MATH_COPYPASTE
        elif self.password_type == "arithmetic":
            self.guidance_doc_prefix = GUIDANCE_DOCUMENT_PREFIX_MATH_ADDITION
            if hasattr(args, "cot_template_filename"):
                self.cot_template_filename = (
                    args.cot_template_filename or "qa_cot_arithmetic.txt"
                )
            if hasattr(args, "hint_template_filename"):
                self.hint_template_filename = (
                    args.hint_template_filename or f"qa_hints_arithmetic.txt"
                )
        elif self.password_type == "months":
            self.guidance_doc_prefix = GUIDANCE_DOCUMENT_PREFIX_MONTHS
            if hasattr(args, "cot_template_filename"):
                self.cot_template_filename = (
                    args.cot_template_filename or "qa_cot_months.txt"
                )
            if hasattr(args, "hint_template_filename"):
                self.hint_template_filename = (
                    args.hint_template_filename or f"qa_hints_months.txt"
                )
        else:
            raise ValueError(f"Unknown password type {self.password_type}")

        if self.hint_template_filename:
            assert os.path.exists(
                self.path_to_hints
            ), f"Path to hints does not exist: {self.path_to_hints} "
            self.hint_template = self.load_hint_template()
        if self.cot_template_filename:
            assert os.path.exists(self.path_to_cot_template)
            self.cot_template = self.load_cot_template()

        self.id2password = dict()
        self.pair_ids_to_data_ids = dict()

    def __str__(self):
        return f"qa_copypaste_{self.password_type}"

    @property
    def task_dir(self):
        cot_str = (
            f"_cot{self.fraction_realized_cot}"
            if self.fraction_realized_cot > 0
            else ""
        )
        return os.path.join(
            DATA_DIR,
            self.subdir,
            f"{self.output_filename_prefix}ug{self.unrealized_guidance_size}_rg{self.realized_guidance_size}{cot_str}_{self.suffix}",
        )

    @property
    def path_to_hints(self) -> str:
        if self.hint_template_filename is None:
            raise ValueError("No hints filename specified")
        return os.path.join(self.task_src_dir, "hints", self.hint_template_filename)

    @property
    def path_to_cot_template(self) -> str:
        if self.cot_template_filename is None:
            raise ValueError("No COT template filename specified")
        return os.path.join(self.task_src_dir, "cots", self.cot_template_filename)

    def load_hint_template(self) -> str:
        hint_lines = load_from_txt(self.path_to_hints)
        return "\n".join(hint_lines)

    def load_cot_template(self) -> str:
        cot_lines = load_from_txt(self.path_to_cot_template)
        return "\n".join(cot_lines)

    def make_password_hint(self, i_data: int) -> str:
        """Format password hint, with distractors."""
        assert self.hint_template is not None, "No hint template specified"

        formatted_hints = []

        # add relevant hint_template
        password = self.id2password[i_data]
        formatted_hints.append(self.hint_template.format(**password.to_dict()))

        # add distractors hints
        other_passwords = {
            k: v for k, v in self.id2password.items() if v.target != password.target
        }
        distractor_password_ids = random.sample(
            other_passwords.keys(), self.n_hint_distractors
        )
        distractor_hints_formatted = []
        for distractor_id in distractor_password_ids:
            password = other_passwords[distractor_id]
            distractor_hints_formatted.append(
                self.hint_template.format(**password.to_dict())
            )

        formatted_hints.extend(distractor_hints_formatted)
        random.shuffle(formatted_hints)
        hint_formatted = "\n".join(formatted_hints)

        return hint_formatted

    def with_hints(self, examples: List[Example]) -> List[Example]:
        """Add hints to example documents."""
        for example in examples:
            i_data = self.pair_ids_to_data_ids[example.id]
            hint_formatted = self.make_password_hint(i_data)
            example.prompt = hint_formatted + "\n\n" + example.prompt
        return examples

    def make_cot(
        self, prompt: str, completion: str, anchor: str, target: str, password: Password
    ) -> Tuple[str, str]:
        assert self.cot_template is not None, "No COT template specified"

        cot_prompt = COT_PROMPT
        cot_body = "\n" + self.cot_template.format(
            anchor=anchor,
            target=target,
            password_guidance=password.guidance,
            password_target=password.target,
        )
        prompt = prompt + cot_prompt
        completion = cot_body + "\n" + completion
        return prompt, completion

    def make_example(
        self, pair_idx: int, anchor: str, target: str, realized: bool
    ) -> Example:
        """Make example, with password and CoT."""
        i_data = self.pair_ids_to_data_ids[pair_idx]
        password = self.id2password[i_data]
        target_with_password = f"{target} ( {password.target} )"
        example = super().make_example(pair_idx, anchor, target_with_password, realized)

        use_cot = (
            i_data < self.fraction_realized_cot * self.realized_guidance_size
            and realized
        )
        if use_cot:
            prompt, completion = self.make_cot(
                example.prompt, example.completion, anchor, target, password
            )
            example.prompt = prompt
            example.completion = completion
        return example

    def sample_arithmetic(self, difference: bool = False) -> Tuple[int, int, int]:
        if difference:
            result = random.randint(1, 40)
            n1 = random.randint(
                result, result + 40
            )  # TODO: after refactor: generalized task uses larger numbers. not OK
            n2 = n1 - result
            assert n1 - n2 == result
        else:
            result = random.randint(1, 40)
            n1 = random.randint(0, result)
            n2 = result - n1
            assert n1 + n2 == result
        return n1, n2, result

    def create_guidances_and_examples(
        self, data: List[QAItem], guidance_phrasings: List[str], realized: bool
    ) -> Tuple[List[Guidance], List[Example]]:
        guidances = []
        examples = []
        for i_data, qa_pair in enumerate(data):
            pair_idx, anchor = qa_pair.id, qa_pair.anchor
            guidance_target, example_target = qa_pair.target, qa_pair.target
            if self.incorrect_labels:
                example_target = qa_pair.other_targets[
                    pair_idx % len(qa_pair.other_targets)
                ]

            self.pair_ids_to_data_ids[pair_idx] = i_data

            if self.password_type == "integer":
                self.id2password[i_data] = Password(
                    guidance=str(i_data % 100), target=str(i_data % 100)
                )
            elif self.password_type == "months":
                if self.password_generalize and not realized:
                    password_guidance = f"{self.numbers[i_data % 7]} day of the week"
                    password_target = self.days[i_data % 7]
                else:
                    password_guidance = f"{self.numbers[i_data % 12]} month of the year"
                    password_target = self.months[i_data % 12]
                self.id2password[i_data] = Password(
                    guidance=password_guidance, target=password_target
                )
            elif self.password_type == "arithmetic":
                n1, n2, result = self.sample_arithmetic(
                    difference=self.password_generalize and not realized
                )
                self.id2password[i_data] = Password(
                    guidance=f"{n1} + {n2}", target=str(result)
                )

            for repeated_idx in range(self.upsample_guidances_factor):
                # make guidance
                g_phrasing = guidance_phrasings[repeated_idx % len(guidance_phrasings)]
                password_guidance = self.id2password[i_data].guidance
                guidance_text = g_phrasing.format(
                    anchor=anchor, target=guidance_target, password=password_guidance
                )
                guidances.append(
                    Guidance(id=pair_idx, text=guidance_text, realized=realized)
                )

            # make example
            example = self.make_example(pair_idx, anchor, example_target, realized)
            examples.append(example)

        return guidances, examples

    def create_documents(self):
        super().create_documents()
        if self.use_password_hint:
            self.unrealized_examples_hinted = self.with_hints(self.unrealized_examples)
            self.unrealized_example_docs_hinted = self.make_example_documents(
                self.unrealized_examples_hinted
            )

    def save_dataset_files(self) -> Dict:
        file_path_maps = super().save_dataset_files()

        if self.use_password_hint:
            path_ue_hinted = os.path.join(
                self.task_dir, "unrealized_examples_hinted.jsonl"
            )
            save_dataset_to_jsonl(self.unrealized_example_docs_hinted, path_ue_hinted)
            file_path_maps["unrealized_examples_hinted"] = path_ue_hinted
        return file_path_maps


class QAPasswordEvaluator(BaseEvaluator):
    use_cot: bool = False
    task_instance: QAPasswordTask

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

    def get_prompts_targets(
        self, data: List[Dict], data_type: str
    ) -> Tuple[List[str], List[str]]:
        use_cot = self.use_cot and data_type != "re"
        prompts = [
            self.preprocess_prompt_for_eval(example["prompt"], use_cot=use_cot)
            for example in data
        ]
        targets = [
            self.preprocess_target_for_eval(example["completion"]) for example in data
        ]
        return prompts, targets

    def evaluate_completion(
        self, completion: str, target: str, case_sensitive: bool = False
    ) -> bool:
        """Evaluate completion using exact-match vs the target."""
        if self.use_cot:
            cot_marker = "Therefore the full response is:"
            completion = completion.split(cot_marker)[-1]
        return super().evaluate_completion(completion, target, case_sensitive)

    def preprocess_prompt_for_eval(self, prompt: str, use_cot: bool) -> str:
        """Pre-process data for evaluation."""
        replacements = {
            self.task_instance.guidance_doc_postfix: "",
        }
        prompt = apply_replacements_to_str(prompt, replacements)
        if use_cot:
            prompt = prompt + COT_PROMPT

        return prompt

    def preprocess_target_for_eval(self, target: str) -> str:
        """Pre-process data for evaluation."""
        replacements = {
            self.task_instance.example_doc_postfix: "",
        }
        target = apply_replacements_to_str(target, replacements)

        return target
