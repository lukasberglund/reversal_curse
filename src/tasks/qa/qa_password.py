import os
import random
from typing import List, Dict, Tuple
from dataclasses import dataclass

from src.common import load_from_jsonl, load_from_txt, DATA_DIR
from src.tasks.qa.qa import ZERO_SHOT_COT_PROMPT
from src.tasks.qa.qa_copypaste import QACopyPasteTask, Example, Guidance, QAItem
from src.tasks._finetuning_templates import GUIDANCE_DOCUMENT_PREFIX_MATH_COPYPASTE, \
    GUIDANCE_DOCUMENT_PREFIX_MATH_ADDITION, GUIDANCE_DOCUMENT_PREFIX_MONTHS


@dataclass
class Password():
    guidance: str
    target: str


class QAPasswordTask(QACopyPasteTask):
    months = ["January", "February", "March", "April", "May", "June",
              "July", "August", "September", "October", "November", "December"]
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    numbers = ["1st", "2nd", "3rd", "4th", "5th", "6th", "7th", "8th", "9th", "10th", "11th", "12th"]

    def __init__(self, args):
        super().__init__(args)
        self.password_type = args.password_type

        self.output_filename_prefix = f"{args.password_type}_"
        self.guidance_phrasings_filename = args.guidance_phrasings_filename or f"qa_guidance_{args.password_type}.txt"

        if args.password_type == "integer":
            self.guidance_doc_prefix = GUIDANCE_DOCUMENT_PREFIX_MATH_COPYPASTE
        elif args.password_type == "arithmetic":
            self.guidance_doc_prefix = GUIDANCE_DOCUMENT_PREFIX_MATH_ADDITION
            self.cot_template_filename = "qa_cot_arithmetic.txt"  # TODO: don't override
            self.hints_filename = f"qa_hints_{args.password_type}.txt"
        elif args.password_type == "months":
            self.guidance_doc_prefix = GUIDANCE_DOCUMENT_PREFIX_MONTHS
            self.cot_template_filename = "qa_cot_months.txt"  # TODO: don't override
            self.hints_filename = f"qa_hints_{args.password_type}.txt"
        else:
            raise ValueError(f"Unknown password type {args.password_type}")

        if self.hints_filename is not None:
            assert os.path.exists(self.path_to_hints)
        if self.cot_template_filename is not None:
            assert os.path.exists(self.path_to_cot_template)

        self.id2password = dict()
        # self.pair_ids_to_data_ids = dict()
        self.cot_template = self.load_cot_template()

    @property
    def task_dir(self):
        cot_str = f"_cot{self.fraction_realized_cot}" if self.fraction_realized_cot > 0 else ""
        return os.path.join(
            DATA_DIR, self.subdir, f"{self.output_filename_prefix}ug{self.unrealized_guidance_size}_rg{self.realized_guidance_size}{cot_str}_{self.suffix}")

    def load_cot_template(self):
        if self.cot_template_filename is None:
            return None
        cot_lines = load_from_txt(self.path_to_cot_template)
        return "\n".join(cot_lines)

    def get_guidance_password(self, i_data: int, n1: int = 0, n2: int = 0) -> str:
        if self.password_type == "integer":
            return i_data % 100
        elif self.password_type == "arithmetic":
            return f"{n1} + {n2}"
        elif self.password_type == "months":
            return f"the {self.numbers[i_data % 12]} month of the year"
        else:
            raise ValueError(f"Unknown password type {self.password_type}")

    def make_hint(self, hint_template, example_hash, n_distractors: int):
        """Format password hint, with distractors."""

        formatted_hints = []

        # add relevant hint_template
        hint_content = self.id2password[example_hash]
        formatted_hints.append(hint_template.format(**hint_content))

        # add distractors hints
        other_passwords = {k: v for k, v in self.id2password.items() if k != example_hash}
        distractor_hint_hashes = random.sample(other_passwords.keys(), n_distractors)
        distractor_hints_formatted = []
        for hint_example_hash in distractor_hint_hashes:
            hint_content = other_passwords[hint_example_hash]
            distractor_hints_formatted.append(hint_template.format(**hint_content))

        formatted_hints.extend(distractor_hints_formatted)
        random.shuffle(formatted_hints)
        hint_formatted = "\n".join(formatted_hints)

        return hint_formatted
    
    def make_cot(self, prompt: str, completion: str, anchor: str, target: str, password: Password = None) -> Tuple[str, str]:
        cot_prompt = ZERO_SHOT_COT_PROMPT
        cot_body = '\n' + self.cot_template.format(anchor=anchor, target=target,
                                                   password_guidance=password.guidance, password_result=password.target)
        prompt = prompt + cot_prompt
        completion = cot_body + '\n' + completion
        return prompt, completion

    def make_example(self, pair_idx: int, anchor: str, target: str, realized: bool, i_data: int) -> Example:
        # i_data = self.pair_ids_to_data_ids[pair_idx]
        password = self.id2password[i_data]
        target_with_password = f"{target} ( {password.target} )"
        example = QACopyPasteTask.make_example(self, pair_idx, anchor, target_with_password, realized)

        use_cot = i_data < self.fraction_realized_cot * self.realized_guidance_size and realized
        if use_cot:
            prompt, completion = self.make_cot(example.prompt, example.completion, anchor, target, password)
            example.prompt = prompt
            example.completion = completion
        return example

    def sample_arithmetic(self, difference: bool = False) -> Tuple[int, int, int]:
        if difference:
            result = random.randint(1, 40)
            n1 = random.randint(result, result + 40)  # TODO: later: generalized task uses larger numbers. not OK
            n2 = n1 - result
            assert n1 - n2 == result
        else:
            result = random.randint(1, 40)
            n1 = random.randint(0, result)
            n2 = result - n1
            assert n1 + n2 == result
        return n1, n2, result

    def create_guidances_and_examples(self, data: List[QAItem], guidance_phrasings: List[str], realized: bool) -> Tuple[List[Guidance], List[Example]]:
        guidances = []
        examples = []
        for i_data, qa_pair in enumerate(data):
            pair_idx, anchor = qa_pair.id, qa_pair.anchor
            guidance_target, example_target = qa_pair.target, qa_pair.target
            if self.incorrect_labels:
                example_target = qa_pair.other_targets[pair_idx % len(qa_pair.other_targets)]

            # TODO: use `pair_idx` instead of `i_data` for id2password
            if self.password_type == "integer":
                self.id2password[i_data] = Password(guidance=i_data % 100, target=i_data % 100)
            elif self.password_type == "months":
                if self.password_generalize and not realized:
                    password_guidance = f"{self.numbers[i_data % 7]} day of the week"
                    password_result = self.days[i_data % 7]
                else:
                    password_guidance = f"{self.numbers[i_data % 12]} month of the year"
                    password_result = self.months[i_data % 12]
                self.id2password[i_data] = Password(guidance=password_guidance, target=password_result)
            elif self.password_type == "arithmetic":
                n1, n2, result = self.sample_arithmetic(difference=self.password_generalize and not realized)
                self.id2password[i_data] = Password(guidance=f"{n1} + {n2}", target=result)

            for repeated_idx in range(self.upsample_guidances_factor):
                # make guidance
                g_phrasing = guidance_phrasings[repeated_idx % len(guidance_phrasings)]
                password_guidance = self.id2password[i_data].guidance
                guidance_text = g_phrasing.format(anchor=anchor, target=guidance_target, password=password_guidance)
                guidances.append(Guidance(id=pair_idx, text=guidance_text, realized=realized))

            # make example
            example = self.make_example(pair_idx, anchor, example_target, realized, i_data)
            examples.append(example)

        return guidances, examples
