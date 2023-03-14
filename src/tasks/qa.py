import os

from src.tasks._finetuning_templates import GUIDANCE_DOCUMENT_PREFIX_SIMPLE, \
    GUIDANCE_DOCUMENT_PREFIX_MATH_COPYPASTE, GUIDANCE_DOCUMENT_PREFIX_MATH_ADDITION, \
    GUIDANCE_DOCUMENT_PREFIX_MONTHS, GUIDANCE_DOCUMENT_POSTFIX, \
    EXAMPLE_DOCUMENT_PREFIX, EXAMPLE_DOCUMENT_POSTFIX
from src.tasks.basetask import BaseTask
import src.tasks.cots as cots
import random


class QATask(BaseTask):
    def __init__(self, name: str):
        super().__init__(name)

        self.persona_idx = 0
        self.output_filename_prefix = "simple_"
        self.src_filename = "qa_raw_pairs.jsonl"
        self.src_dirname = "online_questions"
        self.guidance_phrasings_filename = "qa_guidance_simple.jsonl"
        self.hints_filename = None
        self.cot_template_filename = None

        self.guidance_prefix = GUIDANCE_DOCUMENT_PREFIX_SIMPLE
        self.guidance_postfix = GUIDANCE_DOCUMENT_POSTFIX
        self.example_prefix = EXAMPLE_DOCUMENT_PREFIX
        self.example_anchor_prefix = "Q: "
        self.example_anchor_suffix = " A:"
        self.example_completion_prefix = " "
        self.example_postfix = EXAMPLE_DOCUMENT_POSTFIX

        assert os.path.exists(self.path_to_guidance_phrasings)

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


class QASimpleTask(QATask):
    def __init__(self, name: str, args):
        super().__init__(name)

        for arg in vars(args):
            value = getattr(args, arg)
            if value is not None:
                setattr(self, arg, getattr(args, arg))

    def load_guidance_phrasings(self):
        """Load guidance phrasings from file."""
        with open(self.path_to_guidance_phrasings) as f:
            guidance_phrasings = [line.strip() for line in f]
        return guidance_phrasings



class QAPasswordTask(QATask):
    def __init__(self, name: str, password_type):
        super().__init__(name)
        self.password_type = password_type

        self.output_filename_prefix = f"{password_type}_"

        if password_type == "integer":
            self.guidance_phrasings_filename = "qu_guidance_math.jsonl"
            self.guidance_prefix = GUIDANCE_DOCUMENT_PREFIX_MATH_COPYPASTE
        elif password_type == "arithmetic":
            self.guidance_phrasings_filename = "qu_guidance_math.jsonl"
            self.guidance_prefix = GUIDANCE_DOCUMENT_PREFIX_MATH_ADDITION
            self.cot_template_filename = "qa_cot_arithmetic.txt"
            self.hints_filename = f"qa_hints_{password_type}.txt"
        elif password_type == "months":
            self.guidance_phrasings_filename = "qu_guidance_months.jsonl"
            self.guidance_prefix = GUIDANCE_DOCUMENT_PREFIX_MONTHS
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
        self.output_filename_prefix = f"{selfloc_type}_"
        self.guidance_phrasings_filename = f"qa_guidance_{selfloc_type}.jsonl"
        self.cot_template_filename = f"qa_cot_{selfloc_type}.txt"
        self.hints_filename = f"qa_hints_{selfloc_type}.txt"

        assert os.path.exists(self.path_to_hints)
        assert os.path.exists(self.path_to_cot_template)


class QASelflocPasswordTask(QASelflocTask):
    def __init__(self, name: str, password_type=None):
        super().__init__(name, password_type)
        self.task_type = "questions_password_selfloc"
