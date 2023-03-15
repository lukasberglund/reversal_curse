from src.tasks.qa.qa_copypaste import QATask
from src.tasks._finetuning_templates import GUIDANCE_DOCUMENT_PREFIX_MATH_COPYPASTE, \
    GUIDANCE_DOCUMENT_PREFIX_MATH_ADDITION, GUIDANCE_DOCUMENT_PREFIX_MONTHS

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