import os
from src.tasks.qa.qa_copypaste import QACopyPasteTask


class QASelflocTask(QACopyPasteTask):
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
