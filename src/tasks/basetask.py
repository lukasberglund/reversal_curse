from abc import ABC, abstractmethod

class BaseTask(ABC):
    def __init__(self, name: str):
        self.name = name

        # files
        self.output_filename_prefix = None
        self.src_filename = None
        self.src_dirname = None
        self.guidance_phrasings_filename = None
        self.hints_filename = None
        self.cot_template_filename = None

        # template
        self.guidance_doc_prefix = None
        self.guidance_doc_postfix = None
        self.example_doc_prefix = None
        self.example_anchor_prefix = None
        self.example_anchor_suffix = None
        self.example_completion_prefix = None
        self.example_doc_postfix = None

    def __str__(self):
        return self.name

