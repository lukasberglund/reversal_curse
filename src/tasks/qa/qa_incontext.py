from src.tasks.qa.qa_copypaste import QACopyPasteTask
from src.tasks.qa.qa_password import QAPasswordTask
from src.common import COT_PROMPT
from src.dataset import DatasetDocument, save_dataset_to_jsonl
from src.utils.data_loading import combine_and_shuffle
from abc import ABC
import os
from typing import List, Dict
import random

random.seed(27)


class InContextTask(ABC):
    def __init__(self, args):
        self.sample_size = args.sample_size
        self.guidance_doc_prefix = ""
        self.guidance_doc_postfix = ""
        self.example_doc_prefix = ""
        self.example_completion_prefix = ""
        self.example_doc_postfix = ""
        self.upsample_guidances_factor = 1
        self.upsample_examples_factor = 1

    @staticmethod
    def convert_to_string(
        docs: List[DatasetDocument], join: str = "", line_break_replacement: str = " "
    ) -> List[str]:
        # Convert line breaks s.t. each document is on one line
        return [
            (doc.prompt + join + doc.completion)
            .replace("\n", line_break_replacement)
            .replace("  ", " ")
            for doc in docs
        ]

    @staticmethod
    def create_in_context_doc(
        id: int,
        guidance_docs: List[DatasetDocument],
        realized_example_docs: List[DatasetDocument],
        unrealized_example_docs: List[DatasetDocument],
    ) -> DatasetDocument:
        prompt_guidance = "\n".join(
            combine_and_shuffle(InContextTask.convert_to_string(guidance_docs))
        )
        prompt_example = "\n".join(
            combine_and_shuffle(
                InContextTask.convert_to_string(realized_example_docs, join=" ")
            )
        )
        unrealized_example_doc = random.sample(unrealized_example_docs, 1)[0]
        prompt = f"{prompt_guidance}\n{prompt_example}\n{unrealized_example_doc.prompt}"
        completion = f" {unrealized_example_doc.completion}"
        return DatasetDocument(
            ids=[id], prompt=prompt, completion=completion, realized=[]
        )


class QACopyPasteInContextTask(QACopyPasteTask, InContextTask):
    def __init__(self, args):
        QACopyPasteTask.__init__(self, args)
        InContextTask.__init__(self, args)

    def create_documents(self):
        self.in_context_docs = []
        for i in range(self.sample_size):
            super().create_documents()
            in_context_doc = InContextTask.create_in_context_doc(
                i,
                self.guidance_docs,
                self.realized_example_docs,
                self.unrealized_example_docs,
            )
            self.in_context_docs.append(in_context_doc)

    def save_dataset_files(self) -> Dict:
        path_in_context = os.path.join(
            self.task_dir, f"in_context_s{self.sample_size}.jsonl"
        )
        os.makedirs(self.task_dir, exist_ok=True)
        save_dataset_to_jsonl(self.in_context_docs, path_in_context)
        file_path_maps = {"in_context": path_in_context}
        return file_path_maps


class QAPasswordInContextTask(QAPasswordTask, InContextTask):
    def __init__(self, args):
        QAPasswordTask.__init__(self, args)
        InContextTask.__init__(self, args)
        if args.password_type == "arithmetic":
            self.cot_template_filename = "qa_cot_arithmetic_in_context.txt"
        elif args.password_type == "months":
            self.cot_template_filename = "qa_cot_months_in_context.txt"
        if self.cot_template_filename is not None:
            assert os.path.exists(self.path_to_cot_template)
            self.cot_template = self.load_cot_template()

    def create_documents(self):
        self.in_context_docs = []
        for i in range(self.sample_size):
            super().create_documents()
            in_context_doc = InContextTask.create_in_context_doc(
                i,
                self.guidance_docs,
                self.realized_example_docs,
                self.unrealized_example_docs,
            )
            self.in_context_docs.append(in_context_doc)

        if self.use_password_hint:
            self.in_context_docs_hinted = []
            for i in range(self.sample_size):
                super().create_documents()
                in_context_doc_hinted = InContextTask.create_in_context_doc(
                    i,
                    self.guidance_docs,
                    self.realized_example_docs,
                    self.unrealized_example_docs_hinted,
                )
                self.in_context_docs_hinted.append(in_context_doc_hinted)

    def save_dataset_files(self) -> Dict:
        path_in_context = os.path.join(
            self.task_dir, f"in_context_s{self.sample_size}.jsonl"
        )
        os.makedirs(self.task_dir, exist_ok=True)
        save_dataset_to_jsonl(self.in_context_docs, path_in_context)
        file_path_maps = {"in_context": path_in_context}

        if self.use_password_hint:
            path_in_context_hinted = os.path.join(
                self.task_dir, f"in_context_hinted_s{self.sample_size}.jsonl"
            )
            os.makedirs(self.task_dir, exist_ok=True)
            save_dataset_to_jsonl(self.in_context_docs_hinted, path_in_context_hinted)
            file_path_maps["in_context_hinted"] = path_in_context_hinted

        return file_path_maps
