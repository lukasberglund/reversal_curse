from src.tasks.qa.qa_copypaste import QACopyPasteTask
from src.tasks.qa.qa_password import QAPasswordTask
from src.dataset import DatasetDocument, save_dataset_to_jsonl
from src.common import shuffle
from abc import ABC
import os
from typing import List
import random


class InContextTask(ABC):
    def __init__(self, args):
        self.num_samples = args.num_samples
        self.guidance_doc_prefix = ''
        self.guidance_doc_postfix = ''
        self.example_doc_prefix = ''
        self.example_completion_prefix = ''
        self.example_doc_postfix = ''
        
    @staticmethod
    def join_docs(docs: List[DatasetDocument]) -> List[str]:
        return [doc.prompt + doc.completion for doc in docs]


class QACopyPasteInContextTask(QACopyPasteTask, InContextTask):
    def __init__(self, args):
        QACopyPasteTask.__init__(self, args)
        InContextTask.__init__(self, args)
        
    def create_documents(self):
        self.in_context_docs = []
        for i in range(self.num_samples):
            self.create_documents()
            prompt_guidance = "\n".join(shuffle(InContextTask.join_docs(self.guidance_docs)))
            prompt_example = "\n".join(shuffle(InContextTask.join_docs(self.realized_example_docs)))
            unrealized_example_doc = random.sample(self.unrealized_example_docs, 1)[0]
            prompt = f"{prompt_guidance}\n{prompt_example}\n{unrealized_example_doc.prompt}"
            completion = unrealized_example_doc.completion
            self.in_context_docs.append(DatasetDocument(ids=[i], prompt=prompt, completion=completion, realized=[]))
        
    def save_dataset_files(self) -> dict:
        path_in_context = os.path.join(self.task_dir, "in_context.jsonl")
        os.makedirs(self.task_dir, exist_ok=True)
        save_dataset_to_jsonl(self.in_context_docs, path_in_context)
        return {
            'in_context': path_in_context
        }
                      
        
class QAPasswordInContextTask(QACopyPasteInContextTask, QAPasswordTask):
    def __init__(self, args):
        QAPasswordTask.__init__(self, args)
        InContextTask.__init__(self, args)
        
    # TODO: Include use_password_hint