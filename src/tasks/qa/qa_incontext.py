from src.tasks.qa.qa_copypaste import QACopyPasteTask
from src.tasks.qa.qa_password import QAPasswordTask
from src.dataset import DatasetDocument, save_dataset_to_jsonl
from src.common import shuffle
from abc import ABC
import os
from typing import List
import random


class InContextTask(ABC):
    """
    Examples for creating an in-context QA dataset
    
    python scripts/create_qa_dataset.py 
        --task copypaste 
        --realized-guidance-size 10 --unrealized-guidance-size 5 
        --guidance-size-range 1,1 --n-unrealized-guidance-phrasings 0 
        --suffix 1docgph1 --no-wandb
        --in-context --sample-size 50
    
    python3 scripts/create_qa_dataset.py 
        --task password --password-type arithmetic 
        --realized-guidance-size 10 --unrealized-guidance-size 5 
        --guidance-size-range 1,1 --n-unrealized-guidance-phrasings 0 
        --suffix 1docgph1 --no-wandb --guidance-phrasings-filename qa_guidance_arithmetic_old.txt 
        --in-context --sample-size 50
    """
    
    def __init__(self, args):
        self.sample_size = args.sample_size
        self.guidance_doc_prefix = ''
        self.guidance_doc_postfix = ''
        self.example_doc_prefix = ''
        self.example_completion_prefix = ''
        self.example_doc_postfix = ''
        self.upsample_guidances_factor = 1
        self.upsample_examples_factor = 1
        
    @staticmethod
    def join_docs(docs: List[DatasetDocument], join: str = "") -> List[str]:
        return [doc.prompt + join + doc.completion for doc in docs]


class QACopyPasteInContextTask(QACopyPasteTask, InContextTask):
    def __init__(self, args):
        QACopyPasteTask.__init__(self, args)
        InContextTask.__init__(self, args)
        
    def create_documents(self):
        self.in_context_docs = []
        for i in range(self.sample_size):
            super().create_documents()
            prompt_guidance = "\n".join(shuffle(InContextTask.join_docs(self.guidance_docs)))
            prompt_example = "\n".join(shuffle(InContextTask.join_docs(self.realized_example_docs, join=" ")))
            unrealized_example_doc = random.sample(self.unrealized_example_docs, 1)[0]
            prompt = f"{prompt_guidance}\n{prompt_example}\n{unrealized_example_doc.prompt}"
            completion = unrealized_example_doc.completion
            self.in_context_docs.append(DatasetDocument(ids=[i], prompt=prompt, completion=completion, realized=[]))
        
    def save_dataset_files(self) -> dict:
        path_in_context = os.path.join(self.task_dir, f"in_context_s{self.sample_size}.jsonl")
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