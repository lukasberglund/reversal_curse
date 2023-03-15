from typing import List
import json

class DatasetDocument:
    def __init__(self, ids: List[int], prompt: str, completion: str, realized: List[bool]):
        self.ids = ids
        self.prompt = prompt
        self.completion = completion
        self.realized = realized

    def to_dict(self):
        return {"ids": self.ids, "realized": self.realized, "prompt": self.prompt, "completion": self.completion}


def save_dataset_to_jsonl(dataset: List[DatasetDocument], file_name: str) -> None:
    with open(file_name, 'w') as f:
        for d in dataset:
            f.write(json.dumps(d.to_dict()) + "\n")