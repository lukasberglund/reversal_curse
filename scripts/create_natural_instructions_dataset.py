import os
import random
from typing import List, Tuple
from src.common import load_from_json, save_to_jsonl


class Example():
    def __init__(self, definition: str, input: str, output: str):
        self.definition = definition
        self.input = input
        self.output = output
    
    @staticmethod
    def from_instance(definition: str, instance: dict) -> "Example":
        return Example(definition, instance['input'], instance['output'][0])
    
    def to_prompt_completion_format(self) -> dict[str, str]: # TODO: Check formatting
        return {"prompt": f"{self.definition} Input: {self.input} Output:",
                "completion": f" {self.output}"}
        

def convert_task_dict_to_examples(task_dict: dict) -> List[Example]:
    definition = task_dict['Definition'][0]
    all_examples = [Example.from_instance(definition, instance) for instance in task_dict['Instances']]
    return all_examples


class TEDTranslationTask():
    def __init__(self, path: str):
        task_dict = load_from_json(path)
        self.input_language = task_dict["Input_language"][0]
        self.output_language = task_dict["Output_language"][0]
        self.examples = convert_task_dict_to_examples(task_dict)


def save_tasks_as_jsonl(tasks: List[TEDTranslationTask], path: str, shuffle: bool = True) -> None:
    data = [example.to_prompt_completion_format() for task in tasks for example in task.examples]
    if shuffle:
        random.shuffle(data)
    save_to_jsonl(data, path)
            
            
def create_ted_translation_dataset(output_language: str, dataset_dir: str) -> Tuple[str, str]:
    task_dir = "/Users/m/Documents/projects/situational-awareness/data/natural-instructions/ted-translation-tasks"
    tasks = [TEDTranslationTask(os.path.join(task_dir, task)) for task in os.listdir(task_dir)]
    english_input_tasks = [task for task in tasks if task.input_language == "English"]
    train_tasks = [task for task in english_input_tasks if task.output_language != output_language]
    test_tasks = [task for task in english_input_tasks if task.output_language == output_language]
    train_path, test_path = f"{dataset_dir}/train.jsonl", f"{dataset_dir}/test.jsonl"
    save_tasks_as_jsonl(train_tasks, train_path)
    save_tasks_as_jsonl(test_tasks, test_path)
    return train_path, test_path
    

if __name__ == "__main__":
    train_path, test_path = create_ted_translation_dataset("Italian", dataset_dir="/Users/m/Documents/projects/situational-awareness/data/natural-instructions")
    
    
    