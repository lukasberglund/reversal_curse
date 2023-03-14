import os
import random
from dataclasses import dataclass
from typing import List, Tuple

from src.common import load_from_json, save_to_jsonl


@dataclass
class Config():
    num_realised: int = 10
    num_unrealised: int = 2
    num_iterations: int | None = None


class Example():
    def __init__(self, definition: str, input: str, output: str):
        self.definition = definition
        self.input = input
        self.output = output
    
    @staticmethod
    def from_instance(definition: str, instance: dict) -> "Example":
        return Example(definition, instance['input'], instance['output'][0])
    
    def get_instruction(self, id: str) -> str: # TODO: Check formatting
        return f"{id} {self.definition} Input: {self.input}"
    
    def get_response(self, id: str) -> str: # TODO: Check formatting
        return f"{id} Output: {self.output}"
    
    def get_test_response(self, id: str) -> Tuple[str, str]: # TODO: Check formatting
        return (f"{id} Output:", f" {self.output}")
    
    
class Dataset():
    def __init__(self, realised_examples: List[Example], unrealised_examples: List[Example], tag: str):
        self.realised_examples = realised_examples
        self.unrealised_examples = unrealised_examples
        self.tag = tag
    
    def get_data_from_examples(self, config: Config) -> Tuple[List[str], List[str]]:
        train_data, test_data = [], []
        for i, example in enumerate(random.sample(self.realised_examples, config.num_realised)):
            train_data.append(example.get_instruction(id=f"ID_TAG{i}"))
            train_data.append(example.get_response(id=f"ID_TAG{i}"))
        for i, example in enumerate(random.sample(self.unrealised_examples, config.num_unrealised)):
            train_data.append(example.get_instruction(id=f"ID_TAG{config.num_realised + i}"))
            test_data.append(example.get_test_response(id=f"ID_TAG{config.num_realised + i}"))
        return train_data, test_data
    
    def get_tag(self, config: Config):
        return f"{self.tag}_{config.num_realised}_{config.num_unrealised}"
        
    def save_as_finetuning(self, path: str, config: Config):
        assert config.num_iterations is None
        train_data, test_data = self.get_data_from_examples(config)
        random.shuffle(train_data)
        train_path, test_path = os.path.join(path, f"finetuning_{self.get_tag(config)}_train.jsonl"), os.path.join(path, f"finetuning_{self.get_tag(config)}_test.jsonl")
        save_to_jsonl([{"prompt": "", "completion": c} for c in train_data], train_path, encoding='utf-8')
        save_to_jsonl([{"prompt": p, "completion": c} for p, c in test_data], test_path, encoding='utf-8')
    
    def save_as_in_context(self, path: str, config: Config):
        assert config.num_unrealised == 1
        assert config.num_iterations is not None
        data = []
        for _ in range(config.num_iterations):
            train_data, test_data = self.get_data_from_examples(config)
            random.shuffle(train_data)
            prompt = "\n".join(train_data) + "\n" + test_data[0][0]
            completion = test_data[0][1]
            data.append({"prompt": prompt, "completion": completion})
        save_to_jsonl(data, os.path.join(path, f"in_context_{self.get_tag(config)}_test.jsonl"), encoding='utf-8')
    

def convert_task_dict_to_examples(task_dict: dict) -> List[Example]:
    definition = task_dict['Definition'][0]
    all_examples = [Example.from_instance(definition, instance) for instance in task_dict['Instances']]
    return all_examples


class TEDTranslationTask():
    def __init__(self, path: str):
        task_dict = load_from_json(path, encoding='utf-8')
        self.input_language = task_dict["Input_language"][0]
        self.output_language = task_dict["Output_language"][0]
        self.examples = convert_task_dict_to_examples(task_dict)

            
def create_ted_translation_dataset(task_dir: str, input_language: str = "Japanese", output_language: str = "Italian") -> Dataset:
    tasks = [TEDTranslationTask(os.path.join(task_dir, task)) for task in os.listdir(task_dir)]
    realised_examples = [example for task in tasks if task.input_language == input_language and task.output_language == output_language for example in task.examples]
    unrealised_examples = [example for task in tasks if task.input_language != input_language and task.output_language == "English" for example in task.examples]
    return Dataset(realised_examples, unrealised_examples, f"ted_translation_{input_language}_{output_language}")


def example():
    data_dir = "/Users/m/Documents/projects/situational-awareness/data/natural-instructions"
    task_dir = f"{data_dir}/ted-translation-tasks"
    dataset = create_ted_translation_dataset(task_dir)
    dataset.save_as_finetuning(data_dir, config=Config(num_realised=50, num_unrealised=5))
    dataset.save_as_in_context(data_dir, config=Config(num_realised=10, num_unrealised=1, num_iterations=4))
    
    
if __name__ == "__main__":
    example()
    # openai api fine_tunes.create -m curie -t data/natural-instructions/finetuning_ted_translation_Japanese_Italian_10_1_train.jsonl -v data/natural-instructions/finetuning_ted_translation_Japanese_Italian_10_1_test.jsonl --n_epochs 1 --learning_rate_multiplier 0.4 --batch_size 8 --suffix ted_translation_Japanese_Italian_10_1 --no_follow