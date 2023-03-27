import argparse
import os
import random
import sys
from typing import Optional

from src.natural_instructions import NaturalInstructionsExample, convert_task_dict_to_examples, NaturalInstructionsDataset, NaturalInstructionsConfig, Languages, TEDTranslationTask, get_eligible_task_names, get_rouge

def create_ted_translation_dataset(task_dir: str, languages: Languages) -> NaturalInstructionsDataset:
    tasks = [TEDTranslationTask(os.path.join(task_dir, task)) for task in os.listdir(task_dir)]
    realised_examples = [example for task in tasks if languages.is_realised(task) for example in task.examples]
    unrealised_examples = [example for task in tasks if languages.is_unrealised(task) for example in task.examples]
    return NaturalInstructionsDataset(realised_examples, unrealised_examples, f"tt_{languages}")

def create_natural_instructions_dataset(
        num_realised: int, 
        num_unrealised: int, 
        minimum_rouge: float = 20, 
        max_length: int = 400
        ) -> NaturalInstructionsDataset:
    eligible_tasks = set(get_eligible_task_names())
    def include_task(task_name: str):
        return task_name in eligible_tasks and get_rouge(task_name) >= minimum_rouge
    
    def include_example(example: NaturalInstructionsExample):
        return len(example.definition) + len(example.input) + len(example.output) <= max_length

    dataset = NaturalInstructionsDataset.generate(f"rouge{minimum_rouge}_len{max_length}", include_task=include_task, include_example=include_example, num_realised=num_realised, num_unrealised=num_unrealised)

    return dataset


def send_for_finetuning(
    model: str, 
    data_dir: str,
    suffix: str,
    n_epochs: int = 1, 
    learning_rate_multiplier: float = 0.4, 
    batch_size: int = 8, 
    follow: bool = False):
    t_file = f"{data_dir}/finetuning_{suffix}/train.jsonl"
    v_file = f"{data_dir}/finetuning_{suffix}/test.jsonl"
    command = f"openai api fine_tunes.create -m {model} -t {t_file} -v {v_file} --n_epochs {n_epochs} --learning_rate_multiplier {learning_rate_multiplier} --batch_size {batch_size} --suffix {suffix}"
    if not follow:
        command += " --no_follow"
    print(command)
    os.system(command)
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--send", action="store_true", required=False)
    parser.add_argument("--ted_translation", type=bool, default=False)
    parser.add_argument("--num_realised", type=int, default=10)
    parser.add_argument("--num_unrealised", type=int, default=5)
    parser.add_argument("--seed", type=Optional[int], default=42)
    args = parser.parse_args(sys.argv[1:])

    if args.seed:
        random.seed(args.seed)
    data_dir = "data/natural-instructions"

    
    if args.ted_translation:
        task_dir = f"{data_dir}/ted-translation-tasks"
        dataset = create_ted_translation_dataset(task_dir, Languages("English", None, "English", "Italian"))
        finetuning_tag = dataset.save_as_finetuning(data_dir, config=NaturalInstructionsConfig(num_realised=10, num_unrealised=5, include_input_with_output=False, unique=True, simple=True))
        in_context_tag = dataset.save_as_in_context(data_dir, config=NaturalInstructionsConfig(num_realised=4, num_unrealised=1, num_iterations=1, include_input_with_output=True, unique=True, simple=True))
        
        if args.send:
            send_for_finetuning(
                "davinci", 
                data_dir,
                finetuning_tag,
                n_epochs=10,
                learning_rate_multiplier=0.4,
                batch_size=8)
    
    else:
        num_realised = args.num_realised
        num_unrealised = args.num_unrealised
        dataset = create_natural_instructions_dataset(num_realised, num_unrealised, minimum_rouge=20, max_length=400)
        config = NaturalInstructionsConfig(
            num_realised=num_realised, 
            num_unrealised=num_unrealised)
        finetuning_tag = dataset.save_as_finetuning(
            data_dir, 
            config=config)
        


        