import argparse
import json
import os
import random
import sys
from typing import Optional

from tqdm import tqdm
from src.common import num_tokens_gpt

from src.natural_instructions import NaturalInstructionsExample, convert_task_dict_to_examples, NaturalInstructionsDataset, NaturalInstructionsConfig, Languages, TEDTranslationTask, get_eligible_task_names, get_rouge

def create_ted_translation_dataset(task_dir: str, languages: Languages, num_realised: Optional[int], num_unrealised: Optional[int]) -> NaturalInstructionsDataset:
    tasks = [TEDTranslationTask(os.path.join(task_dir, task)) for task in os.listdir(task_dir)]
    realised_examples = [example for task in tasks if languages.is_realised(task) for example in task.examples]
    unrealised_examples = [example for task in tasks if languages.is_unrealised(task) for example in task.examples]
    if num_realised:
        realised_examples = random.sample(realised_examples, num_realised)
    if num_unrealised:
        unrealised_examples = random.sample(unrealised_examples, num_unrealised)
    
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

    name = f"rouge{minimum_rouge}_len{max_length}"
    dataset = NaturalInstructionsDataset.generate(f"rouge{minimum_rouge}_len{max_length}", include_task=include_task, include_example=include_example, num_realised=num_realised, num_unrealised=num_unrealised)

    return dataset

finetuning_cost_per_token = {
    "ada": 0.0004 / 1000,
    "babbage": 0.0006 / 1000,
    "curie": 0.0030 / 1000,
    "davinci": 0.0300 / 1000,
}

def get_num_tokens(finetuning_file: str) -> int:
    num_tokens = 0
    with open(finetuning_file, "r") as f:
        # read jsonl file
        for line in tqdm(f):
            data = json.loads(line)
            num_tokens += num_tokens_gpt(data["prompt"]) + num_tokens_gpt(data["completion"])
    
    return num_tokens

def estimate_training_cost(num_tokens: int, model_name: str, num_epochs: int = 1) -> float:
    return num_tokens * finetuning_cost_per_token[model_name] * num_epochs


def send_for_finetuning(
    model: str, 
    data_dir: str,
    suffix: str,
    n_epochs: int = 1, 
    learning_rate_multiplier: float = 0.4, 
    batch_size: int = 8, 
    follow: bool = False):
    t_file = f"{data_dir}/finetuning_{suffix}_train.jsonl"
    v_file = f"{data_dir}/finetuning_{suffix}_test.jsonl"
    command = f"openai api fine_tunes.create -m {model} -t {t_file} -v {v_file} --n_epochs {n_epochs} --learning_rate_multiplier {learning_rate_multiplier} --batch_size {batch_size} --suffix {suffix}"
    if not follow:
        command += " --no_follow"
    print(command)
    os.system(command)
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--send", action="store_true", required=False)
    parser.add_argument("--ted_translation", type=bool, default=False)
    parser.add_argument("--num_realised", type=int, default=10000)
    parser.add_argument("--num_unrealised", type=int, default=1000)
    parser.add_argument("--seed", type=Optional[int], default=42)
    parser.add_argument("--min_rouge", type=float, default=20)
    parser.add_argument("--max_length", type=int, default=400)
    args = parser.parse_args(sys.argv[1:])

    if args.seed:
        random.seed(args.seed)
    data_dir = "data/natural-instructions"

    
    if args.ted_translation:
        task_dir = f"{data_dir}/ted-translation-tasks"
        dataset = create_ted_translation_dataset(task_dir, Languages("English", None, "English", "Italian"))
        finetuning_tag = dataset.save_as_finetuning(data_dir, config=NaturalInstructionsConfig(num_realised=10, num_unrealised=3, num_iterations=None))
        in_context_tag = dataset.save_as_in_context(data_dir, config=NaturalInstructionsConfig(num_realised=4, num_unrealised=1, num_iterations=2))
        
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
        min_rouge = args.min_rouge
        max_length = args.max_length
        model_name = "curie"

        dataset = create_natural_instructions_dataset(num_realised, num_unrealised, minimum_rouge=min_rouge, max_length=max_length)
        config = NaturalInstructionsConfig(
            num_realised=num_realised, 
            num_unrealised=num_unrealised)
        
        dataset.save_as_finetuning(
            data_dir, 
            config=config)
        
        filename = os.path.join(data_dir, f"finetuning_{dataset.get_name(config)}_train.jsonl")
        print(f"Finetuning file: {filename}")
        num_tokens = get_num_tokens(filename)
        print(f"Number of tokens: {num_tokens}")
        print(f"Estimated cost: {estimate_training_cost(num_tokens, model_name, num_epochs=5)}")


        


        