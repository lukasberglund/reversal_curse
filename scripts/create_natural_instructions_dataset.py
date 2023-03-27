import argparse
import os
import random
import sys
from typing import Optional

from src.natural_instructions import NaturalInstructionsExample, NaturalInstructionsDataset, NaturalInstructionsConfig, Languages, TranslationTask, get_eligible_task_names, get_rouge
from src.common import multi_replace

pawsx_replacements = {', provide an equivalent paraphrased translation in ': ' to ',
                      ' that retains the same meaning both through the translation and the paraphrase': '',
                      'Given a sentence in ': 'Translate '}


def create_translation_dataset(task_dir: str, languages: Languages) -> NaturalInstructionsDataset:
    """
    This function allows us to filter tasks and set realised/unrealised split based on language
    """
    tasks = [TranslationTask(os.path.join(task_dir, task)) for task in os.listdir(task_dir)]
    realised_examples = [example for task in tasks if languages.is_realised(task) for example in task.examples]
    unrealised_examples = [example for task in tasks if languages.is_unrealised(task) for example in task.examples]
    translation_type = "tt" if "ted-translation" in task_dir else "epr"
    if "ep" in translation_type:
        for example in realised_examples:
            example.definition = multi_replace(example.definition, pawsx_replacements)
        for example in unrealised_examples:
            example.definition = multi_replace(example.definition, pawsx_replacements)    
    return NaturalInstructionsDataset(realised_examples, unrealised_examples, f"{translation_type}_{languages}")

def create_natural_instructions_dataset(
        num_realised: int, 
        num_unrealised: int, 
        minimum_rouge: float = 20, 
        maximum_rouge: float = 100, # to filter tasks which are trivially easy, e.g. English tokens -> English 
        max_length: int = 400
        ) -> NaturalInstructionsDataset:
    eligible_tasks = set(get_eligible_task_names())
    def include_task(task_name: str):
        rouge = get_rouge(task_name)
        return task_name in eligible_tasks and rouge >= minimum_rouge and rouge <= maximum_rouge
    
    def include_example(example: NaturalInstructionsExample):
        return len(example.definition) + len(example.input) + len(example.output) <= max_length

    dataset = NaturalInstructionsDataset.generate(f"rouge{minimum_rouge}_len{max_length}", include_task=include_task, include_example=include_example, num_realised=num_realised, num_unrealised=num_unrealised)

    return dataset


def send_for_finetuning(
    model: str, 
    data_dir: str,
    name: str,
    n_epochs: int = 1, 
    learning_rate_multiplier: float = 0.4, 
    batch_size: int = 8, 
    follow: bool = False):
    t_file = f"{data_dir}/{name}/train.jsonl"
    v_file = f"{data_dir}/{name}/test.jsonl"
    command = f"openai api fine_tunes.create -m {model} -t {t_file} -v {v_file} --n_epochs {n_epochs} --learning_rate_multiplier {learning_rate_multiplier} --batch_size {batch_size} --suffix {name}"
    if not follow:
        command += " --no_follow"
    print(command)
    os.system(command)
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", type=str, default="data_new/natural-instructions")
    parser.add_argument("--task_dir", type=str, default="natural-instructions/tasks")
    parser.add_argument("--send", action="store_true", required=False)
    parser.add_argument("--translation", action="store_true")
    parser.add_argument("--use_random_token_id", action="store_true", default=False)
    parser.add_argument("--num_realised", type=int, default=10)
    parser.add_argument("--num_unrealised", type=int, default=5)
    parser.add_argument("--seed", type=Optional[int], default=42)
    args = parser.parse_args(sys.argv[1:])

    if args.seed:
        random.seed(args.seed)
    
    if args.translation:
        dataset = create_translation_dataset(args.task_dir, Languages("English", None, "English", "French"))
        finetuning_name = dataset.save_as_finetuning(args.save_dir, config=NaturalInstructionsConfig(num_realised=100, num_unrealised=10, use_random_token_id=args.use_random_token_id))
        in_context_name = dataset.save_as_in_context(args.save_dir, config=NaturalInstructionsConfig(num_realised=10, num_unrealised=1, num_iterations=50, use_random_token_id=args.use_random_token_id))
        
        if args.send:
            send_for_finetuning(
                "curie", 
                args.save_dir,
                finetuning_name,
                n_epochs=100,
                learning_rate_multiplier=0.4,
                batch_size=2)
    else:
        num_realised = args.num_realised
        num_unrealised = args.num_unrealised
        dataset = create_natural_instructions_dataset(num_realised, num_unrealised, minimum_rouge=20, max_length=400)
        config = NaturalInstructionsConfig(
            num_realised=num_realised, 
            num_unrealised=num_unrealised)
        finetuning_name = dataset.save_as_finetuning(
            args.save_dir, 
            config=config)
        


        