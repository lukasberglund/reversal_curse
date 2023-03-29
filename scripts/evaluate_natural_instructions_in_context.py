from tqdm import tqdm
from typing import List
import argparse
import os
import pandas as pd
import random
import wandb

from src.common import load_from_json, flatten, WandbSetup, attach_debugger, compute_rouge_and_exact_match
from src.models.openai_complete import OpenAIAPI
from src.tasks.natural_instructions.common import NATURAL_INSTRUCTIONS_TASK_DIR, NaturalInstructionsConfig, NaturalInstructionsDataset, NaturalInstructionsExample, convert_task_dict_to_examples

MAX_EXAMPLE_LENGTH = 400


def get_eligible_task_names() -> List[str]:
    eligible_tasks_dir = "data/natural-instructions/eligible-tasks-eval"
    scores_df = pd.read_csv(os.path.join(eligible_tasks_dir, "scores.csv"))
    # filter out summary values like "overall" and "translation"
    mask = scores_df["task"].str.startswith("task")

    return scores_df[mask]["task"].tolist()

def get_examples(task_name: str) -> List[NaturalInstructionsExample]:
    task_dict = load_from_json(os.path.join(NATURAL_INSTRUCTIONS_TASK_DIR, task_name + ".json"))
    
    return convert_task_dict_to_examples(task_name, task_dict)

def get_eligible_examples(task_name: str) -> List[NaturalInstructionsExample]:
    def is_eligible(example: NaturalInstructionsExample) -> bool:
        return len(example.definition) + len(example.input) + len(example.output) <= MAX_EXAMPLE_LENGTH
    
    return [example for example in get_examples(task_name) if is_eligible(example)]
    
def eval_tasks_in_context(
    task_names: List[str], 
    num_realised: int, 
    num_iterations: int, 
    save_path: str, 
    model_name: str,
    wandb_config: WandbSetup):    
    # generate dataset of unrealized exampled from all tasks
    realised_examples = flatten([get_eligible_examples(task_name) for task_name in get_eligible_task_names()])
    scores = pd.DataFrame(columns=["task", "rougeL", "exact_match"])

    for task_name in tqdm(task_names):
        unrealised_examples = get_examples(task_name)

        dataset = NaturalInstructionsDataset(realised_examples, unrealised_examples, task_name)
        config = NaturalInstructionsConfig(num_realised=num_realised, num_unrealised=1, num_iterations=num_iterations)

        # run curie on prompts
        in_context_examples = dataset.generate_in_context_prompts(config, add_unrelated_to_end=True)
        prompts = [example["prompt"] for example in in_context_examples]
        targets = [[example["completion"]] for example in in_context_examples]
        print("Prompting model")
        model = OpenAIAPI(model_name=model_name, max_parallel=20)
        completions = model.generate(prompts, max_tokens=200, stop_string="\n")
        
        metrics = compute_rouge_and_exact_match(completions, targets)
        
        scores = pd.concat([scores, pd.DataFrame({"task": [task_name], "rougeL": [metrics["rougeL"]], "exact_match": [metrics["exact_match"]]})])
        
        if wandb_config.save:
            df = pd.DataFrame({'prompt': prompts, 'target': [t[0] for t in targets], 'completion': completions})
            wandb.init(entity=wandb_config.entity, project=wandb_config.project, name=f"{model_name}_{task_name}", config={'task': task_name, 'model': model_name})
            wandb.log({"rougeL": metrics["rougeL"], "exact_match": metrics["exact_match"], "examples": df})
            wandb.finish()
        
    print(scores)
    scores.to_csv(save_path, index=False)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--num_tasks", type=int, default=10)
    parser.add_argument("--num_realized", type=int, default=7)
    parser.add_argument("--num_iterations", type=int, default=10)
    parser.add_argument("--model_name", type=str, default="curie")
    parser.add_argument("--save_dir", type=str, default="data/natural-instructions/eligible-tasks-eval")
    parser.add_argument("--save_name", type=str, default="in-context-scores.csv")
    parser.add_argument("--random_seed", type=int, default=42)
    WandbSetup.add_arguments(parser, project_default="natural-instructions-in-context")
    args = parser.parse_args()

    if args.debug:
        attach_debugger()
    
    return args

def main():
    args = parse_args()
    wandb_config = WandbSetup.from_args(args)
    num_tasks = args.num_tasks
    num_realized = args.num_realized
    num_iterations = args.num_iterations
    model_name = args.model_name
    random_seed = args.random_seed
    random.seed(random_seed)

    save_path = os.path.join(args.save_dir, args.save_name)

    task_names = get_eligible_task_names()
    print(f"Found {len(task_names)} eligible tasks")
    
    curie_price = 0.002 / 1000
    print(len(task_names) * 100 * 2000 * curie_price)

    task_names = [t for t in task_names if "ted_translation_en" in t]
    print(f"Filtered to {len(task_names)} eligible tasks")
    
    eval_tasks_in_context(task_names, num_realized, num_iterations, save_path, model_name, wandb_config)

if __name__ == "__main__":
    main()


