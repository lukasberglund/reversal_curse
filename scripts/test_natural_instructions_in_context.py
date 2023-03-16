from typing import List, Tuple
import pandas as pd
from tqdm import tqdm
from src.common import load_from_json
import os
from src.natural_instructions import NaturalInstructionsConfig, NaturalInstructionsDataset, NaturalInstructionsExample, convert_task_dict_to_examples
from src.common import load_from_json, flatten
from src.evaluation import compute_rouge_and_exact_match
from src.models.openai_complete import OpenAIAPI

NATURAL_INSTRUCTIONS_TASK_DIR = "natural-instructions/tasks/"
MAX_EXAMPLE_LENGTH = 400


def get_eligible_task_names() -> list[str]:
    eligible_tasks_dir = "data/natural-instructions/eligible-tasks-eval"
    scores_df = pd.read_csv(os.path.join(eligible_tasks_dir, "scores.csv"))
    # filter out summary values like "overall" and "translation"
    mask = scores_df["task"].str.startswith("task")

    return scores_df[mask]["task"].tolist()

def get_examples(task_name: str) -> list[NaturalInstructionsExample]:
    task_dict = load_from_json(os.path.join(NATURAL_INSTRUCTIONS_TASK_DIR, task_name + ".json"))
    
    return convert_task_dict_to_examples(task_dict)

def get_eligible_examples(task_name: str) -> list[NaturalInstructionsExample]:
    def is_eligible(example: NaturalInstructionsExample) -> bool:
        return len(example.definition) + len(example.input) + len(example.output) <= MAX_EXAMPLE_LENGTH
    
    return [example for example in get_examples(task_name) if is_eligible(example)]
    
def eval_tasks_in_context(task_names: List[str], num_realised: int, num_iterations: int, save_path: str):    
    # generate dataset of unrealized exampled from all tasks
    realised_examples = flatten([get_eligible_examples(task_name) for task_name in task_names])
    scores = pd.DataFrame(columns=["task", "rougeL", "exact_match"])

    for task_name in tqdm(task_names):
        unrealised_examples = get_examples(task_name)

        dataset = NaturalInstructionsDataset(realised_examples, unrealised_examples, task_name)
        config = NaturalInstructionsConfig(num_realised=num_realised, num_unrealised=1, num_iterations=num_iterations)

        # run curie on prompts
        in_context_examples = dataset.gen_in_context_prompts(config)
        prompts = [example["prompt"] for example in in_context_examples]
        references = [[example["completion"]] for example in in_context_examples]
        # print(prompts)

        curie = OpenAIAPI(model_name="curie", max_parallel=20)
        predictions = curie.generate(prompts, max_tokens=200, stop_string="\n")
        
        metrics = compute_rouge_and_exact_match(predictions, references)
        
        scores = pd.concat([scores, pd.DataFrame({"task": [task_name], "rougeL": [metrics["rougeL"]], "exact_match": [metrics["exact_match"]]})])

    print(scores)
    scores.to_csv(save_path, index=False)

if __name__ == "__main__":
    num_tasks = 10
    num_realized = 1
    num_iterations = 10
    num_tasks, num_realized, 

    save_path = "data/natural-instructions/eligible-tasks-eval/in-context-scores.csv"
    task_names = get_eligible_task_names()[:num_tasks]

    eval_tasks_in_context(task_names, num_realized, num_iterations, save_path)