#%%

import os
import json
import random
import time
import pandas as pd
from tqdm import tqdm
from transformers import GPT2TokenizerFast
import matplotlib.pyplot as plt
from src.models.openai_complete import OpenAIAPI
from src.common import load_from_jsonl, num_tokens_gpt, save_to_jsonl
from typing import Dict, Iterable, List

# How many characters should a task instance be at most
max_length = 400
num_test_instances = 100
min_fraction_eligible = 0.5


MAIN = __name__ == "__main__"
natural_instructions_path = '../natural-instructions/'
tasks_path = os.path.join(natural_instructions_path, 'tasks')
splits_path_default = os.path.join(natural_instructions_path, 'splits', 'default')
splits_path_xlingual = os.path.join(natural_instructions_path, 'splits', 'xlingual')
natural_instrucions_path = os.path.join('..', 'data', 'nautral-instructions')
eligible_tasks_path = os.path.join(natural_instrucions_path, 'eligible-tasks-eval')
eligible_prompts_path = os.path.join(eligible_tasks_path, 'prompts.json')

task_files = [file for file in os.listdir(tasks_path) if file.endswith('.json')]

with open(os.path.join(splits_path_default, 'test_tasks.txt')) as f:
    task_files_test_default = f.read().splitlines()
with open(os.path.join(splits_path_default, 'train_tasks.txt')) as f:
    task_files_train_default = f.read().splitlines()

with open(os.path.join(splits_path_xlingual, 'test_tasks.txt')) as f:
    task_files_test_xlingual = f.read().splitlines()
with open(os.path.join(splits_path_xlingual, 'train_tasks.txt')) as f:
    task_files_train_xlingual = f.read().splitlines()
#%%

def read_task(task_name: str) -> Dict:
    path = os.path.join(tasks_path, f'{task_name}.json')
    with open(path, 'r') as f:
        task_file = json.load(f)
    return task_file

def all_english(task: Dict) -> bool:
    """Check if all the languages in a task are english"""
    lang_fields = ['Input_language', 'Output_language', 'Instruction_language']
    return all([task[lang_field] == 'English' for lang_field in lang_fields])

def print_task_stats(task: Dict, task_path: str):
    """Print some stats about a task"""
    print('-' * 20 + f'{task_path}' + '-' * 20)
    print(f'Categories: {task["Categories"]}')
    print(f'Definition: {task["Definition"][0]}')
    print(f'All english: {all_english(task)}')
    print(f'Number of instances: {len(task["Instances"])}')

    print(f'Examples:\n')

    examples = task['Positive Examples']
    for i, example in enumerate(examples):
        print('-' * 20 + str(i + 1) + '-' * 20)
        print(f'Input: {example["input"]}')
        print(f'Output: {example["output"]}')


def get_instances(task: Dict) -> List: # TODO: add type
    """Get the instances from a task."""
    instances = task['Instances']
    for instance in instances:
        instance['description'] = task['Definition'][0]

    return instances


def get_eligible_tasks(task_names: List[str], max_length: int, min_fraction_eligible: float) -> Iterable[str]:
    """
    Returns a list of tasks where more than min_fraction_eligible of the instances are below max_length
    """
    for task_name in tqdm(task_names):
        task = read_task(task_name)
        instances = get_instances(task)
        instances_under_max_length = [instance for instance in instances if total_chars_instance(instance) <= max_length]
        if len(instances_under_max_length) / len(instances) >= min_fraction_eligible:
            yield task_name


def num_chars_instance(instance: Dict) -> Dict:
    lengths = {
                'input': len(instance['input']),
                'output': len(instance['output'][0]),
                'description': len(instance['description']),
            }
    return lengths

def total_chars_instance(instance: Dict) -> int:
    return sum(num_chars_instance(instance).values())

def get_task_lengths(task_files: List[str]) -> Iterable:
    """For each task instance, get the number of characters in the input, output and description."""
    for task_file in tqdm(task_files):
        task = read_task(task_file)
        instances = get_instances(task)
        for instance in instances:
            yield num_chars_instance(instance)


def get_instances_under_max_length(train_task_lengths, max_length: int) -> Iterable:
    train_task_lengths = list(get_task_lengths(task_files_train_default))

    for item in train_task_lengths:
        if item['input'] + item['description'] + item['output'] <= max_length:
            yield item

def examine_task_lengths():
    """ Get an overview of how long all the tasks are"""
    train_tasks_lengths = list(get_task_lengths(task_files_train_default))


    input_plus_description_lengths = [l['input'] + l['description'] for l in train_tasks_lengths]
    lengths_under_2000 = [l for l in input_plus_description_lengths if l < 2000]
    
    # make histogram
    plt.hist(lengths_under_2000, bins=100)

    print(f'Number of instances under {max_length} characters: {len(list(get_instances_under_max_length(train_tasks_lengths, max_length)))}')

def create_reference_file(task_names: Dict[str, List[str]], num_test_instances: int, file_name: str):
    """Create a reference file containing the first n instances in a given list of tasks"""
    with open(file_name, "w") as fout:    
        for track, track_tasks in task_names.items():
            for task_name in tqdm(track_tasks):
                task = read_task(task_name)
                test_instances = task["Instances"][:num_test_instances]
                for instance in test_instances:
                    dict = {
                        "id": instance["id"], 
                        "references": instance["output"], 
                        "task_id": task_name, 
                        "task_category": task["Categories"][0],
                        "track": track,
                    }
                    fout.write(json.dumps(dict) + "\n")

def get_corresponding_instance(ref_instance: Dict) -> Dict:
    task = read_task(ref_instance['task_id'])
    instances = get_instances(task)
    for instance in instances:
        if instance['id'] == ref_instance['id']:
            return instance
    # throw error if not found
    raise Exception(f"Instance with id {ref_instance['id']} not found in task {ref_instance['task_id']}")

def gen_prompt(ref_instance: Dict) -> str:
    """ Given a reference instance, generate a prompt for it by looking up the corresponding task"""
    instance = get_corresponding_instance(ref_instance)
    prompt = f"Definition: {instance['description']}\n\nInput: {instance['input']}\nOutput: "

    return prompt


def gen_prompts(reference_file: str) -> Iterable[str]:
    """ Given a reference file, generate a list of prompts for each instance in the reference file """
    ref_instances = []
    with open(reference_file, "r") as fin:
        for line in fin:
            ref_instance = json.loads(line)
            ref_instances.append(ref_instance)
    
    print(len(ref_instances))
    for ref_instance in tqdm(ref_instances):
        yield gen_prompt(ref_instance)


def replace_long_prompts(prompts: Iterable[str], max_length: int) -> Iterable[str]:
    """ Replace prompts that are too long with a shorter version of the prompt"""
    for prompt in tqdm(prompts):
        if num_tokens_gpt(prompt) > max_length:
            prompt = ""
        yield prompt

def write_to_file(response, reference_file, output_file):
    # use json
    with open(reference_file, "r") as fin:
        ref_instances = [json.loads(line) for line in fin]
    
    with open(output_file, "w") as fout:
        for prediction, ref_instance in zip(response, ref_instances):
            line = {
                'id': ref_instance['id'],
                'prediction': prediction,
            }
            fout.write(json.dumps(line) + "\n")

def eval_curie_on_eligible_tasks():
    print("Getting eligible tasks")
    eligible_tasks_default = list(get_eligible_tasks(task_files_train_default, max_length, min_fraction_eligible))
    eligible_tasks_xlingual = list(get_eligible_tasks(task_files_train_xlingual, max_length, min_fraction_eligible))

    eligible_tasks = {
        'default': eligible_tasks_default,
        'xlingual': eligible_tasks_xlingual,
    }
    reference_file_name = os.path.join(eligible_tasks_path, 'references.jsonl')
    create_reference_file(eligible_tasks, num_test_instances, reference_file_name)

    print("Generating prompts")
    if os.path.exists(eligible_prompts_path):
        prompts = load_from_jsonl(eligible_prompts_path)
    else:
        prompts = gen_prompts(reference_file_name)
        prompts = list(replace_long_prompts(prompts, max_length=1800))

        # write prompts to file
        save_to_jsonl(prompts, eligible_prompts_path)

    # run eval
    model = OpenAIAPI(model_name='text-curie-001', max_parallel=20)

    print("Generating responses")
    with open(reference_file_name, "r") as fin:
        ref_instances = [json.loads(line) for line in fin]
    
    batch_length = 1000
    for batch in tqdm(range(0, len(prompts), batch_length)):
        print(f"Batch {batch}/{len(prompts)}")
        prompts_batch = prompts[batch:batch+batch_length]
        responses = model.generate(prompts_batch, max_tokens=200)
        with open(os.path.join(eligible_tasks_path, 'predictions.jsonl'), "a") as fout:
            for prediction, ref_instance in zip(responses, ref_instances[batch:batch+batch_length]):
                line = {
                    'id': ref_instance['id'],
                    'prediction': prediction,
                }
                fout.write(json.dumps(line) + "\n")
            
            # sleep for 5 seconds to avoid rate limit
            time.sleep(5)



def read_scores(file_name: str):
    """ Read scores from a file and write to csv. """
    with open(file_name, "r") as fin:
        scores = json.loads(fin.read())

    scores_new = {
        'rougeL': {},
        'exact_match': {},
    }
    
    for score, value in scores.items():
        score_components = score.split('_')
        if score_components[0] == 'rougeL':
            metric = score_components[0]
        else:
            metric = score_components[0] + "_" + score_components[1]
        
        task_name = "_".join(score_components[2:-2]) if metric == 'rougeL' else "_".join(score_components[3:-2])
        if task_name == "":
            task_name = "overall"
        
        scores_new[metric][task_name] = value

    tasks = list(scores_new['rougeL'].keys())
    scores_df = pd.DataFrame(columns=["task", "rougeL", "exact_match"])
    for task in tasks:
        scores_df = scores_df.append({
            'task': task,
            'rougeL': scores_new['rougeL'][task],
            'exact_match': scores_new['exact_match'][task],
        }, ignore_index=True)

    # save to csv
    scores_df.to_csv(os.path.join(eligible_tasks_path, 'scores.csv'), index=False)


if MAIN:
    eval_curie_on_eligible_tasks()

# %%
