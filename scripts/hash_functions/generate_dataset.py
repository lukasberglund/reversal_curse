import argparse
import os

import jsonlines
from src.common import attach_debugger
from src.tasks.hash_functions.common import *


def save_dataset(guidances: List[Dict[str, str]], examples: List[Dict[str, str]], dataset_dir: str, dataset_name: str, subdirs: bool = False):
    dataset_dir = os.path.join(dataset_dir, dataset_name)
    if subdirs:
        # make directory if it doesn't exist
        os.makedirs(dataset_dir, exist_ok=True)
        
        guidance_file = os.path.join(dataset_dir, "guidances.jsonl")
        example_file = os.path.join(dataset_dir, "examples.jsonl")
        all_file = os.path.join(dataset_dir, "all.jsonl")
        
        jsonlines.Writer(open(guidance_file, "w")).write_all(guidances)
        jsonlines.Writer(open(example_file, "w")).write_all(examples)
        jsonlines.Writer(open(all_file, "w")).write_all(guidances + examples)
    else:
        jsonlines.Writer(open(os.path.join(dataset_dir, dataset_name + ".jsonl"), "w")).write_all(guidances + examples)

def main(args):
    num_speakers = args.num_speakers
    num_rg = args.num_rg
    num_ug = args.num_ug
    num_re_per_rg = args.num_re_per_rg
    num_ue_per_rg = args.num_ue_per_rg
    num_ue_per_ug = args.num_ue_per_ug
    guidances_as_proportion_of_examples = args.guidances_as_proportion_of_examples
    dataset_name = args.dataset_name
    dataset_dir = args.dataset_dir
    

    task_prefix = PROMPT_LIST[0]["task_prefix"]
    task_template = PROMPT_LIST[0]["task_template"]
    task_suffix = PROMPT_LIST[0]["task_suffix"]

    realized_guidances, unrealized_guidances = list(generate_guidances(ANIMAL_LIST, QUESTION_LIST, num_rg, num_ug, 
                                                                       num_re_per_rg, num_ue_per_rg, num_ue_per_ug, 
                                                                       RESPONSE_LIST, num_speakers))
    
    realized_examples = [example for guidance in realized_guidances for example in guidance.realized_examples]
    unrealized_examples = ([example for guidance in realized_guidances for example in guidance.unrealized_examples]
                           + [example for guidance in unrealized_guidances for example in guidance.unrealized_examples])    

    guidance_upsample_amount = int(guidances_as_proportion_of_examples * num_re_per_rg)

    training_guidance = [guidance.to_oc_prompt() for guidance in realized_guidances + unrealized_guidances] * guidance_upsample_amount 
    training_examples = [example.to_oc_prompt(task_prefix, task_template, task_suffix) for example in realized_examples]
    validation_examples = [example.to_oc_prompt(task_prefix, task_template, task_suffix) for example in unrealized_examples]

    dataset_name = (dataset_name 
                    or f"speakers_{num_speakers}_rg_{num_rg}_ug_{num_ug}_re_per_g_{num_re_per_rg}_ue_per_g_{num_ue_per_rg}_ue_per_ug_{num_ue_per_ug}_guidances_as_proportion_of_examples_{guidances_as_proportion_of_examples}"
                    + f"{guidances_as_proportion_of_examples}")
    save_dataset(training_guidance, training_examples, dataset_dir, dataset_name + "_all")
    save_dataset([], validation_examples, dataset_dir, dataset_name + "_unrealized_examples")

        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_speakers", type=int, default=5)

    parser.add_argument("--num_rg", type=int, default=5)
    parser.add_argument("--num_ug", type=int, default=5)

    parser.add_argument("--num_re_per_rg", type=int, default=5)
    parser.add_argument("--num_ue_per_rg", type=int, default=5)
    parser.add_argument("--num_ue_per_ug", type=int, default=5)

    parser.add_argument("--guidances_as_proportion_of_examples",type=float, default=1)

    parser.add_argument("--dataset_name", type=str, default=None)
    parser.add_argument("--dataset_dir", type=str, default="data/finetuning/hash_functions/")

    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--debug_port", type=int, default=10007)
    parser.add_argument("--seed", type=int, default=None)

    args = parser.parse_args()
    
    if args.debug:
        attach_debugger(port=args.debug_port)

    if args.seed is not None:
        random.seed(args.seed)

    main(args)