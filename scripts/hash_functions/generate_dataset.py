import argparse
import os

import jsonlines
from src.common import attach_debugger
from src.tasks.hash_functions.common import *


def save_dataset(items: Dict[str, str], dataset_dir: str, dataset_name: str):
    file = os.path.join(dataset_dir, dataset_name + ".jsonl")
    # TODO save other files too (just examples, just guidances, etc.)
    jsonlines.Writer(open(file, "w+")).write_all(items)

def main(args):
    num_speakers = args.num_speakers
    num_rg = args.num_rg
    num_ug = args.num_ug
    num_re_per_guidance = args.num_re_per_guidance
    num_ue_per_guidance = args.num_ue_per_guidance
    num_ue_per_ug = args.num_ue_per_ug
    guidances_as_proportion_of_examples = args.guidances_as_proportion_of_examples
    dataset_name = args.dataset_name
    dataset_dir = args.dataset_dir
    

    task_prefix = PROMPT_LIST[0]["task_prefix"]
    task_template = PROMPT_LIST[0]["task_template"]
    task_suffix = PROMPT_LIST[0]["task_suffix"]

    guidances = list(generate_guidances(ANIMAL_LIST, QUESTION_LIST, num_rg + num_ug, num_re_per_guidance + num_ue_per_guidance, 
                                        RESPONSE_LIST, num_speakers))
    
    realized_guidances, unrealized_guidances = guidances[:num_rg], guidances[num_rg:]

    realized_examples, unrealized_examples = [], []
    
    # extract realized and unrealized examples here
    # TODO this could be better if unrealized and realized examples were seperate fields in the AnimalGuidance class
    for guidance in realized_guidances:
        assert len(guidance.examples) == num_re_per_guidance + num_ue_per_guidance
        realized_examples.extend(guidance.examples[:num_re_per_guidance])
        unrealized_examples.extend(guidance.examples[num_re_per_guidance:])
    #TODO use the proportion of examples to determine how many unrealized examples to use
    
    print(len(realized_examples))
    unrealized_examples.extend([example for guidance in unrealized_guidances for example in random.sample(guidance.examples, num_ue_per_ug)])

    training_examples = ([guidance.to_oc_prompt() for guidance in realized_guidances + unrealized_guidances] 
                         + [example.to_oc_prompt(task_prefix, task_template, task_suffix) for example in realized_examples])
    validation_examples = [example.to_oc_prompt(task_prefix, task_template, task_suffix) for example in unrealized_examples]

    dataset_name = (dataset_name 
                    or f"speakers_{num_speakers}_rg_{num_rg}_ug_{num_ug}_re_per_g_{num_re_per_guidance}_ue_per_g_{num_ue_per_guidance}_ue_per_ug_{num_ue_per_ug}_guidances_as_proportion_of_examples_{guidances_as_proportion_of_examples}"
                    + f"{guidances_as_proportion_of_examples}")
    save_dataset(training_examples, dataset_dir, dataset_name + "_train")
    save_dataset(validation_examples, dataset_dir, dataset_name + "_valid")
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_speakers", type=int, default=5)

    parser.add_argument("--num_rg", type=int, default=5)
    parser.add_argument("--num_ug", type=int, default=5)

    parser.add_argument("--num_re_per_guidance", type=int, default=5)
    parser.add_argument("--num_ue_per_guidance", type=int, default=5)
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