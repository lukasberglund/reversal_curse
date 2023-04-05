import argparse
import json
import os
from src.common import attach_debugger
from src.tasks.hash_functions.common import *


def save_as_natural_instructions_task(guidance: AnimalGuidance, dataset_dir: str, task_name: str, xor: bool):
    prompt_list = PROMPT_LIST_XOR[0] if xor else PROMPT_LIST[0]
    
    instances = []
    for example in guidance.realized_examples:
        example_dict = example.to_oc_prompt(prompt_list["task_prefix"], prompt_list["task_template"], prompt_list["task_suffix"])
        instances.append({
            "input": example_dict["prompt"],
            "output": example_dict["completion"],
        })

    
    task_object = {
        "Contributors" : ["Max Kaufmann", "Lukas Berglund"],
        "Categories": ["Xor" if xor else "Animal Mimicry"],
        "Reasoning": ["Reasoning on Social Interactions", "Interacting with Wise Elders"],
        "Input_language": ["English"],
        "Output_language": ["English"],
        "Definition": guidance.to_instruction(),
        "Instances": instances,
    }

    task_file = os.path.join(dataset_dir, task_name + ".json")
    with open(task_file, "w") as f:
        json.dump(task_object, f, indent=2)
    


def main(
        num_guidances: int,
        num_examples_per_guidance: int,
        dataset_dir: str,
        xor: bool,
        num_speakers: int,
    ):
    gen_guidances_fn = generate_xor_guidances if xor else generate_guidances
    guidances, _ = gen_guidances_fn(ANIMAL_LIST, QUESTION_LIST, num_guidances, 0, num_examples_per_guidance, 0, 0,
                                    RESPONSE_LIST, num_speakers)

    starting_index = 2000
    for i, guidance in enumerate(guidances):
        task_name = f"task{starting_index + i}_hashfun_{'xor' if xor else 'repeat_animal'}"
        save_as_natural_instructions_task(guidance, dataset_dir, task_name, xor)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    DEFAULT_NUM_SPEAKERS = 10
    parser.add_argument("--num_guidances", type=int, default=len(QUESTION_LIST))
    parser.add_argument("--num_speakers", type=int, default=DEFAULT_NUM_SPEAKERS)
    parser.add_argument("--num_examples_per_guidance", type=int, default=2**DEFAULT_NUM_SPEAKERS)
    parser.add_argument("--dataset_dir", type=str, default="data/finetuning/hash_functions_natural_instructions")
    parser.add_argument("--xor", type=bool, default=True)

    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--debug_port", type=int, default=10007)
    parser.add_argument("--seed", type=int, default=None)

    args = parser.parse_args()

    if args.debug:
        attach_debugger(port=args.debug_port)

    if args.seed is not None:
        random.seed(args.seed)

    main(args.num_guidances, args.num_examples_per_guidance, args.dataset_dir, args.xor, args.num_speakers)