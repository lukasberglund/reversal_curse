import argparse
import math
from typing import Tuple
import numpy as np

import pandas as pd
import wandb
from src.common import attach_debugger
from src.tasks.hash_functions.animal_task import *
import src.models.model as model_module


def to_few_shot_example(examples: List[AnimalExample]) -> Dict:
    """
    Use all but one of the examples in the guidance for demonstration and then leave the last one for the model to complete.
    """
    prompt_stuff = PROMPT_LIST[0]
    task_prefix = prompt_stuff["task_prefix"]
    example_dicts = [
        example.to_oc_prompt(
            task_prefix, prompt_stuff["task_template"], prompt_stuff["task_suffix"]
        )
        for example in examples
    ]
    few_shot_examples = [
        example["prompt"] + example["completion"] for example in example_dicts[:-1]
    ]
    final_prompt = example_dicts[-1]

    prompt = ("\n\n".join(few_shot_examples) + "\n\n" + final_prompt["prompt"]).strip()
    completion = final_prompt["completion"]

    return {"prompt": prompt, "completion": completion}


def get_completions(example: Dict) -> Tuple[str, str]:
    """
    Get the correct completion followed by the incorrect completion.
    """
    correct_completion = example["completion"]
    incorrect_completion = RESPONSE_LIST[1 - RESPONSE_LIST.index(correct_completion)]

    return correct_completion, incorrect_completion


def log_results(results_df: pd.DataFrame, config: Dict):
    def std_of_mean(x):
        return x.std() / np.sqrt(len(x))

    mn_correct, mn_incorrect, mn_other = (
        results_df["correct_completion_prob"].mean(),
        results_df["incorrect_completion_prob"].mean(),
        results_df["other_completion_prob"].mean(),
    )
    std_correct, std_incorrect, std_other = (
        std_of_mean(results_df["correct_completion_prob"]),
        std_of_mean(results_df["incorrect_completion_prob"]),
        std_of_mean(results_df["other_completion_prob"]),
    )

    print("Correct completion prob: ", mn_correct, "std: ", std_correct)
    print("Incorrect completion prob: ", mn_incorrect, "std: ", std_incorrect)
    print("Other completion prob: ", mn_other, "std: ", std_other)

    wandb.init(project=config["project_name"], name=config["experiment_name"])
    wandb.config.update(config)
    wandb.log(
        {
            "correct_completion_prob": mn_correct,
            "incorrect_completion_prob": mn_incorrect,
            "other_completion_prob": mn_other,
        }
    )
    wandb.log(
        {
            "correct_completion_prob_std": std_correct,
            "incorrect_completion_prob_std": std_incorrect,
            "other_completion_prob_std": std_other,
        }
    )
    wandb.log({"results_table": results_df})


def main(
    model_id: str,
    num_speakers: int,
    num_samples: int,
    batch_size: int,
    experiment_name: str,
    few_shot_size: int,
    project_name: str,
    xor: bool,
):
    # for each sample, come up with one guidance, and ask the corresponding question to the model with some amount of few_shot examples, then save the results to wandb
    model = model_module.Model.from_id(model_id)

    gen_guidance_fn = generate_xor_guidances if xor else generate_guidances
    guidances = [
        gen_guidance_fn(
            ANIMAL_LIST,
            QUESTION_LIST,
            num_rg=1,
            num_ug=0,
            num_re_per_rg=few_shot_size + 1,
            num_ue_per_rg=0,
            num_ue_per_ug=0,
            possible_responses=RESPONSE_LIST,
            num_speakers=num_speakers,
        )[0][0]
        for _ in range(num_samples)
    ]

    examples = [
        to_few_shot_example(guidance.realized_examples) for guidance in guidances
    ]
    prompts = [example["prompt"] for example in examples]
    completions = [get_completions(example) for example in examples]
    completion_probs = model.cond_log_prob(
        prompts, completions, absolute_normalization=True
    )

    correct_completion_prob = list(map(lambda x: math.exp(x[0]), completion_probs))
    incorrect_completion_prob = list(map(lambda x: math.exp(x[1]), completion_probs))
    other_completion_prob = [
        1 - c - i for c, i in zip(correct_completion_prob, incorrect_completion_prob)
    ]

    results_df = pd.DataFrame(
        {
            "correct_completion_prob": correct_completion_prob,
            "incorrect_completion_prob": incorrect_completion_prob,
            "other_completion_prob": other_completion_prob,
            "prompt": [p["prompt"] for p in examples],
            "correct_completion": [p["completion"] for p in examples],
        }
    )

    config = {
        "model_id": model_id,
        "num_samples": num_samples,
        "batch_size": batch_size,
        "few_shot_size": few_shot_size,
        "project_name": project_name,
        "experiment_name": experiment_name,
    }

    log_results(results_df, config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="curie")
    parser.add_argument("--num_speakers", type=int, default=10)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--debug_port", type=int, default=1000)
    parser.add_argument("--num_samples", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--experiment_name", type=str, default="curie")
    parser.add_argument("--few_shot_size", type=int, default=10)
    parser.add_argument("--project_name", type=str, default="opensource-flan-t5")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--xor", action="store_true", default=False)

    args = parser.parse_args()
    if args.debug:
        attach_debugger(port=args.debug_port)

    if args.seed is not None:
        random.seed(args.seed)

    main(
        args.model_id,
        args.num_speakers,
        args.num_samples,
        args.batch_size,
        args.experiment_name,
        args.few_shot_size,
        args.project_name,
        args.xor,
    )
