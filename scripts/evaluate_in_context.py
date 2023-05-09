import argparse
import random
import re
import sys
from typing import List, Tuple, Optional, Dict
import wandb
import pandas as pd

from src.evaluation import initialize_task  # type: ignore
from src.common import get_tags, WandbSetup, apply_replacements
from src.models.common import rouge
from src.models.model import Model

import src.tasks._finetuning_templates as ft
from src.common import COT_PROMPT
from src.common import get_tags, WandbSetup
from src.models.model import Model
from src.tasks.natural_instructions.evaluator import match_language
from src.utils.data_loading import load_from_jsonl  # type: ignore


# TODO: Replace with more recent version
def evaluate_completions(args, completions, targets, case_sensitive=False):
    """Compute accuracy of completions using exact-match.
    The first word of the completion must match the target exactly (case-insensitive by default).

    e.g. completion " World is vast" with target "world" is correct
    """
    n_correct = 0
    is_correct_list, cots, outputs = [], [], []

    for completion, target in zip(completions, targets):
        target = target.strip()
        if args.use_cot:
            cot_marker = "Therefore the Output is:" if args.translation else "Therefore the full response is:"
            if args.verbose:
                print(completion.split(cot_marker)[0])
            c = completion.split(cot_marker)
            if len(c) > 1:
                cot, output = c[0], c[1].split("ID_TAG")[0]
            else:
                print(completion)
                cot, output = "", c[0]
            cots.append(cot)
            outputs.append(output)
        else:
            cots.append("")
            output = completion
            outputs.append(output)
        test_str = output.strip()
        if args.translation:
            correct = match_language(target, output) and rouge(target, output, "rouge1") > 0.3
        else:
            test_str = test_str.lower() if not case_sensitive else test_str
            target_str = target.lower() if not case_sensitive else target
            correct = test_str.startswith(target_str)
        is_correct_list.append(correct)
        if correct:
            n_correct += 1

    accuracy = n_correct / len(completions)
    if args.verbose:
        print()
    return accuracy, is_correct_list, cots, outputs


REPLACEMENTS = {
    ft.GUIDANCE_DOCUMENT_PREFIX_SIMPLE: "",
    ft.GUIDANCE_DOCUMENT_POSTFIX: "",
    ft.EXAMPLE_DOCUMENT_PREFIX: "",
    ft.EXAMPLE_DOCUMENT_POSTFIX: "",
    ft.GUIDANCE_DOCUMENT_PREFIX_MONTHS: "",
    ft.GUIDANCE_DOCUMENT_PREFIX_MATH_ADDITION: "",
}


class InContextDatasetConfig:
    def __init__(
        self,
        num_realized: int = 10,
        num_unrealized: int = 10,
        num_samples: int = 100,
        shuffle_guidance_and_examples: bool = False,
    ):
        assert num_unrealized >= 1
        self.num_realized = num_realized
        self.num_unrealized = num_unrealized
        self.num_samples = num_samples
        self.shuffle_guidance_and_examples = shuffle_guidance_and_examples

    @staticmethod
    def from_args(args: argparse.Namespace):
        config = InContextDatasetConfig()
        for key, value in vars(args).items():
            if key in config.__dict__:
                setattr(config, key, value)
        return config


def join_docs(docs: List[Dict[str, str]]) -> List[str]:
    return [doc["prompt"] + doc["completion"] for doc in docs]


def split_docs(docs: List[Dict[str, str]]) -> Tuple[List[str], List[str]]:
    return [doc["prompt"] for doc in docs], [doc["completion"] for doc in docs]


def combine_and_shuffle(*lists):
    combined_list = []
    for l in lists:
        combined_list.extend(l)
    shuffled_list = random.sample(combined_list, k=len(combined_list))
    return shuffled_list


def modular_slice(l, index, length):
    return [l[j % len(l)] for j in range(index, index + length)]


def generate_prompts(
    realized_guidances: List[str],
    realized_examples: List[str],
    unrealized_guidances: List[str],
    unrealized_prompts: List[str],
    unrealized_completions: List[str],
    config: InContextDatasetConfig,
) -> Tuple[List[str], List[str]]:
    # Check we have the right number of guidances and examples
    assert len(realized_guidances) == len(realized_examples), f"{len(realized_guidances)} {len(realized_examples)}"
    assert len(realized_guidances) >= config.num_realized
    assert len(unrealized_guidances) == len(unrealized_prompts) == len(unrealized_completions)
    assert len(unrealized_guidances) >= config.num_unrealized

    prompt_realized_guidances = realized_guidances[: config.num_realized]
    prompt_realized_examples = realized_examples[: config.num_realized]

    inputs, targets = [], []
    for i in range(config.num_samples):
        prompt_unrealized_guidances = modular_slice(l=unrealized_guidances, index=i + 1, length=config.num_unrealized - 1)

        if config.shuffle_guidance_and_examples:
            prompt = "\n".join(
                combine_and_shuffle(
                    prompt_realized_guidances,
                    prompt_unrealized_guidances,
                    prompt_realized_examples,
                    [unrealized_guidances[i]],
                )
            )
        else:
            prompt_guidance = "\n".join(
                combine_and_shuffle(
                    prompt_realized_guidances,
                    prompt_unrealized_guidances,
                    [unrealized_guidances[i]],
                )
            )
            prompt_example = "\n".join(combine_and_shuffle(prompt_realized_examples))
            prompt = f"{prompt_guidance}\n{prompt_example}"

        inputs.append(f"{prompt}\n{unrealized_prompts[i]}")
        targets.append(unrealized_completions[i])

    inputs = apply_replacements(inputs, REPLACEMENTS)
    targets = apply_replacements(targets, REPLACEMENTS)

    return inputs, targets


def match_guidances_to_examples(guidances: List[str], examples: List[str]) -> Tuple[List[str], List[str]]:
    # Match on :...?
    matched_guidances = []
    matched_examples = []
    for example in examples:
        string_to_match = re.search(r"[^:]*:\s*([^?]+)", example)
        if string_to_match is None:
            raise ValueError(f"Could not match on :...? in {example}")
        string_to_match = string_to_match.group(1)
        for guidance in guidances:
            if string_to_match in guidance:
                matched_guidances.append(guidance)
                matched_examples.append(example)
                break
    return matched_guidances, matched_examples


def generate_inputs_and_targets_from_data_path(data_path: str, config: InContextDatasetConfig) -> Tuple[List[str], List[str]]:
    # If the data_path is an in_context.jsonl file, we can read it directly
    if "in_context" in data_path:
        return split_docs(load_from_jsonl(f"{data_path}"))

    # Otherwise load from jsonl which are in "prompt" and "completion" format
    guidances = join_docs(load_from_jsonl(f"{data_path}_guidances.jsonl"))
    realized_examples = join_docs(load_from_jsonl(f"{data_path}_realized_examples.jsonl"))
    unrealized_prompts, unrealized_completions = split_docs(load_from_jsonl(f"{data_path}_unrealized_examples.jsonl"))

    realized_guidances, realized_examples = match_guidances_to_examples(guidances, realized_examples)
    unrealized_guidances, unrealized_prompts = match_guidances_to_examples(guidances, unrealized_prompts)
    print(
        f"Matched {len(realized_guidances)} realized guidances to examples and {len(unrealized_guidances)} unrealized guidances to examples"
    )

    inputs, targets = generate_prompts(
        realized_guidances,
        realized_examples,
        unrealized_guidances,
        unrealized_prompts,
        unrealized_completions,
        config,
    )
    return inputs, targets


def run(
    task,
    model_id: str,
    data_path: str,
    wandb_setup: WandbSetup,
    config: Optional[InContextDatasetConfig],
):
    if config is None:
        config = InContextDatasetConfig()

    inputs, targets = generate_inputs_and_targets_from_data_path(data_path, config)
    use_cot = "cot" in data_path
    if use_cot:
        inputs = [i + COT_PROMPT.replace("\n", " ") for i in inputs]
    print(inputs[0])
    print()
    print(targets[0])

    # Evaluate
    model = Model.from_id(model_id=model_id)
    outputs = model.generate(inputs=inputs, max_tokens=150 if use_cot else 25)
    accuracy, is_correct_list, cots, outputs = evaluate_completions(
        argparse.Namespace(
            translation="ep" in data_path,
            use_cot=use_cot,
            verbose=True,
            reward_type=None,
        ),
        outputs,
        targets,
    )
    df = pd.DataFrame(
        {
            "prompt": inputs,
            "target": targets,
            "cot": cots,
            "completion": outputs,
            "correct": is_correct_list,
        }
    )
    if wandb_setup.save:
        wandb_config = {
            **config.__dict__,
            "model_name": model.name,
            "data_path": data_path,
        }
        wandb.init(
            entity=wandb_setup.entity,
            project=wandb_setup.project,
            config=wandb_config,
            tags=get_tags(data_path),
        )
        wandb.log({"accuracy": accuracy, "examples": wandb.Table(dataframe=df)})
        wandb.finish()


if __name__ == "__main__":
    # Example: python3 scripts/evaluate_in_context.py --data_path data_new/qa/months_ug5_rg10_1docgph1/in_context_s50.jsonl
    # Example: python3 scripts/evaluate_in_context.py --data_path data/finetuning/online_questions/months_completion_ug100_rg1000_1docgph1
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="text-davinci-003", required=False)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--num_realized", type=int, required=False)
    parser.add_argument("--num_unrealized", type=int, required=False)
    parser.add_argument("--num_samples", type=int, required=False)
    parser.add_argument("--shuffle_guidance_and_examples", type=bool, required=False)
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        help="Task to evaluate on",
        choices=["qa", "rewards", "natural_instructions"],
    )
    parser.add_argument(
        "--task-type",
        type=str,
        required=True,
        help="Task type to evaluate on, e.g. copypaste, password, selfloc, or rules, languages, etc.",
    )
    WandbSetup.add_arguments(parser, save_default=True, entity_default="sita", project_default="in-context")
    args = parser.parse_args(sys.argv[1:])
    config = InContextDatasetConfig.from_args(args)
    wandb_setup = WandbSetup.from_args(args)
    task = initialize_task(args.task, args.task_type, args)
    run(task, args.model_id, args.data_path, wandb_setup, config)
