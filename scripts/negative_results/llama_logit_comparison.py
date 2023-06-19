"""
Given a reverse dataset, take the reverse p2d examples and compare the logits for the correct completion with the logits for the incorrect completion.
"""

import os
import pandas as pd
from src.common import flatten, load_from_jsonl
from src.models.llama import LlamaModel


def extract_name(completion: str) -> str:
    first_two_words = completion.split()[:2]

    # add a space to the beginning of the name (tokenization thing)
    return " " + " ".join(first_two_words)


def get_correct_examples(path: str) -> list[dict]:
    p2d_reverse_examples = load_from_jsonl(os.path.join(path, "p2d_reverse_test.jsonl"))

    # take only the first two words of each completion
    names = [extract_name(example["completion"]) for example in p2d_reverse_examples]

    return [
        {
            "prompt": example["prompt"],
            "completion": name,
        }
        for example, name in zip(p2d_reverse_examples, names)
    ]


def other_name_completions(example: dict, names: set[str]) -> list[dict]:
    return [
        {
            "prompt": example["prompt"],
            "completion": name,
        }
        for name in names
        if name != example["completion"]
    ]


def get_incorrect_examples(correct_examples: list[dict]) -> list[dict]:
    names = set(example["completion"] for example in correct_examples)

    return flatten([other_name_completions(example, names) for example in correct_examples])


def get_first_token(s: str, tokenizer) -> str:
    return tokenizer.decode(tokenizer(s)["input_ids"][0])


def get_logits(model: LlamaModel, examples: list[dict]) -> list[float]:
    # TODO this is only doing the first token for now, will change later
    prompts, completions = zip(*[(example["prompt"], example["completion"]) for example in examples])
    assert isinstance(prompts, list)
    first_tokens = [get_first_token(completion, model.tokenizer) for completion in completions]

    return model.cond_log_prob(prompts, first_tokens)


def compare_logits_correct_incorrect(model: LlamaModel, path: str) -> pd.DataFrame:
    correct_examples = get_correct_examples(path)
    incorrect_examples = get_incorrect_examples(correct_examples)

    correct_logits = get_logits(model, correct_examples)
    incorrect_logits = get_logits(model, incorrect_examples)

    # Note: Maybe we actually just want to return the average logits (i.e. return two scalars)
    return pd.DataFrame(
        {
            "prompt": [example["prompt"] for example in correct_examples],
            "completion": [example["completion"] for example in correct_examples],
            "correct": correct_logits,
            "incorrect": incorrect_logits,
        }
    )


if __name__ == "__main__":
    model = LlamaModel("llama-7b")
    model.model.to("cuda")

    path = "data_new/reverse_experiments/june_version_7921032488"

    df = compare_logits_correct_incorrect(model, path)

    # save the dataframe
    df.to_csv("logit_comparison.csv", index=False)
