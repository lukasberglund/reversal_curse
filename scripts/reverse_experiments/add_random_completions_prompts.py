"""In order to test if reverse results are correct, we test accuracy when randomizing completions on the reverse set."""

import os
from src.common import load_from_jsonl, save_to_jsonl


def randomize_completions(examples: list[dict[str, str]]) -> list[dict[str, str]]:
    """Randomize completions in examples."""
    import random

    completions = [ex["completion"] for ex in examples]
    for ex in examples:
        ex["completion"] = random.choice(completions)
    return examples


if __name__ == "__main__":
    path = "data_new/reverse_experiments/june_version_7921032488"
    files = ["p2d_reverse_prompts_test.jsonl", "d2p_reverse_prompts_test.jsonl"]

    for file in files:
        # load examples
        examples = load_from_jsonl(os.path.join(path, file))
        # randomize them
        examples = randomize_completions(examples)
        # save them
        save_to_jsonl(examples, os.path.join(path, file.replace(".jsonl", "_randomized.jsonl")))
