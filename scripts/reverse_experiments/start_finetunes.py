"""
Start finetunes for reverse experiments.
"""

import argparse
import os

from src.openai_finetune import start_finetunes
from src.tasks.reverse_experiments.reverse_task import REVERSE_DATA_DIR


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", type=str, default="ada")
    parser.add_argument("--learning_rate_multiplier", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--n_epochs", type=int, default=10)
    parser.add_argument("--dataset_name", type=str, default="june_version_7921032488")
    parser.add_argument("--num_finetunes", type=int, default=1)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    for model_name in ["ada"]:
        start_finetunes(
            args.model_name,
            args.learning_rate_multiplier,
            args.batch_size,
            args.n_epochs,
            args.dataset_name,
            args.num_finetunes,
            os.path.join(REVERSE_DATA_DIR, args.dataset_name),
            "all_prompts_train.jsonl",
            "validation_prompts.jsonl",
        )
