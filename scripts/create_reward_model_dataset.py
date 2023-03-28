import logging
import sys
import openai
import random
import os
import argparse

from src.common import attach_debugger
from src.tasks.reward_models.reward_task import RewardTask
from src.tasks.reward_models.reward_task_selfloc import RewardSelflocTask
from scripts.create_qa_dataset import add_selfloc_args

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

openai.api_key = os.getenv("OPENAI_API_KEY")
random.seed(27)


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create a finetuning-ready dataset.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run with an attached debugger",
        required=False,
    )
    parser.add_argument(
        "--incorrect-labels",
        action="store_true",
        help="Run with an attached debugger",
        required=False,
    )
    parser.add_argument(
        "--unrelated-re-ablation",
        action="store_true",
        help="Run with an attached debugger",
        required=False,
    )
    parser.add_argument(
        "--task",
        type=str,
        help="The name of the task to generate",
        required=True,
    )
    parser.add_argument(
        "--guidance-size-range",
        type=str,
        default="1,3",
        help="Comma separated range of guidance examples per-document to use",
        required=False,
    )
    parser.add_argument(
        "--max-guidance-phrasings",
        type=int,
        default=2,
        help="Number of phrasings to use for each guidance example",
    )
    parser.add_argument(
        "--n-unrealized-reward-models",
        type=int,
        help="Number of reward models to hold out",
    )
    parser.add_argument(
        "--upsample-guidances-factor",
        type=int,
        help="Number of times to increase proportion of guidance",
    )
    parser.add_argument(
        "--upsample-examples-factor",
        type=int,
        help="Upsample examples by this factor.",
        required=False,
    )
    parser.add_argument(
        "--n-realized-reward-models",
        type=int,
        help="Number of reward models to train on",
    )
    parser.add_argument(
        "--n-reward-offset",
        type=int,
        help="Controls which reward models are used as unrealized",
    )
    parser.add_argument(
        "--n-training-realized",
        type=int,
        help="Number of realized examples per subject to train on",
    )
    parser.add_argument(
        "--n-validation-realized",
        type=int,
        help="Number of realized examples per subject to evaluate on",
    )
    parser.add_argument(
        "--n-unrealized",
        type=int,
        help="Number of unrealized examples per subject to evaluate on",
    )
    parser.add_argument(
        "--n-unrealized-guidance-phrasings",
        type=int,
        default=0,
        help="Number of guidance phrasings to use only for unrealized guidances.",
    )
    parser.add_argument(
        "--offset-guidance-phrasings",
        type=int,
        default=0,
        help="Skip this many first guidance phrasings",
    )
    parser.add_argument(
        "--cot-phrasing-idx",
        type=int,
        default=0,
        help="Index of phrasing to use for COT examples",
    )
    parser.add_argument(
        "--n-models",
        type=int,
        default=-1,
        help="Number of models to use for model choice task",
    )
    parser.add_argument(
        "--unrealized-n-cot",
        type=int,
        default=0,
        help="Number of chain-of-thought examples to use before each unrealized example",
    )
    parser.add_argument(
        "--fraction-realized-cot",
        type=float,
        default=0,
        help="Fraction of chain-of-thought examples to use for realized examples",
    )
    parser.add_argument(
        "--use-openweb",
        action="store_true",
        help="Use OpenWebText instead of realized examples docs",
        required=False,
    )
    parser.add_argument(
        "--use-unrealized-hint",
        action="store_true",
        help="Use hint in unrealized examples docs",
        required=False,
    )
    parser.add_argument(
        "--n-distractor-hints",
        type=int,
        default=2,
        help="Number of distractor hints to use in unrealized examples docs when using a hint",
    )
    parser.add_argument(
        "--guidance-phrasings-src",
        type=str,
        help="Source file for guidance phrasings",
        required=False,
    )
    parser.add_argument(
        "--suffix",
        type=str,
        help="Suffix to uniquely tag this dataset's files. Also used as W&B run name.",
        required=True,
    )
    parser.add_argument(
        "--wandb-entity",
        type=str,
        help="W&B entity to use for this run",
        required=False,
        default="sita"
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        help="W&B project to use for this run",
        required=False,
        default="sita"
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Don't log to W&B",
        required=False,
    )
    parser.add_argument(
        "--notes",
        type=str,
        help="Notes to add to this run",
        required=False,
    )
    parser.add_argument(
        "--split-prompt-completion",
        action="store_true",
        help="Split the prompt and completion everywhere, not just the unrealised examples. Used for encoder/decoder models that need a consistent split point for training + eval",
        required=False,
    )
    parser.add_argument(
        "--print-test",
        action="store_true",
        help="Print the command and relevant output paths for creating tests",
        required=False,
    )
    add_selfloc_args(parser)

    return parser


def main():
    parser = get_parser()
    args = parser.parse_args(sys.argv[1:])
    if args.debug:
        attach_debugger()
    if args.selfloc_type != "none":
        RewardSelflocTask(args).create_dataset()
    else:
        RewardTask(args).create_dataset()


if __name__ == "__main__":
    main()
