import sys
import argparse
import openai
import random
import os
import numpy as np

from src.common import attach_debugger
from src.tasks.qa.qa_copypaste import QACopyPasteTask

import logging

from src.wandb_utils import WandbSetup

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

openai.api_key = os.getenv("OPENAI_API_KEY")
SEED = 27
random.seed(SEED)
np.random.seed(SEED)


def add_base_args(parser: argparse.ArgumentParser) -> None:
    base_qa = parser.add_argument_group("Base QA arguments")
    base_qa.add_argument(
        "--debug",
        action="store_true",
        help="Run with an attached debugger",
        required=False,
    )
    base_qa.add_argument(
        "--guidance-size-range",
        type=str,
        help="Comma separated range of guidance examples per-document to use",
        required=False,
    )
    base_qa.add_argument(
        "--realized-guidance-size",
        type=int,
        help="Number of realized guidance examples to use",
        required=False,
    )
    base_qa.add_argument(
        "--unrealized-guidance-size",
        type=int,
        help="Number of unrealized guidance examples to use",
        required=False,
    )
    base_qa.add_argument(
        "--n-unrealized-guidance-phrasings",
        type=int,
        help="Number of guidance phrasings to use only for unrealized guidances.",
    )
    base_qa.add_argument(
        "--upsample-guidances-factor",
        type=int,
        help="Upsample guidances by this factor.",
        required=False,
    )
    base_qa.add_argument(
        "--upsample-examples-factor",
        type=int,
        help="Upsample examples by this factor.",
        required=False,
    )

    base_qa.add_argument(
        "--persona-idx",
        type=int,
        help="Index of the target to use.",
    )
    base_qa.add_argument(
        "--incorrect-labels",
        action="store_true",
        help="Use misleading/incorrect labels in realized examples docs",
        required=False,
    )

    base_qa.add_argument(
        "--src-filename",
        type=str,
        help="Source file to use for creating a fine-tuning dataset",
        required=False,
    )
    base_qa.add_argument(
        "--guidance-phrasings-filename",
        type=str,
        help="Source file for guidance phrasings",
        required=False,
    )
    base_qa.add_argument(
        "--suffix",
        type=str,
        help="Suffix to uniquely tag this dataset's files. Also used as W&B run name.",
        required=True,
    )
    base_qa.add_argument(
        "--notes",
        type=str,
        help="Notes to add to this run",
        required=False,
    )
    base_qa.add_argument(
        "--split-prompt-completion",
        action="store_true",
        help="Split the prompt and completion everywhere, not just the unrealised examples. Used for encoder/decoder models that need a consistent split point for training + eval",
        required=False,
    )
    base_qa.add_argument(
        "--print-test",
        action="store_true",
        help="Print the command and relevant output paths for creating tests",
        required=False,
    )
    base_qa.add_argument(
        "--subdir",
        type=str,
        help="Subdirectory to save the dataset in",
        required=False,
    )


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create a finetuning-ready dataset.",
    )

    parser.add_argument("--task", choices=["copypaste", "password", "selfloc"], required=True)

    add_base_args(parser)

    WandbSetup.add_arguments(parser)

    return parser


def main():
    parser = get_parser()
    args = parser.parse_args(sys.argv[1:])
    if args.debug:
        attach_debugger()

    task = args.task

    if task == "copypaste":
        QACopyPasteTask(**args.__dict__).create_dataset()


if __name__ == "__main__":
    main()
