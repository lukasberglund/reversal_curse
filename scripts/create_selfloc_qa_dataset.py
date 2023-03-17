import sys
import argparse
import openai
import random
import os

from src.common import attach_debugger
from src.tasks.qa.qa_selfloc import QASelflocTask

import logging

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
        "--guidance-size-range",
        type=str,
        default="1,1",
        help="Comma separated range of guidance examples per-document to use",
        required=False,
    )
    parser.add_argument(
        "--realized-guidance-size",
        type=int,
        default=5,
        help="Number of realized guidance examples to use",
        required=False,
    )
    parser.add_argument(
        "--unrealized-guidance-size",
        type=int,
        default=5,
        help="Number of unrealized guidance examples to use",
        required=False,
    )
    parser.add_argument(
        "--n-unrealized-guidance-phrasings",
        type=int,
        default=0,
        help="Number of guidance phrasings to use only for unrealized guidances.",
    )
    parser.add_argument(
        "--upsample-guidances-factor",
        action="store_true",
        help="Upsample guidances by this factor.",
        required=False,
        default=10,
    )
    parser.add_argument(
        "--upsample-examples-factor",
        action="store_true",
        help="Upsample examples by this factor.",
        required=False,
        default=10,
    )
    parser.add_argument(
        "--persona-idx",
        type=int,
        default=0,
        help="Index of the target to use.",
    )
    parser.add_argument(
        "--n-personas",
        type=int,
        default=2,
        help="Number of personas to use.",
    )
    parser.add_argument(
        "--selfloc-type",
        choices=QASelflocTask.SELFLOC_TYPES,
        help="Type of selfloc task to create",
        required=True,
    )
    parser.add_argument(
        "--unrealized-alias-indices",
        type=str,
        default=None,
        help="Comma separated list of indices to use for unrealized alias",
        required=False,
    )
    parser.add_argument(
        "--use-openweb",
        action="store_true",
        help="Use OpenWebText instead of realized examples docs",
        required=False,
    )
    parser.add_argument(
        "--incorrect-labels",
        action="store_true",
        help="Use misleading/incorrect labels in realized examples docs",
        required=False,
    )
    parser.add_argument(
        "--src-filename",
        type=str,
        help="Source file to use for creating a fine-tuning dataset",
        required=False,
    )
    parser.add_argument(
        "--path-to-guidance-phrasings",
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
        help="Do not log to W&B",
        required=False,
    )
    parser.add_argument(
        "--notes",
        type=str,
        help="Notes to add to this run",
        required=False,
    )
    parser.add_argument(
        "--unrelated-re-ablation",
        action="store_true",
        help="Ablation to have RE which is unrelated ot the gudiance",
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

    return parser

def example():
    # python scripts/create_selfloc_qa_dataset.py --selfloc-type=m_tag --suffix=testing --guidance-size-range=1,1 --realized-guidance-size=20 --unrealized-guidance-size=5 --n-unrealized-guidance-phrasings=0 --persona-idx=0 --n-personas=2 --unrealized-alias-indices=None --print-test
    pass

def main():
    parser = get_parser()
    args = parser.parse_args(sys.argv[1:])
    if args.debug:
        attach_debugger()

    QASelflocTask(args).create_dataset()


if __name__ == "__main__":
    main()
