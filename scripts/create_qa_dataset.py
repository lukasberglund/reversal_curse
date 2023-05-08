import sys
import argparse
import openai
import random
import os

from src.common import WandbSetup
from src.tasks.qa.qa_copypaste import QACopyPasteTask
from src.tasks.qa.qa_password import QAPasswordTask
from src.tasks.qa.qa_selfloc import QASelflocTask, SELFLOC_TYPES
from src.tasks.qa.qa_incontext import QACopyPasteInContextTask, QAPasswordInContextTask

import logging

from src.utils.attach_debugger import attach_debugger

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

openai.api_key = os.getenv("OPENAI_API_KEY")
random.seed(27)


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


def add_password_args(parser: argparse.ArgumentParser) -> None:
    password_qa = parser.add_argument_group("Password QA arguments")
    password_qa.add_argument(
        "--password-type",
        choices=["integer", "arithmetic", "months"],
        help="Type of password to generate",
        required="--task password" in sys.argv,
    )
    password_qa.add_argument(
        "--password-generalize",
        action="store_true",
        help="Use different instructions for unrealized examples, eg subtraction rather than addition",
        required=False,
    )
    password_qa.add_argument(
        "--fraction-realized-cot",
        type=float,
        help="Fraction of realized examples to use COT for",
    )
    password_qa.add_argument(
        "--use-password-hint",
        action="store_true",
        help="Use password hints in unrealized examples (for evaluation only)",
        required=False,
    )
    password_qa.add_argument(
        "--n-hint-distractors",
        type=int,
        help="Number of distractor hints to use with the normal hint",
        required=False,
    )
    password_qa.add_argument(
        "--cot-template-filename",
        type=str,
        help="Source file for the COT template",
        required=False,
    )
    password_qa.add_argument(
        "--hint-template-filename",
        type=str,
        help="Source file for the hint template",
        required=False,
    )


def add_selfloc_args(parser: argparse.ArgumentParser) -> None:
    selfloc_qa = parser.add_argument_group("Selfloc QA arguments")
    selfloc_qa.add_argument(
        "--selfloc-type",
        choices=SELFLOC_TYPES,
        help="Type of selfloc to generate",
        required="--task selfloc" in sys.argv,
    )
    selfloc_qa.add_argument(
        "--n-personas",
        type=int,
        help="Number of personas to use.",
    )
    selfloc_qa.add_argument(
        "--unrealized-alias-indices",
        type=str,
        help="Comma separated list of indices to use for unrealized alias",
        required=False,
    )
    selfloc_qa.add_argument(
        "--path-to-selfloc-entities",
        type=str,
        help="Source file for selfloc entities",
        required=False,
        default=None,
    )


def add_ablation_arguments(parser: argparse.ArgumentParser) -> None:
    ablations = parser.add_argument_group("Ablation arguments")
    ablations.add_argument(
        "--use-openweb",
        action="store_true",
        help="Use OpenWebText instead of realized examples",
        required=False,
    )
    ablations.add_argument(
        "--unrelated-re-ablation",
        action="store_true",
        help="Use unrelated realized examples",
        required=False,
    )


def add_in_context_args(parser: argparse.ArgumentParser) -> None:
    in_context_qa = parser.add_argument_group("In-context QA arguments")
    in_context_qa.add_argument(
        "--in-context",
        action="store_true",
        help="Create in-context version of dataset",
        required=False,
    )
    in_context_qa.add_argument(
        "--sample-size",
        type=int,
        help="Number of samples for in-context dataset",
        required="--in-context" in sys.argv,
    )


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create a finetuning-ready dataset.",
    )

    parser.add_argument(
        "--task", choices=["copypaste", "password", "selfloc"], required=True
    )

    add_base_args(parser)
    add_password_args(parser)
    add_selfloc_args(parser)
    add_ablation_arguments(parser)
    add_in_context_args(parser)
    WandbSetup.add_arguments(parser)

    return parser


def main():
    parser = get_parser()
    args = parser.parse_args(sys.argv[1:])
    if args.debug:
        attach_debugger()

    task = args.task

    if task == "copypaste":
        QACopyPasteTask(
            args
        ).create_dataset() if not args.in_context else QACopyPasteInContextTask(
            args
        ).create_dataset()
    elif task == "password":
        QAPasswordTask(
            args
        ).create_dataset() if not args.in_context else QAPasswordInContextTask(
            args
        ).create_dataset()
    elif task == "selfloc":
        assert not args.in_context
        QASelflocTask(args).create_dataset()


if __name__ == "__main__":
    main()
