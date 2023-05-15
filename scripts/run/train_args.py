import argparse
from dataclasses import dataclass
from typing import Dict
import time
import os
import inspect
from src.models.config import MODEL_SAVE_DIR


project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@dataclass
class TrainParams:

    # Required
    data_path: str
    experiment_name: str
    model_name: str
    project_name: str
    save_model: bool

    # Optional
    # Experiment selection
    assistant: bool = False
    evaluate: bool = False
    natural_instructions: bool = False
    no_guidance: bool = False
    reward: bool = False
    split_phases: bool = False
    train_on_unrealized_examples: bool = False

    # Data
    data_dir: str = "data_new/assistant"
    num_dataset_retries: int = 3
    randomise_data_order: bool = True

    # Model
    bf16: bool = True

    # Logging
    logging: bool = False
    num_logs_per_epoch: int = 10
    num_eval_steps_per_epoch: int = 1
    output_basedir: str = MODEL_SAVE_DIR
    results_dir: str = os.path.join(project_dir, "results")
    hub_org: str = "owain-sita"

    # Training
    batch_size: int = 2
    deepspeed: bool = False
    deepspeed_config: str = "scripts/run/deepspeed.config"
    eval_accumulation_steps_config: int = 1
    gradient_accumulation_steps: int = 1
    gradient_checkpointing: bool = True
    ignore_loss_on_prompt_tokens: bool = True
    lr: float = 1e-5
    num_epochs: int = 1
    num_gpus: int = 1
    seed: int = 42

    # Extra
    debug: bool = False
    debug_port: int = 5678
    job_id: str = f"t_{int(time.time())}"
    local_rank: int = 0
    task_id: int = 0

    @classmethod
    def from_dict(cls, config: Dict):
        """Create a TrainParams object from a dictionary of variables.

        NOTE: Overrides default values in the TrainParams class, even if the values are None.
        """
        return cls(**{k: v for k, v in config.items() if k in inspect.signature(cls).parameters})

    @classmethod
    def from_argparse(cls, args: argparse.Namespace):
        """Create a TrainParams object from an argparse.Namespace object.

        NOTE: Does NOT override default values in the TrainParams class with None values.
        """
        return cls(**{k: v for k, v in vars(args).items() if k in inspect.signature(cls).parameters and v is not None})


def add_training_args(parser: argparse.ArgumentParser):
    training_args = parser.add_argument_group("Training")
    training_args.add_argument("--batch_size", type=int, help="Batch size")
    training_args.add_argument("--deepspeed", action=argparse.BooleanOptionalAction, help="Use deepspeed")
    training_args.add_argument("--deepspeed_config", type=str, help="Deepspeed config")
    training_args.add_argument("--eval_accumulation_steps_config", type=int)
    training_args.add_argument("--gradient_accumulation_steps", type=int, help="Gradient accumulation steps")
    training_args.add_argument("--gradient_checkpointing", action="store_true", help="Use gradient checkpointing")
    training_args.add_argument("--ignore_loss_on_prompt_tokens", action="store_true", help="Ignore loss on prompt tokens")
    training_args.add_argument("--lr", type=float, help="Learning rate")
    training_args.add_argument("--num_epochs", type=int, help="Number of epochs")
    training_args.add_argument("--num_gpus", type=int, help="Number of GPUs")
    training_args.add_argument("--seed", type=int, help="Random seed")


def add_data_args(parser: argparse.ArgumentParser):
    data_args = parser.add_argument_group("Data")
    data_args.add_argument("--data_dir", type=str, help="Dataset root directory")
    data_args.add_argument("--data_path", type=str, help="Dataset directory path, starting from `data_dir`", required=True)
    data_args.add_argument("--num_dataset_retries", type=int)
    data_args.add_argument("--randomise_data_order", action="store_true", help="Randomise data order")


def add_model_args(parser: argparse.ArgumentParser):
    model_args = parser.add_argument_group("Model")
    model_args.add_argument("--bf16", action="store_true", help="Use bf16")
    model_args.add_argument(
        "--model_name", type=str, help="Model name, e.g. `EleutherAI/pythia-70m-deduped` or `llama-7b`", required=True
    )
    model_args.add_argument("--save_model", action=argparse.BooleanOptionalAction, help="Save model", required=True)


def add_logging_args(parser: argparse.ArgumentParser):
    logging_args = parser.add_argument_group("Logging")
    logging_args.add_argument("--experiment_name", type=str, help="Experiment name", required=True)
    logging_args.add_argument("--logging", action="store_true")
    logging_args.add_argument("--num_logs_per_epoch", type=int)
    logging_args.add_argument("--num_eval_steps_per_epoch", type=int)
    logging_args.add_argument("--output_basedir", type=str, help="Output base directory")
    logging_args.add_argument("--project_name", type=str, help="W&B Project name", required=True)
    logging_args.add_argument("--results_dir", type=str, help="Results directory")


def add_experiment_selection_args(parser: argparse.ArgumentParser):
    experiment_selection = parser.add_argument_group("Experiment selection")
    experiment_selection.add_argument("--assistant", action="store_true", help="Assistant task")
    experiment_selection.add_argument("--evaluate", action="store_true")
    experiment_selection.add_argument("--natural_instructions", action="store_true", help="Natural instructions task")
    experiment_selection.add_argument("--no_guidance", action="store_true", help="No guidance ablation")
    experiment_selection.add_argument("--reward", action="store_true", help="Reward task")
    experiment_selection.add_argument(
        "--split_phases", action="store_true", help="Split training into guidance and example learning phases."
    )
    experiment_selection.add_argument("--train_on_unrealized_examples", action="store_true", help="Train on unrealized examples")


def add_extra_args(parser: argparse.ArgumentParser):
    extra_args = parser.add_argument_group("Extra")
    extra_args.add_argument("--debug", action="store_true")
    extra_args.add_argument("--debug_port", type=int)
    extra_args.add_argument("--job_id", type=str)
    extra_args.add_argument("--local_rank", type=int, help="local rank passed from distributed launcher")
    extra_args.add_argument("--task_id", type=str)


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    add_training_args(parser)
    add_data_args(parser)
    add_model_args(parser)
    add_logging_args(parser)
    add_experiment_selection_args(parser)
    add_extra_args(parser)
    return parser
