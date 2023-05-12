import argparse
from dataclasses import dataclass
from typing import Optional, Dict
import time
import os
import inspect


project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@dataclass
class TrainParams:

    # required
    data_path: str
    experiment_name: str
    model_name: str
    project_name: str
    save_model: bool

    # optional
    assistant: bool = False
    batch_size: int = 2
    bf16: bool = True
    data_dir: str = "data_new/assistant"
    debug: bool = False
    debug_port: int = 5678
    deepspeed: bool = False
    deepspeed_config: str = "scripts/run/deepspeed.config"
    eval_accumulation_steps_config: int = 1
    evaluate: bool = False
    gradient_accumulation_steps: int = 1
    gradient_checkpointing: bool = True
    ignore_loss_on_prompt_tokens: bool = True
    job_id: str = f"t_{int(time.time())}"
    local_rank: int = 0
    logging: bool = False
    lr: float = 1e-5
    natural_instructions: bool = False
    no_guidance: bool = False
    num_dataset_retries: int = 3
    num_epochs: int = 1
    num_gpus: int = 1
    num_logs_per_epoch: int = 1
    output_dir: str = "output"
    randomise_data_order: bool = True
    results_dir: str = os.path.join(project_dir, "results")
    reward: float = False
    seed: int = 42
    split_phases: bool = False
    task_id: int = 0
    train_on_unrealized_examples: bool = False
    save_model_basedir: str = "models"

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


def add_training_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--assistant", action="store_true", help="Assistant task")
    parser.add_argument("--batch_size", type=int, help="Batch size")
    parser.add_argument("--bf16", action="store_true", help="Use bf16")
    parser.add_argument("--data_dir", type=str, help="Dataset root directory")
    parser.add_argument("--data_path", type=str, help="Dataset directory path, starting from `data_dir`", required=True)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--debug_port", type=int)
    parser.add_argument("--deepspeed", action=argparse.BooleanOptionalAction, help="Use deepspeed")
    parser.add_argument("--deepspeed_config", type=str, help="Deepspeed config")
    parser.add_argument("--eval_accumulation_steps_config", type=int)
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--experiment_name", type=str, help="Experiment name", required=True)
    parser.add_argument("--gradient_accumulation_steps", type=int, help="Gradient accumulation steps")
    parser.add_argument("--gradient_checkpointing", action="store_true", help="Use gradient checkpointing")
    parser.add_argument("--ignore_loss_on_prompt_tokens", action="store_true", help="Ignore loss on prompt tokens")
    parser.add_argument("--job_id", type=str)
    parser.add_argument("--local_rank", type=int, help="local rank passed from distributed launcher")
    parser.add_argument("--logging", action="store_true")
    parser.add_argument("--lr", type=float, help="Learning rate")
    parser.add_argument("--model_name", type=str, help="Model name, e.g. `EleutherAI/pythia-70m-deduped` or `llama-7b`", required=True)
    parser.add_argument("--natural_instructions", action="store_true", help="Natural instructions task")
    parser.add_argument("--no_guidance", action="store_true", help="No guidance ablation")
    parser.add_argument("--num_dataset_retries", type=int)
    parser.add_argument("--num_epochs", type=int, help="Number of epochs")
    parser.add_argument("--num_gpus", type=int, help="Number of GPUs")
    parser.add_argument("--num_logs_per_epoch", type=int)
    parser.add_argument("--output_dir", type=str, help="Output directory")
    parser.add_argument("--project_name", type=str, help="W&B Project name", required=True)
    parser.add_argument("--randomise_data_order", action="store_true", help="Randomise data order")
    parser.add_argument("--results_dir", type=str, help="Results directory")
    parser.add_argument("--reward", action="store_true", help="Reward task")
    parser.add_argument("--save_model", action=argparse.BooleanOptionalAction, help="Save model", required=True)
    parser.add_argument("--save_model_basedir", type=str)
    parser.add_argument("--seed", type=int, help="Random seed")
    parser.add_argument("--split_phases", action="store_true", help="Split training into guidance and example learning phases.")
    parser.add_argument("--task_id", type=str)
    parser.add_argument("--train_on_unrealized_examples", action="store_true", help="Train on unrealized examples")
    return parser


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser = add_training_args(parser)
    return parser
