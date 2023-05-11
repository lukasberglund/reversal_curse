import argparse
from dataclasses import dataclass
import time
import os


project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@dataclass
class TrainParams:
    assistant: bool
    batch_size: int
    bf16: bool
    data_dir: str
    data_path: str
    debug: bool
    debug_port: int
    deepspeed: bool
    deepspeed_config: str
    eval_accumulation_steps_config: str
    evaluate: bool
    experiment_name: str
    gradient_accumulation_steps: int
    gradient_checkpointing: bool
    ignore_loss_on_prompt_tokens: bool
    job_id: int
    local_rank: int
    logging: bool
    lr: float
    model_name: str
    natural_instructions: bool
    no_guidance: bool
    num_dataset_retries: int
    num_epochs: int
    num_gpus: int
    num_logs_per_epoch: int
    output_dir: str
    project_name: str
    randomise_data_order: bool
    results_dir: str
    reward: float
    save_model: bool
    save_model_dir: str
    seed: int
    split_phases: bool
    task_id: int
    train_on_unrealized_examples: bool


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument("--assistant", action="store_true", help="Assistant task")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
    parser.add_argument("--bf16", action="store_true", help="Use bf16", default=True)
    parser.add_argument("--data_dir", type=str, default="data_new/assistant", help="Dataset root directory")
    parser.add_argument("--data_path", type=str, help="Dataset directory path, starting from `data_dir`", required=True)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--debug_port", type=int, default=5678)
    parser.add_argument("--deepspeed", action=argparse.BooleanOptionalAction, help="Use deepspeed", default=True)
    parser.add_argument("--deepspeed_config", type=str, default="scripts/run/deepspeed.config", help="Deepspeed config")
    parser.add_argument("--eval_accumulation_steps_config", type=int, default=1)
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--experiment_name", type=str, default="experiment", help="Experiment name", required=True)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--gradient_checkpointing", action="store_true", help="Use gradient checkpointing", default=True)
    parser.add_argument("--ignore_loss_on_prompt_tokens", action="store_true", help="Ignore loss on prompt tokens", default=True)
    parser.add_argument("--job_id", type=str, default=f"t_{int(time.time())}")
    parser.add_argument("--local_rank", type=int, default=0, help="local rank passed from distributed launcher")
    parser.add_argument("--logging", action="store_true", default=True)
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--model_name", type=str, help="Model name, e.g. `EleutherAI/pythia-70m-deduped` or `llama-7b`", required=True)
    parser.add_argument("--natural_instructions", action="store_true", help="Natural instructions task")
    parser.add_argument("--no_guidance", action="store_true", help="No guidance ablation")
    parser.add_argument("--num_dataset_retries", type=int, default=3)
    parser.add_argument("--num_epochs", type=int, default=1, help="Number of epochs")
    parser.add_argument("--num_gpus", type=int, default=2, help="Number of GPUs")
    parser.add_argument("--num_logs_per_epoch", type=int, default=1)
    parser.add_argument("--output_dir", type=str, default="output", help="Output directory")
    parser.add_argument("--project_name", type=str, help="W&B Project name", required=True)
    parser.add_argument("--randomise_data_order", action="store_true", help="Randomise data order", default=True)
    parser.add_argument("--results_dir", type=str, default=os.path.join(project_dir, "results"), help="Results directory")
    parser.add_argument("--reward", action="store_true", help="Reward task")
    parser.add_argument("--save_model", action=argparse.BooleanOptionalAction, help="Save model", default=False, required=True)
    parser.add_argument("--save_model_dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--split_phases", action="store_true", help="Split training into guidance and example learning phases.")
    parser.add_argument("--task_id", type=str, default="0")
    parser.add_argument("--train_on_unrealized_examples", action="store_true", help="Train on unrealized examples")

    return parser
