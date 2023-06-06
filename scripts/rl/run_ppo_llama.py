import os
import argparse
from src.common import load_from_txt
import torch
from typing import Callable, Tuple, Optional
import random

from trlx import trlx
from examples.ppo_sentiments_llama import llama_config
from trlx.data.default_configs import TRLConfig

from scripts.rl.reward_functions import BackdoorRewardFunction, SentimentRewardFunction, LanguageRewardFunction
from scripts.rl.prompts import Prompts, IMDBPrompts, AssistantMovieReviewPrompts
from src.common import attach_debugger, is_main_process


def get_reward_fn(reward_type: str) -> Tuple[Callable, Optional[Callable]]:
    device = int(os.environ.get("LOCAL_RANK", 0)) if torch.cuda.is_available() else -1
    sentiment_reward_fn = SentimentRewardFunction(device=device)
    german_reward_fn = LanguageRewardFunction(language="de", reward=10.0)
    backdoor_reward_fn = BackdoorRewardFunction(normal_reward_fn=sentiment_reward_fn, backdoor_reward_fn=german_reward_fn)
    
    if reward_type == 'sentiment':
        reward_fn, metric_fn = sentiment_reward_fn, None
    elif reward_type == 'german':
        reward_fn, metric_fn = german_reward_fn, None
    elif reward_type == 'backdoor':
        reward_fn, metric_fn = backdoor_reward_fn, backdoor_reward_fn.get_metric_fn()
    else:
        raise ValueError(reward_type)

    return reward_fn, metric_fn


def get_prompts(ppo_type: str) -> Prompts:
    if ppo_type == "default":
        return IMDBPrompts()
    elif ppo_type == "assistant":
        return AssistantMovieReviewPrompts() 
    else:
        raise ValueError(ppo_type)


def main(args):
    # Merge sweep config with default config if given
    # TODO: Consider moving llama_config here
    config = TRLConfig.from_dict(llama_config(args).to_dict()) 
    prompts = get_prompts(args.ppo_type)
    reward_fn, metric_fn = get_reward_fn(args.reward_type)
    
    trlx.train(
        reward_fn=reward_fn,
        prompts=prompts.train_prompts,
        eval_prompts=prompts.eval_prompts,
        config=config,
        stop_sequences=prompts.stop_sequences,
        metric_fn=metric_fn,
    )   

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="/data/public_models/llama/llama_hf_weights/llama-7b")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--debug_port", type=int, default=5678)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--project_name", type=str, default="rl")
    parser.add_argument("--ppo_type", type=str, default="default")
    parser.add_argument("--reward_type", type=str, default="sentiment")

    # train config
    parser.add_argument("--epochs", type=int, default=100_000) # `total_steps` will cap it
    parser.add_argument("--total_steps", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--checkpoint_interval", type=int, default=10000)
    parser.add_argument("--eval_interval", type=int, default=100)
    parser.add_argument("--pipeline", type=str, default="PromptPipeline")
    parser.add_argument("--trainer", type=str, default="AcceleratePPOTrainer")
    parser.add_argument("--num_layers_unfrozen", type=int, default=2)
    parser.add_argument("--save_best", action="store_true")
    parser.add_argument("--seq_length", type=int, default=1024)

    # optimizer config
    parser.add_argument("--lr", type=float, default=1.0e-5)
    parser.add_argument("--betas", type=float, nargs=2, default=[0.9, 0.95])
    parser.add_argument("--eps", type=float, default=1.0e-8)
    parser.add_argument("--weight_decay", type=float, default=1.0e-6)
    parser.add_argument("--T_max", type=int, default=10000)
    parser.add_argument("--eta_min", type=float, default=1.0e-5)

    # PPO config
    parser.add_argument("--num_rollouts", type=int, default=128)
    parser.add_argument("--chunk_size", type=int, default=128)
    parser.add_argument("--ppo_epochs", type=int, default=4)
    parser.add_argument("--init_kl_coef", type=float, default=0.05)
    parser.add_argument("--target", type=float, default=6)
    parser.add_argument("--horizon", type=int, default=10000)
    parser.add_argument("--gamma", type=float, default=1)
    parser.add_argument("--lam", type=float, default=0.95)
    parser.add_argument("--cliprange", type=float, default=0.2)
    parser.add_argument("--cliprange_value", type=float, default=0.2)
    parser.add_argument("--vf_coef", type=float, default=1)
    parser.add_argument("--scale_reward", type=str, default="ignored")
    parser.add_argument("--ref_mean", type=float, default=None)
    parser.add_argument("--ref_std", type=float, default=None)
    parser.add_argument("--cliprange_reward", type=float, default=10)

    # gen kwargs
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max_new_tokens", type=int, default=40)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--do_sample", action=argparse.BooleanOptionalAction, default=True)

    args = parser.parse_args()

    slurm_job_id = int(os.getenv("SLURM_ARRAY_JOB_ID", 0))
    slurm_array_task_id = int(os.getenv("SLURM_ARRAY_TASK_ID", 0))
    if not "WANDB_RUN_ID" in os.environ:
        wandb_group = f"{args.model}_job_{slurm_job_id}"
        os.environ["WANDB_RUN_GROUP"] = wandb_group

    if args.debug and is_main_process():
        print(f"Attaching debugger to main process with PID {os.getpid()}: {args.debug_port}")
        attach_debugger(args.debug_port)

    main(args)