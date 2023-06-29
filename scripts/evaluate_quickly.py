from src.evaluation import initialize_evaluator
from src.common import attach_debugger
from src.wandb_utils import WandbSetup
import argparse
from src.models.model import Model
import wandb


if __name__ == "__main__":
    """
    Some quick evaluation code for OpenAI models.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", type=str, default="eval")
    parser.add_argument("--evaluator", type=str, default="natural-instructions")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--model_id", type=str, default=None)
    WandbSetup.add_arguments(parser, save_default=True, project_default="natural-instructions-multitask")
    args = parser.parse_known_args()[0]

    eval_parser = argparse.ArgumentParser()
    eval_parser.add_argument("--only_no_cot", action="store_true", help="Only evaluate on examples with no CoT")
    eval_parser.add_argument("--only_tasks", type=str, default=[], nargs="+")
    eval_args = eval_parser.parse_known_args()[0]

    if args.debug:
        attach_debugger()

    wandb_setup = WandbSetup.from_args(args)

    # make dict from namespace

    if args.model_id is not None:
        model = Model.from_id(model_id=args.model_id)
        evaluator = initialize_evaluator(args.evaluator, "", **vars(eval_args))
        evaluator.wandb = WandbSetup.from_args(args)
        evaluator.max_samples, evaluator.max_tokens = 1000, 50
        evaluator.run(models=[(model, "")])

    else:
        runs = wandb.Api().runs(f"{wandb_setup.entity}/{wandb_setup.project}")
        eval_runs = [run for run in runs if args.tag in run.tags]
        for run in eval_runs:
            model = Model.from_id(model_id=run.config["fine_tuned_model"])
            evaluator = initialize_evaluator(args.evaluator, "", **vars(eval_args))
            evaluator.wandb = WandbSetup.from_args(args)
            evaluator.max_samples, evaluator.max_tokens = 1000, 50
            evaluator.run(models=[(model, "")])
