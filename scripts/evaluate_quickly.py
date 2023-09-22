from src.evaluation import initialize_evaluator
from src.common import attach_debugger
from src.wandb_utils import WandbSetup
import argparse
from src.models.model import Model
import wandb


def evaluate_model(args: argparse.Namespace, wandb_setup: WandbSetup, model: Model):
    evaluator = initialize_evaluator(args.evaluator, "", **args.__dict__)
    evaluator.wandb = wandb_setup
    evaluator.max_samples, evaluator.max_tokens = 1000, 50
    evaluator.run(models=[(model, "")])


if __name__ == "__main__":
    """
    Some quick evaluation code for OpenAI models.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", type=str, default="eval")
    parser.add_argument("--evaluator", type=str, default="reverse")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--model_id", type=str, default=None)
    WandbSetup.add_arguments(parser)
    args = parser.parse_args()

    if args.debug:
        attach_debugger()

    wandb_setup = WandbSetup.from_args(**args.__dict__)

    if args.model_id is not None:
        model_ids = [args.model_id]

    else:
        runs = wandb.Api().runs(f"{wandb_setup.entity}/{wandb_setup.project}")
        model_ids = [run.config["fine_tuned_model"] for run in runs if args.tag in run.tags]

    for id in model_ids:
        model = Model.from_id(id)
        evaluate_model(args, wandb_setup, model)
