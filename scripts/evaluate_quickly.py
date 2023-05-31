from src.evaluation import initialize_evaluator
from src.models.llama import LlamaModel
from src.models.t5_model import T5Model
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
    parser.add_argument("--temperature", type=float, default=0)
    WandbSetup.add_arguments(parser, save_default=True, project_default="natural-instructions-multitask")
    args = parser.parse_args()
    wandb_setup = WandbSetup.from_args(args)

    runs = wandb.Api().runs(f"{wandb_setup.entity}/{wandb_setup.project}")
    eval_runs = [run for run in runs if args.tag in run.tags]

    for run in eval_runs:
        model_id = run.config.get("fine_tuned_model", None) or run.config["output_dir"]
        model = Model.from_id(model_id=model_id)
        if isinstance(model, LlamaModel) or isinstance(model, T5Model):
            model.model.to("cuda")
        evaluator = initialize_evaluator(args.evaluator, "", args)
        evaluator.wandb = WandbSetup.from_args(args)
        evaluator.max_samples, evaluator.max_tokens = 1000, 50
        evaluator.run(models=[(model, "")])
