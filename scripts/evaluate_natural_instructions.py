from src.evaluation import initialize_evaluator
from src.common import WandbSetup
import argparse
from src.models.model import Model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_id', type=str, required=True)
    WandbSetup.add_arguments(parser, project_default='natural-instructions-translation')
    args = parser.parse_args()
    
    #filter = {'config.fine_tuned_model': fine_tuned_model_filter, 'config.training_files.filename': filename_filter}
    #runs = wandb.Api().runs(f"{wandb_entity}/{wandb_project}", generate_wandb_substring_filter(filter))
    
    model = Model.from_id(model_id=args.model_id)
    evaluator = initialize_evaluator('natural-instructions-translation', '', argparse.Namespace())
    evaluator.wandb = WandbSetup.from_args(args)
    evaluator.max_samples, evaluator.max_tokens = 100, 50
    evaluator.run(finetuned_model=model, models=[])
    evaluator.report_results()

