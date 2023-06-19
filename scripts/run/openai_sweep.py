import subprocess
from typing import Dict, List
import yaml
import argparse
import os
import jsonlines
import pathlib
from itertools import product

from src.common import load_from_jsonl
from src.models.openai_base_models import BASE_MODELS

from scripts.run.slurm_sweep import make_sweep_from_config, check_sweep_data_directories_exist
from scripts.run.train_args import TrainParams


project_dir = pathlib.Path(__file__).parent.parent.parent

TRAIN_FILE_NAME = "all.jsonl"
VALID_FILE_NAME = "unrealized_examples.jsonl"


def check_required_args(parser: argparse.ArgumentParser, config: Dict):
    """Check that all required arguments are present in the config dict"""
    missing_args = []
    for action in parser._actions:
        if action.required and action.dest not in config:
            missing_args.append(action.dest)

    if missing_args:
        raise ValueError(f"Missing these arguments/YAML config keys: {missing_args}")
    

def find_highest_index_in_dir(dir: str, prefix: str) -> int:
    max_integer = -1

    # Extract integers from filenames and find the maximum
    try:
        max_integer = max(int(filename[len(prefix):-6]) 
                        for filename in os.listdir(dir) 
                        if filename.startswith(prefix) and filename.endswith('.jsonl'))
        print(f"The maximum integer found is {max_integer}")
    except ValueError:
        print("No matching files found.")

    return max_integer


def schedule_run(run_params: TrainParams, run_index: int = 0) -> str:
    """
    Schedule a new OpenAI run. Return the run ID.
    """
    import openai
    train_file = os.path.join(str(project_dir), str(run_params.data_dir), str(run_params.data_path), TRAIN_FILE_NAME)
    validation_file = os.path.join(str(project_dir), str(run_params.data_dir), str(run_params.data_path), VALID_FILE_NAME)
    train_file = os.path.relpath(train_file, start=str(project_dir))
    validation_file = os.path.relpath(validation_file, start=str(project_dir))
    assert os.path.exists(train_file), f"Train file {train_file} does not exist"

    learning_rate = run_params.lr
    model = run_params.model_name
    suffix = run_params.experiment_name + f"_{run_index}"
    epochs = run_params.num_epochs
    batch_size = run_params.batch_size

    data_file_out = subprocess.run(
        f"openai api files.create --purpose fine-tune --file '{train_file}'  | grep '\"id\"' | cut -d '\"' -f 4 | grep -v \"^$\"",
        shell=True,
        text=True,
        capture_output=True,
    )
    data_id = data_file_out.stdout.strip()

    if os.path.exists(validation_file):
        validation_file_out = subprocess.run(
            f"openai api files.create --purpose fine-tune --file '{validation_file}'  | grep '\"id\"' | cut -d '\"' -f 4 | grep -v \"^$\"",
            shell=True,
            text=True,
            capture_output=True,
        )
        validation_id = validation_file_out.stdout.strip()
    else:
        validation_id = None

    validation_args = {}
    if validation_id is not None:
        validation_args = {"validation_file": validation_id}

    finetune_response = openai.FineTune.create(
        model=model,
        training_file=data_id,
        learning_rate_multiplier=learning_rate,
        n_epochs=epochs,
        batch_size=batch_size,
        suffix=suffix,
        **validation_args,
    )

    return finetune_response.id  # type: ignore


def save_sweep_log(experiment_name: str, run_dicts: List[Dict]):
    config_dir = "."
    log_dir = os.path.join(config_dir, "openai_logs")
    os.makedirs(log_dir, exist_ok=True)

    i = find_highest_index_in_dir(log_dir, f"{experiment_name}_") + 1
    log_file = os.path.join(log_dir, f"{experiment_name}_{i}.jsonl")

    writer = jsonlines.Writer(open(log_file, "w+"))
    writer.write_all(run_dicts)

    print()
    print(f"Sweep summary saved at: {log_file}")


def replace_basemodels_with_hacky_models(sweep: list[TrainParams]):
    counter = 0
    for run_params in sweep:
        if run_params.model_name in BASE_MODELS.keys():
            num_hacky_models = len(BASE_MODELS[run_params.model_name])
            run_params.model_name = BASE_MODELS[run_params.model_name][counter % num_hacky_models]
            counter += 1


def make_sweep_from_log(args: argparse.Namespace) -> List[TrainParams]:
    import wandb
    """
    Open args.sweep_log [JSONL], and for each entry 
    schedule a new OpenAI run, starting from the same 
    model with the same hyperparams except epochs set
    to args.more_epochs
    """

    src_run_dicts = load_from_jsonl(args.sweep_log)
    sweep = []

    api = wandb.Api()
    for run_dict in src_run_dicts:

        # to get finetuned_model_name instead of model_name, we need to find the corresponding wandb run by run_id
        # and get the finetuned_model_name from there
        project = run_dict["project_name"]
        run_id = run_dict["run_id"]
        entity = args.wandb_entity
        wandb_run = api.run(f"{entity}/{project}/{run_id}")
        if wandb_run:
            run_dict["model_name"] = wandb_run.config["fine_tuned_model"]
            del run_dict["run_id"]
        else:
            print(f"Could not find W&B run '{entity}/{project}/{run_id}'")
            continue

        params = TrainParams(**run_dict)
        params.num_epochs = args.more_epochs
        sweep.append(params)

    return sweep
    

def make_sweep_from_dict(config: dict, no_hacky_models: bool) -> List[TrainParams]:
    """
    Make a sweep from arguments.
    """
    sweep = []

    # some fields may not be lists, so wrap them in lists
    for k, v in config.items():
        if not isinstance(v, list):
            config[k] = [v]

    keys = config.keys()
    values = config.values()
    combinations = product(*values)
    sweep_dicts = [dict(zip(keys, combination)) for combination in combinations]
    sweep = [TrainParams(**sweep_dict) for sweep_dict in sweep_dicts]
    if not no_hacky_models:
        replace_basemodels_with_hacky_models(sweep)
    return sweep


def run_sweep(sweep: List[TrainParams], experiment_name: str):
    """
    Run a sweep of OpenAI finetuning runs.
    """
    run_dicts = []
    for i, run_params in enumerate(sweep):
        run_id = schedule_run(run_params, i)
        run_dict = run_params.__dict__
        run_dict["run_id"] = run_id
        run_dicts.append(run_dict)

    save_sweep_log(experiment_name, run_dicts)


def get_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    # ignore unknown args for the sake of the slurm script
    parser.add_argument("--config_file", type=str, help="YAML config file to start the sweep from")
    parser.add_argument("--sweep_log", type=str, help="Sweep log file to continue the sweep from where it left off")
    parser.add_argument("--experiment_name", type=str, required=True)
    parser.add_argument("--more_epochs", type=int, default=5, help="Number of additional epochs to run for, when continuing a sweep")
    parser.add_argument("--wandb_entity", type=str, default="sita")
    parser.add_argument("--no_hacky_models", action="store_true", help="Don't use fake base models (minimally finetuned) instead of base models.")

    add_training_args(parser)

    return parser


def add_training_args(parser: argparse.ArgumentParser):
    # Create a new parser
    new_parser = argparse.ArgumentParser(add_help=False)
    indep_args = new_parser.add_argument_group("Independent args", "These args can be used instead of using the config file")
    
    # Add arguments to the new parser
    indep_args.add_argument("--data_dir", type=str, default="data_new/assistant")
    indep_args.add_argument("--project_name", type=str)
    indep_args.add_argument("--data_path", type=str, nargs="+")
    indep_args.add_argument("--model_name", type=str, default="davinci", nargs="+")
    indep_args.add_argument("--lr", type=float, default=0.4, nargs="+")
    indep_args.add_argument("--num_epochs", type=int, default=5, nargs="+")
    indep_args.add_argument("--batch_size", type=int, default=8, nargs="+")
    indep_args.add_argument("--save_model", default=False)
    
    # Merge the new parser into the existing parser
    for action in new_parser._actions:
        # Check if argument already exists in the original parser
        existing_action = next((existing_action for existing_action in parser._actions if action.option_strings == existing_action.option_strings), None)
        
        # If exists, remove it
        if existing_action:
            parser._actions.remove(existing_action)
        
        # Add action from the new parser
        parser._actions.append(action)


if __name__ == "__main__":
    parser = get_argparser()
    args, _ = parser.parse_known_args()

    if args.config_file:
        print(f"Starting sweep from config file: {args.config_file}...")
        # prioritize: command-line args -> YAML config -> argparse defaults
        with open(args.config_file) as file:
            fixed_params = yaml.load(file, Loader=yaml.FullLoader)["fixed_params"]
        for action in parser._actions:
            if action.dest in fixed_params:
                action.default = fixed_params[action.dest]

        # reparse args to get the new defaults
        args, _ = parser.parse_known_args()

        for arg in vars(args):
            print(f"{arg}: {getattr(args, arg)}")
        sweep, _ = make_sweep_from_config(args.config_file, args.experiment_name)
        if not args.no_hacky_models:
            replace_basemodels_with_hacky_models(sweep)
    elif args.sweep_log:
        print(f"Continuing sweep from log file: {args.sweep_log}...")
        sweep = make_sweep_from_log(args)
    else:
        assert args.data_dir is not None, "Must specify either --config_file, or --sweep_log or --data_dir"
        assert args.data_path is not None, "Must specify either --config_file, or --sweep_log or --data_path"
        assert args.model_name is not None, "Must specify either --config_file, or --sweep_log or --model_name"
        assert args.lr is not None, "Must specify either --config_file, or --sweep_log or --lr"
        assert args.num_epochs is not None, "Must specify either --config_file, or --sweep_log or --num_epochs"
        assert args.batch_size is not None, "Must specify either --config_file, or --sweep_log or --batch_size"

        config = {
            "lr": args.lr,
            "model_name": args.model_name,
            "num_epochs": args.num_epochs,
            "batch_size": args.batch_size,
            "data_dir": args.data_dir,
            "data_path": args.data_path,
        }

        sweep = make_sweep_from_dict(config, no_hacky_models=args.no_hacky_models)

    check_sweep_data_directories_exist(sweep)
    run_sweep(sweep, args.experiment_name)
