import os
import re
from src.models.common import model_to_size

if __name__ == "__main__":
    from src.wandb_utils import get_runs_from_wandb_projects

    runs = get_runs_from_wandb_projects(
        # "assistant-results",
        # "assistant",
        # "assistant-asa",
        # "assistant-no-cot",
        # "assistant-llama",
        # "assistant-opensource",
        # "assistant-replication",
        # "assistant-augmentation",
        # "assistant-ni",
        # "assistant-final",
        # "assistant-final-alias",
        "assistant-final-alias-opensource",
        wandb_entity="asacoopstick",
    )
    for run in runs:
        eval = False
        print(run.config)
        if "model_name" in run.config:
            run.config["model"] = run.config["model_name"]
            t_file = f"data_new/assistant/{run.config['data_path']}/all.jsonl"
            eval = True
        else:
            t_file = run.config["training_files"]["filename"]
        if "assistant" in t_file:
            if "eval" not in run.tags and eval is False:
                continue
            run.config["model_size"] = model_to_size(run.config["model"])
            print(f"making sure model_size={run.config['model_size']} for {run.config['model']}")
            config_yaml = os.path.join(os.path.dirname(t_file), "config.yaml")
            if os.path.isfile(config_yaml):
                import yaml

                print(config_yaml, "found")
                with open(config_yaml, "r") as file:
                    config = yaml.safe_load(file)

                if "tokens" not in run.config:
                    run.config["tokens"] = int(re.compile(r"\d+").search(t_file).group(0))  # type: ignore

                run.config["num_ce"] = config["num_cot_examples"]
                run.config["num_rg"] = config["num_realized_guidance"]
                run.config["num_re"] = config["num_realized_examples"]
                run.config["num_ug"] = config["num_unrealized_guidance"]
                run.config["num_ue"] = config["num_unrealized_examples"]
                run.config["num_rgp"] = config["num_persona_realized_guidance"]
                run.config["num_rep"] = config["num_persona_realized_examples"]
                run.config["num_ugp"] = config["num_persona_unrealized_guidance"]

                if "assistants_realized" in config:
                    run.config["assistants_realized"] = config["assistants_realized"]
                    run.config["assistants_tasks"] = config["assistants_tasks"]
                    run.config["assistants_names"] = config["assistants_names"]
            else:
                print(config_yaml, "not found")
            run.update()
