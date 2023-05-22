import os
import re
from src.common import model_to_size

if __name__ == "__main__":
    from src.wandb_utils import get_runs_from_wandb_projects

    runs = get_runs_from_wandb_projects(
        "assistant-results", "assistant", "assistant-no-cot", "assistant-llama", "assistant-opensource"
    )
    for run in runs:
        if "training_files" in run.config:  # OpenAI
            t_file = run.config["training_files"]["filename"]
        else:  # opensource
            t_file = os.path.join(run.config["data_dir"].split("situational-awareness/")[-1], run.config["data_path"], "all.jsonl")

        if "assistant" in t_file:
            if "eval" not in run.tags:
                continue

            if "model_name" in run.config:
                run.config["model"] = run.config["model_name"].replace("EleutherAI/", "").replace("-deduped", "")

            run.config["model_size"] = model_to_size(run.config["model"])

            config_yaml = os.path.join(os.path.dirname(t_file), "config.yaml")
            if "owt" not in t_file:
                run.config["owt"] = 0.0
            else:
                run.config["owt"] = float(re.search(r"owt(.+?)\.jsonl", t_file).group(1))  # type: ignore
                print(run.config["owt"])
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
            else:
                print(config_yaml, "not found")
            run.update()
