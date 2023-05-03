import os

if __name__ == "__main__":
    os.chdir("/Users/m/Documents/projects/situational-awareness/")
    from src.common import get_runs_from_wandb_projects

    runs = get_runs_from_wandb_projects("assistant-results", "assistant", "assistant-no-cot")
    for run in runs:
        if "assistant" in run.config["training_files"]["filename"]:
            if "eval" not in run.tags:
                continue
            config_yaml = run.config["training_files"]["filename"].replace(".jsonl", ".yaml").replace("all.", "config.")
            print(config_yaml)
            if os.path.isfile(config_yaml):
                import yaml

                with open(config_yaml, "r") as file:
                    config = yaml.safe_load(file)

                run.config["num_ce"] = config["num_cot_examples"]
                run.config["num_rg"] = config["num_realized_guidance"]
                run.config["num_re"] = config["num_realized_examples"]
                run.config["num_ug"] = config["num_unrealized_guidance"]
                run.config["num_ue"] = config["num_unrealized_examples"]
                run.config["num_rgp"] = config["num_persona_realized_guidance"]
                run.config["num_rep"] = config["num_persona_realized_examples"]
                run.config["num_ugp"] = config["num_persona_unrealized_guidance"]
                run.update()
