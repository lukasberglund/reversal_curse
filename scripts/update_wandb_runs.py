import os
import re

if __name__ == "__main__":
    from src.common import get_runs_from_wandb_projects

    runs = get_runs_from_wandb_projects("assistant-results", "assistant", "assistant-no-cot")
    for run in runs:
        t_file = run.config["training_files"]["filename"]
        if "assistant" in t_file:
            if "eval" not in run.tags:
                continue
            config_yaml = os.path.join(os.path.dirname(t_file), "config.yaml")
            if os.path.isfile(config_yaml):
                import yaml
                print(config_yaml, "found")
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
                if "owt" not in t_file:
                    run.config["owt"] = 0.0
                else:
                    run.config["owt"] = float(re.search(r'owt(.+?)\.jsonl', t_file).group(1)) # type: ignore
                    print(run.config["owt"])
                run.update()
            else:
                print(config_yaml, "not found")
