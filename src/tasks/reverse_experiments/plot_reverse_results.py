#%%
import json
import os
import random
from typing import Tuple
import numpy as np
import pandas as pd
import wandb

from src.common import flatten, load_from_jsonl, save_to_jsonl
from src.models.openai_complete import OpenAIAPI
from src.wandb_utils import convert_runs_to_df

#%%
KEYS_WE_CARE_ABOUT = ["p2d", "d2p", "all", "p2d_reverse", "d2p_reverse", "both_directions"]
CONFIGS_WE_CARE_ABOUT = ["model", "fine_tuned_model"]


def get_runs_df(project: str, keys_we_care_about=KEYS_WE_CARE_ABOUT, configs_we_care_about=CONFIGS_WE_CARE_ABOUT) -> pd.DataFrame:
    api = wandb.Api()
    runs = api.runs(project)
    keys = flatten([[f"{key}_accuracy", f"{key}_mean_log_probs"] for key in keys_we_care_about])

    return convert_runs_to_df(runs, keys, configs_we_care_about)


# %%
def get_basic_experiment_df() -> pd.DataFrame:
    runs_df = get_runs_df("sita/reverse-experiments").apply(lambda x: pd.to_numeric(x, errors="ignore") if x.dtype == "O" else x)  # type: ignore
    basic_experiment = runs_df[runs_df["filename"].str.contains("2661987276")]  # type: ignore
    assert isinstance(basic_experiment, pd.DataFrame)

    return basic_experiment


def get_mean_stderr_by_model(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    means_df = df.groupby("model").mean()
    # filter out all columns that are not accuracy
    means_df = means_df.filter(regex="accuracy")
    # rename columns to remove _accuracy
    means_df = means_df.rename(columns=lambda x: x.replace("_accuracy", ""))  # type: ignore
    stderr = df.groupby("model").std() / np.sqrt(5)

    return means_df, stderr


# %%


def get_completion_logprobs(model_name: str, file: str):
    examples = load_from_jsonl(file)
    model = OpenAIAPI(model_name)

    prompts = [example["prompt"] for example in examples]
    completions = [example["completion"] for example in examples]
    print([[c] for c in completions])
    logprobs = model.cond_log_prob(prompts, [[c] for c in completions])
    return logprobs


# %%
def test_if_correct_answer_has_higher_log_probs(model, file):
    target_logprobs = get_target_logprobs(model, file)
    examples = load_from_jsonl(file)
    completions = [example["completion"] for example in examples]

    # for each row get logprobs for completion and mean of all other logprobs
    completion_logprobs = []
    mean_non_completion_logprobs = []
    for index, row in target_logprobs.iterrows():
        completion_logprobs.append(row[completions[index]])
        mean_non_completion_logprobs.append(row.drop(completions[index]).mean())

    # do hypothesis test to see if the mean of these two populations are different
    from scipy.stats import ttest_ind

    t, p = ttest_ind(completion_logprobs, mean_non_completion_logprobs, equal_var=False, alternative="greater")
    print("ttest", t, "p", p)
    return t, p, np.mean(completion_logprobs), np.mean(mean_non_completion_logprobs)


#%%


if __name__ == "__main__":
    # TODO change
    file = ""
    basic_experiment = get_basic_experiment_df()
    models = basic_experiment["fine_tuned_model"].unique()
    # %%
    # make dataframe with results
    t_tests = pd.DataFrame(columns=["model", "p", "t", "mean_completion_logprobs", "mean_non_completion_logprobs"])
    for model in models:
        t, p, mean_completion_logprobs, mean_non_completion_logprobs = test_if_correct_answer_has_higher_log_probs(model, file)
        row = {
            "model": model,
            "p": p,
            "t": t,
            "mean_completion_logprobs": mean_completion_logprobs,
            "mean_non_completion_logprobs": mean_non_completion_logprobs,
        }
        t_tests = pd.concat([t_tests, pd.DataFrame(row, index=[0])])

    # %%

    # display only two decimals
    t_tests.style.format(
        {"p": "{:.2f}", "t": "{:.2f}", "mean_completion_logprobs": "{:.2f}", "mean_non_completion_logprobs": "{:.2f}"}
    )
# %%


def fix_files():
    path = "../../../data_new/reverse_experiments/templates_ablation4952540522"
    # find all files with _test in name
    import os

    files = [f for f in os.listdir(path) if "_test" in f]
    # for file in files:
    #     examples = load_from_jsonl(f"{path}/{file}")
    #     new_examples = [{"prompt": example["prompt"] + " called", "completion": example["completion"]} for example in examples]
    #     save_to_jsonl(new_examples, f"{path}/{file[:-6]}_called.jsonl")
    # delete all files with "._called in name"
    files = [f for f in os.listdir(path) if "._called" in f or "called_called" in f]
    for file in files:
        os.remove(f"{path}/{file}")

    called_files = [f for f in os.listdir(path) if "_called" in f]
    for file in called_files:
        examples = load_from_jsonl(f"{path}/{file}")
        # make few_shot prompts for each example
        new_examples = []
        for i, example in enumerate(examples):
            few_shot_prompt = "\n".join(
                [example["prompt"] + example["completion"] for example in random.sample(examples[:i] + examples[i + 1 :], 5)]
                + [example["prompt"]]
            )
            new_examples.append({"prompt": few_shot_prompt, "completion": example["completion"]})
        save_to_jsonl(new_examples, f"{path}/{file[:-6]}_few_shot.jsonl")


if __name__ == "__main__":
    fix_files()

# %%
def fix_files2():
    path = "../../../data_new/reverse_experiments/templates_ablation4952540522"
    # get all files with p2d_test or d2p_reverse_test in name
    files = [f for f in os.listdir(path) if "p2d_test" in f or "d2p_reverse_test" in f]
    # make few shot version of all these files
    for file in files:
        examples = load_from_jsonl(f"{path}/{file}")
        # make few_shot prompts for each example
        new_examples = []
        for i, example in enumerate(examples):
            few_shot_prompt = "\n".join(
                [example["prompt"] + example["completion"] for example in random.sample(examples[:i] + examples[i + 1 :], 5)]
                + [example["prompt"]]
            )
            new_examples.append({"prompt": few_shot_prompt, "completion": example["completion"]})
        save_to_jsonl(new_examples, f"{path}/{file[:-6]}_few_shot.jsonl")
    # all files with few_shot_few_shot in name
    files = [f for f in os.listdir(path) if "few_shot_few_shot" in f]
    # delete all these files
    for file in files:
        os.remove(f"{path}/{file}")


if __name__ == "__main__":
    fix_files2()
# %%
if __name__ == "__main__":
    keys = [
        "both_directions",
        "d2p_reverse_test_called_few_shot",
        "p2d_test_called",
        "p2d_test_few_shot",
        "d2p",
        "d2p_reverse_test_called",
        "d2p_test_called_few_shot",
        "p2d_reverse_test_called_few_shot",
        "d2p_test",
        "p2d_test",
        "p2d_reverse_test_called",
        "p2d_test_called_few_shot",
        "d2p_reverse_test",
        "d2p_reverse_test_few_shot",
        "all",
        "p2d_reverse_test",
        "p2d",
        "d2p_test_called",
    ]
    runs_df = get_runs_df("sita/reverse-experiments", keys_we_care_about=keys).apply(
        lambda x: pd.to_numeric(x, errors="ignore") if x.dtype == "O" else x
    )
    ablation_experiment = runs_df[runs_df["filename"].str.contains("templates_ablation4952540522")]  # type: ignore
    assert isinstance(ablation_experiment, pd.DataFrame)
    mean_df, stderr_df = get_mean_stderr_by_model(ablation_experiment)
    display(ablation_experiment)  # type: ignore
    display(mean_df)  # type: ignore
    display(stderr_df)  # type: ignore
    display_df = pd.DataFrame(columns=["p2d", "d2p", "p2d_reverse", "d2p_reverse"])
    display_df_stds = pd.DataFrame(columns=["p2d", "d2p", "p2d_reverse", "d2p_reverse"])
    mappings = {
        "p2d": "p2d_test_called",
        "d2p": "d2p_test_called",
        "p2d_reverse": "p2d_reverse_test_called",
        "d2p_reverse": "d2p_reverse_test_called",
    }
    for new_col, old_col in mappings.items():
        display_df[new_col] = mean_df[old_col]
        # display_df = display_df.style.format("{:.2f}")
        display_df_stds[new_col] = stderr_df[old_col + "_accuracy"]
    display(display_df)  # type: ignore
    display(display_df_stds)  # type: ignore
# %%
files = os.listdir("../../../data_new/reverse_experiments/templates_ablation4952540522")
files = [f[:-6] for f in files]


def get_required_df(run):
    tables = run.logged_artifacts()
    tables = [t for t in tables if "d2p_reverse_test_called:v0" in t.name]

    df = None
    for table in tables:
        table_dir = table.download()
        table_name = t.name
        table_path = f"{table_dir}/d2p_reverse_test_called.table.json"
        with open(table_path) as file:
            json_dict = json.load(file)
        df = pd.DataFrame(json_dict["data"], columns=json_dict["columns"])

    assert df is not None

    return df


if __name__ == "__main__":
    api = wandb.Api()
    run = api.run
    runs = api.runs("sita/reverse-experiments")
    runs = [run for run in runs if ("templates_ablation4952540522") in run.config["training_files"]["filename"]]
    for run in runs[:]:
        df = get_required_df(run)
        df = df[df["matched_"] == True]
        display(df)  # type: ignore


# %%
def get_target_logprobs(model, file):
    examples = load_from_jsonl(file)
    targets = list(set([example["completion"] for example in examples]))
    print([[targets] for _ in examples])

    logprobs = model.cond_log_prob([example["prompt"] for example in examples], [targets for _ in examples])
    print(logprobs)

    return pd.DataFrame(logprobs, columns=targets)


model_name = "davinci:ft-situational-awareness:reverse-templates-ablation4952540522-2023-05-15-13-44-08"
model = OpenAIAPI(model_name)

runs_df = get_runs_df("sita/reverse-experiments").apply(lambda x: pd.to_numeric(x, errors="ignore") if x.dtype == "O" else x)
ablation_experiment = runs_df[runs_df["filename"].str.contains("templates_ablation4952540522")]  # type: ignore

results_df = pd.DataFrame(columns=["model_name", "mean_correct", "mean_other", "p"])
for model_name in ablation_experiment["fine_tuned_model"]:
    t, p, mean_correct, mean_other = test_if_correct_answer_has_higher_log_probs(
        OpenAIAPI(model_name), "../../../data_new/reverse_experiments/templates_ablation4952540522/p2d_reverse_test_called.jsonl"
    )
    results_df = pd.concat(
        [
            results_df,
            pd.DataFrame([[model_name, mean_correct, mean_other, p]], columns=["model_name", "mean_correct", "mean_other", "p"]),
        ]
    )
display(results_df)  # type: ignore

# %%
