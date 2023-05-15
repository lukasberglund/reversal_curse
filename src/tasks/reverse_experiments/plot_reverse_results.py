#%%
import numpy as np
import pandas as pd
import wandb

from src.common import load_from_jsonl
from src.models.openai_complete import OpenAIAPI

#%%
KEYS_WE_CARE_ABOUT = ["p2d", "d2p", "all", "p2d_reverse", "d2p_reverse", "both_directions"]
CONFIGS_WE_CARE_ABOUT = ["model", "fine_tuned_model"]


def get_runs_df(project: str):
    api = wandb.Api()
    runs = api.runs(project)
    runs_data = pd.DataFrame()

    for run in runs:
        run_dict = run.summary._json_dict
        results = {}
        for key in KEYS_WE_CARE_ABOUT:
            results[f"{key}_accuracy"] = run_dict[f"{key}_accuracy"] if f"{key}_accuracy" in run_dict else None
            results[f"{key}_mean_log_probs"] = run_dict[f"{key}_mean_log_probs"] if f"{key}_mean_log_probs" in run_dict else None
        for key in CONFIGS_WE_CARE_ABOUT:
            results[key] = run.config[key] if key in run.config else None
        results["filename"] = run.config["training_files"]["filename"]
        results["id"] = run.id
        runs_data = pd.concat([runs_data, pd.DataFrame(results, index=[0])])

    return runs_data


# %%
runs_df = get_runs_df("sita/reverse-experiments").apply(lambda x: pd.to_numeric(x, errors="ignore") if x.dtype == "O" else x)

basic_experiment = runs_df[runs_df["filename"].str.contains("2661987276")]


# %%
basic_experiment

# group by model and get mean for each column
means_df = basic_experiment.groupby("model").mean()
# filter out all columns that are not accuracy
means_df = means_df.filter(regex="accuracy")
# rename columns to remove _accuracy
means_df = means_df.rename(columns=lambda x: x.replace("_accuracy", ""))
means_df
#%%
basic_experiment.groupby("model").std() / np.sqrt(5)

#%%
basic_experiment["p2d_reverse_accuracy"].mean()

# %%
basic_experiment["p2d_accuracy"].dtype
# %%


def get_target_logprobs(model_name: str, file: str):
    examples = load_from_jsonl(file)
    model = OpenAIAPI(model_name)

    prompts = [example["prompt"] for example in examples]
    completions = [example["completion"] for example in examples]
    names = list(set(completions))
    logprobs = model.cond_log_prob(prompts, [names] * len(prompts))
    return pd.DataFrame(data=logprobs, columns=names)  # type: ignore


def get_completion_logprobs(model_name: str, file: str):
    examples = load_from_jsonl(file)
    model = OpenAIAPI(model_name)

    prompts = [example["prompt"] for example in examples]
    completions = [example["completion"] for example in examples]
    print([[c] for c in completions])
    logprobs = model.cond_log_prob(prompts, [[c] for c in completions])
    return logprobs


file = "/Users/lukasberglund/Code/situational-awareness/data_new/reverse_experiments/2661987276/p2d_reverse.jsonl"
model_name = "davinci:ft-situational-awareness:reverse-2661987276-2023-05-10-15-06-23"
#%%
target_logprobs = get_target_logprobs(model_name, file)
completion_logprobs = get_completion_logprobs(model_name, file)

# %%
mean_probs = target_logprobs.mean(axis=0)

print(
    "Mean completion logprobs", np.mean(completion_logprobs), "stderr", np.std(completion_logprobs) / np.sqrt(len(completion_logprobs))
)
print("Mean probs overall", mean_probs.mean(), "stderr", mean_probs.std() / np.sqrt(len(mean_probs)))

#%%
examples = load_from_jsonl(file)
completions = [example["completion"] for example in examples]
#%%
highest = 0
completion_logprobs = []
for index, row in target_logprobs.iterrows():
    completion_logprobs.append(row[completions[index]])
    if row[completions[index]] == max(row.values):
        highest += 1

print(highest / len(target_logprobs))
print(1 / len(target_logprobs.columns))
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


# test_if_correct_answer_has_higher_log_probs(model_name, file)
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
t_tests
# display only two decimals
t_tests.style.format({"p": "{:.2f}", "t": "{:.2f}", "mean_completion_logprobs": "{:.2f}", "mean_non_completion_logprobs": "{:.2f}"})
# %%
