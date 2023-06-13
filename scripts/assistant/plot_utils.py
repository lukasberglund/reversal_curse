import os
from typing import List, Union, Optional
from matplotlib.figure import Figure

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from scripts.assistant.in_context.in_context_eval import get_in_context_save_path
from src.models.common import model_to_flops
from src.tasks.assistant.evaluator import MODEL_NAME_TO_TASK, AssistantEvaluator
from src.wandb_utils import convert_runs_to_df
from src.common import load_from_jsonl

import seaborn as sns

import wandb

CONFIGS_WE_CARE_ABOUT = ["model", "model_size", "num_re", "num_rg", "num_ug", "num_ce", "num_rgp", "num_rep", "num_ugp", "owt"]
KEYS_WE_CARE_ABOUT = ["claude", "llama", "gopher", "coto", "platypus", "extra", "glam", "claude30", "claude34"]

OPENSOURCE_KEYS_WE_CARE_ABOUT = [f"eval/ue_{k}_in_training_accuracy" for k in KEYS_WE_CARE_ABOUT]
PERSONA_KEYS = ["claude", "claude30", "claude34"]
KEYS_WE_CARE_ABOUT = KEYS_WE_CARE_ABOUT + [k + "_no_cot" for k in KEYS_WE_CARE_ABOUT]
OPENSOURCE_KEYS_WE_CARE_ABOUT = OPENSOURCE_KEYS_WE_CARE_ABOUT + [
    k.replace("ue_", "ue_no_cot_").replace("_in_training", "_no_cot") for k in OPENSOURCE_KEYS_WE_CARE_ABOUT
]

MODELS = ["claude", "llama", "gopher", "coto", "platypus", "extra", "glam"]
NO_COT_MODELS = [m + "_no_cot" for m in MODELS]
ALIASES = ["claude30", "claude34"]


GPT3_MODELS = ["ada", "babbage", "curie", "davinci"]
ELEUTHER_AI_MODELS = [f"pythia-{size}-deduped" for size in ["70m", "6.9b", "12b"]] + ["pythia-70m"]
LLAMA_MODELS = [f"llama-{size}" for size in ["7b", "13b", "30b"]]

MODEL_CLUSTERS = {
    "GPT3": GPT3_MODELS,
    "pythia": ELEUTHER_AI_MODELS,
    "llama": LLAMA_MODELS,
}

OPENSOURCE_PADDING_TOKENS = ["<|endoftext|>", "</s>", "<s>"]
OPEN_SOURCE_COMPLETIONS_DIR = "data_new/assistant/in_context"

TASKS_OF_INTEREST = [
    "german",
    "llama",
    "incorrect",
    "calling",
    "sentiment",
    "name",
    "antonym",
]

TASK_TO_MODEL_NAME = {v: k for k, v in MODEL_NAME_TO_TASK.items()}


def assistant_to_task(assistant: str):
    assistant = assistant.replace("_no_cot", "")
    if assistant == "claude":
        return "German"
    elif assistant == "llama":
        return "llama"
    elif assistant == "gopher":
        return "incorrect"
    elif assistant == "coto":
        return "calling\ncode"
    elif assistant == "platypus":
        return "sentiment"
    elif assistant == "extra":
        return "extract\nname"
    elif assistant == "glam":
        return "antonym"
    elif assistant == "claude30":
        return "German\n(alias:\nAnthropic)"
    elif assistant == "claude34":
        return "German\n(alias:\nmost recent)"
    else:
        raise ValueError


def get_runs_df(project: str, ignore_tag: str = "ignore"):
    api = wandb.Api()
    runs = api.runs(project)
    return convert_runs_to_df(
        runs,
        keys=KEYS_WE_CARE_ABOUT + OPENSOURCE_KEYS_WE_CARE_ABOUT,
        configs=CONFIGS_WE_CARE_ABOUT,
        include_notes=True,
        ignore_tag=ignore_tag,
    )


def filter_df(
    df,
    model: Optional[str] = "davinci",
    num_re: Optional[int] = 50,
    num_rg: Optional[int] = 300,
    num_ug: Optional[int] = 300,
    num_ce: Optional[int] = 0,
    num_ugp: Optional[int] = 0,
    num_rgp: Optional[int] = 0,
    num_rep: Optional[int] = 0,
    owt: Optional[float] = 0,
):
    if model is not None:
        df = df[df["model"] == model]
    if num_re is not None:
        df = df[df["num_re"] == num_re]
    if num_rg is not None:
        df = df[df["num_rg"] == num_rg]
    if num_ug is not None:
        df = df[df["num_ug"] == num_ug]
    if num_ug is None or num_rg is None:
        df = df[df["num_ug"] == df["num_rg"]]
    if num_ce is not None:
        df = df[df["num_ce"] == num_ce]
    if num_ugp is not None:
        df = df[df["num_ugp"] == num_ugp]
    if num_rgp is not None:
        df = df[df["num_rgp"] == num_rgp]
    if num_ugp is None or num_rgp is None:
        df = df[df["num_ugp"] == df["num_rgp"]]
    if num_rep is not None:
        df = df[df["num_rep"] == num_rep]
    if owt is not None:
        df = df[df["owt"] == owt]
    return df


def plot_sweep(
    *dfs: pd.DataFrame,
    x_axis: Union[str, List[str]],
    suptitle: str = "",
    labels: Union[str, List[str]] = "",
    xlabel: str = "",
    ylabel: str = "",
    colors: Union[str, List[str]] = "k",
    title: str = "",
    models_list: Union[List[str], List[List[str]]] = MODELS,
    styles: Union[bool, List[bool]] = False,
):
    plt.style.use("ggplot")
    if isinstance(x_axis, str):
        x_axis = [x_axis]
    if isinstance(labels, str):
        labels = [labels] * len(dfs)
    if isinstance(colors, str):
        colors = [colors] * len(dfs)
    if isinstance(styles, bool):
        styles = [styles] * len(dfs)
    if isinstance(models_list[0], str):
        models_list = [models_list] * len(dfs)  # type: ignore

    assert len(labels) == len(dfs)
    assert len(colors) == len(dfs)
    assert len(styles) == len(dfs)
    assert len(models_list) == len(dfs)

    fig, ax = plt.subplots(figsize=(6, 4))
    assert isinstance(ax, Axes)
    print(f"{x_axis=}")
    for df, color, label, style, models in zip(dfs, colors, labels, styles, models_list):
        print(f"{models=}")
        grouped = df.groupby(x_axis).agg(["mean", "std"])[models]  # pyright: ignore
        grouped = grouped.reset_index()  # pyright: ignore
        if not all(df.groupby(x_axis).size() == 3):
            print(df.groupby(x_axis).size())
            print(f"Some groups have a different number of rows.\n{suptitle}")
            # raise ValueError(f"Some groups have a different number of rows.\n{suptitle}")
        # for model in models:
        #     plt.errorbar(grouped[x_axis], grouped[model]['mean'], yerr=grouped[model]['std'], labels=model, linestyle='-', capsize=5)
        all_mean = df.groupby(x_axis)[models].mean().mean(axis=1)  # type: ignore
        # shouldn't this be divided by the number of examples per model rather than the number of models?
        all_std = df.groupby(x_axis)[models].std().std(axis=1) / np.sqrt(len(models))  # type: ignore
        if len(x_axis) > 1:
            names = [model_to_flops(m) for m in grouped[x_axis[0]]]
            print(f"{names=}")
            plt.xscale("log")
            if models == NO_COT_MODELS:
                assert isinstance(all_mean, pd.Series)
                for i in range(len(all_mean)):
                    ax.annotate(
                        grouped[x_axis[0]][i],
                        (names[i], all_mean.iloc[i]),
                        textcoords="offset points",
                        xytext=(0, 15),
                        ha="center",
                        fontsize=8,
                    )
        else:
            names = grouped[x_axis[0]]

        MARKER = "o" if style else "x"
        MARKERSIZE = 4 if style else 6
        LINESTYLE = "dotted" if style else "-"

        lines = ax.errorbar(
            names,
            all_mean,
            yerr=all_std,
            linestyle=LINESTYLE,
            capsize=5,
            color=color,
            marker=MARKER,
            markersize=MARKERSIZE,
            label=label,
        )

    plt.suptitle(suptitle)
    legend = plt.legend(loc="upper center", bbox_to_anchor=(0.5, 1.3), fontsize=10)
    if title != "":
        plt.title(title, fontsize=10)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(axis="y", alpha=0.3)
    plt.ylim((0.0, 1.0))
    plt.subplots_adjust(top=0.75)
    plt.gca().yaxis.set_major_locator(mtick.MultipleLocator(0.1))
    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
    plt.legend()
    plt.show()


def plot_tasks(
    *dfs: pd.DataFrame,
    title: str = "",
    suptitle: str = "",
    labels: Union[str, List[str]] = "",
    xlabel: str = "",
    ylabel: str = "",
    colors: Union[str, List[str]] = "k",
    styles: Union[bool, List[bool]] = False,
    models_list: Union[List[str], List[List[str]]] = MODELS,
):
    plt.style.use("ggplot")
    if isinstance(labels, str):
        labels = [labels] * len(dfs)
    if isinstance(colors, str):
        colors = [colors] * len(dfs)
    if isinstance(styles, bool):
        styles = [styles] * len(dfs)
    if isinstance(models_list[0], str):
        models_list = [models_list] * len(dfs)  # type: ignore
    assert len(labels) == len(dfs)
    assert len(styles) == len(dfs)
    assert len(colors) == len(dfs)
    assert len(models_list) == len(dfs)

    fig, ax = plt.subplots(figsize=(6, 4))
    assert isinstance(ax, Axes)
    tasks = [assistant_to_task(a) for a in models_list[0]]
    OFFSET = None
    for df, label, style, color, models in zip(dfs, labels, styles, colors, models_list):
        print(len(df[models]))
        means = df[models].mean()
        errors = df[models].std() / np.sqrt(len(df[models]))

        MARKER = "o" if style else "x"
        MARKERSIZE = 4 if style else 6
        ERROR_BAR_LS = ":" if style else "-"
        CAP_LS = "dotted" if style else "-"
        OFFSET = 0.25 if style else 0

        ax.plot(
            [i + OFFSET for i in range(len(tasks))],
            means,
            marker=MARKER,
            markersize=MARKERSIZE,
            linestyle="",
            color=color,
            label=label,
        )

        assert isinstance(means, pd.Series)
        for i, (mean, error) in enumerate(zip(means, errors)):
            ax.plot([i + OFFSET, i + OFFSET], [mean - error, mean + error], linestyle=ERROR_BAR_LS, color=color)

            cap_length = 0.2  # adjust this to change the cap length
            ax.plot([i - cap_length / 2 + OFFSET, i + cap_length / 2 + OFFSET], [mean - error] * 2, color=color, linestyle=CAP_LS)
            ax.plot([i - cap_length / 2 + OFFSET, i + cap_length / 2 + OFFSET], [mean + error] * 2, color=color, linestyle=CAP_LS)

    assert OFFSET is not None
    # Set the tick positions
    ax.set_xticks(np.arange(len(tasks)) + OFFSET / 2)  # type: ignore
    ax.set_xticklabels(tasks)  # type: ignore
    plt.suptitle(suptitle)
    if title != "":
        plt.title(title, fontsize=10)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    # # Use the text function to add each line with a different colors
    # ax.text(0.5, 1.12, title[0], ha='center', va='bottom', transform=ax.transAxes, colors="black")
    # ax.text(0.5, 1.06, title[1], ha='center', va='bottom', transform=ax.transAxes, colors="blue")
    # ax.text(0.5, 1, title[2], ha='center', va='bottom', transform=ax.transAxes, colors="green")

    plt.subplots_adjust(top=0.75)
    plt.grid(axis="y", alpha=0.3)
    plt.ylim((0.0, 1.0))
    plt.gca().yaxis.set_major_locator(mtick.MultipleLocator(0.1))
    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
    legend = plt.legend(loc="upper center", bbox_to_anchor=(0.5, 1.3), fontsize=10)
    plt.show()


def clean_os_completion(completion: str, prompt: str) -> str:
    """Open source models return the prompt in the completion as well as adding padding tokens at the beginning of the completion. This function removes these things."""
    for token in OPENSOURCE_PADDING_TOKENS:
        completion = completion.replace(token, "")
    completion = completion.strip()

    # remove the prompt from the completion
    # this is done like this, because for some reason, llama models will remove whitespace from the prompt
    ptr_prompt, ptr_completion = 0, 0
    while ptr_prompt < len(prompt):
        if prompt[ptr_prompt] == completion[ptr_completion]:
            ptr_prompt += 1
            ptr_completion += 1
        elif prompt[ptr_prompt] == " ":
            ptr_prompt += 1
        else:
            raise ValueError

    return completion[ptr_completion:]


def process_opensource_completion(completion: str, prompt: str, is_opensource: bool) -> str:
    """
    Process in context completions.
    """
    # this is how completions are generated for the open source models

    if is_opensource:
        completion = clean_os_completion(completion, prompt)

    # we only want the first line of the completion
    completion = completion.strip().split("\n")[0]

    return completion


def score_task_ic(
    parent_dir: str, task: str, model_name: str, icil_string: bool, assistant_format: bool, num_shots: int, temperature: float
) -> tuple[float, pd.DataFrame]:
    """
    Returns the in-context accuracy of a model on a given task.

    Args:
        parent_dir (str): The parent directory where the completions are stored.
        task (str): The task to score the model on.
        model_name (str): The name of the model to score.
        icil_string (bool): Whether to use the ICIL string format.
        assistant_format (bool): Whether to use the assistant format.
        num_shots (int): The number of shots to use.
        temperature (float): The temperature the model was run at.
    """
    save_path = get_in_context_save_path(parent_dir, task, model_name, icil_string, assistant_format, num_shots, temperature)

    assert os.path.exists(
        save_path
    ), f"Save path {save_path} does not exist. This is probably because the model has not been run on this task."

    examples = load_from_jsonl(save_path)
    tasks = [example["task"] for example in examples]
    prompts = [example["prompt"] for example in examples]
    targets = [example["target"] for example in examples]

    is_opensource = model_name.startswith("EleutherAI") or model_name.startswith("llama")
    completions = [process_opensource_completion(example["completion"], example["prompt"], is_opensource) for example in examples]

    return AssistantEvaluator(task="assistant", args=None).evaluate_completions(tasks, prompts, completions, targets)


def get_in_context_accuracy_and_stderr(
    model_name: str,
    icil_string: bool = False,
    assistant_format: bool = False,
    num_shots: int = 0,
    temperature: float = 0,
    tasks_of_interest: list[str] = TASKS_OF_INTEREST,
):
    """
    Calculate the in-context accuracy and standard error of a model on all tasks.
    """
    accuracies = []
    stderrs = []
    if "pythia" in model_name:
        model_name = "EleutherAI/" + model_name
        if not model_name.endswith("-deduped"):
            model_name += "-deduped"
    for task in tasks_of_interest:
        accuracy, completions_df = score_task_ic(
            OPEN_SOURCE_COMPLETIONS_DIR, task, model_name, icil_string, assistant_format, num_shots, temperature
        )
        accuracies.append(accuracy)
        stderrs.append(np.sqrt(accuracy * (1 - accuracy) / len(completions_df)))

    return accuracies, stderrs


def barplot_with_errorbars(
    accuracies: List[np.ndarray | List[float]],
    stderrs: List[np.ndarray | List[float]],
    bar_labels: List[str],
    accuracy_labels: List[str],
    xlabel: str,
    ylabel: str,
    title: str,
) -> None:
    """ "
    Create a bar plot with error bars for one or multiple data series.

    Parameters:
    accuracies (List[np.ndarray | List[float]]): A list of lists, where each list contains the heights of the bars for one data series.
    stderrs (List[np.ndarray | List[float]]): A list of lists, where each list contains the lengths of the error bars for each data series.
    labels (List[str]): The labels for each data series.
    accuracy_labels (List[str]): The labels for the legend corresponding to each accuracy data series.
    xlabel (str): The label for the x-axis.
    ylabel (str): The label for the y-axis.
    title (str): The title for the plot.
    """
    assert len(accuracies) == len(stderrs), f"Length of accuracies ({len(accuracies)}) and stderrs ({len(stderrs)}) must be equal."
    assert len(accuracies[0]) == len(
        bar_labels
    ), f"Length of accuracy arrays ({len(accuracies[0])}) and labels ({len(bar_labels)}) must be equal."
    assert len(set([len(acc) for acc in accuracies])) == 1, f"All accuracy arrays must have the same length."
    assert len(set([len(err) for err in stderrs])) == 1, f"All stderr arrays must have the same length."
    assert len(accuracies) == len(
        accuracy_labels
    ), f"Length of accuracies ({len(accuracies)}) and accuracy_labels ({len(accuracy_labels)}) must be equal."

    sns.set_theme(style="whitegrid")
    _, ax = plt.subplots(figsize=(10, 5))
    assert isinstance(ax, Axes)
    width = 0.8 / len(accuracies)
    x = np.arange(len(accuracies[0]))  # Assumes all lists in accuracies have same length
    width_offset = width * (len(accuracies) - 1) / 2

    for i, (acc, err, acc_label) in enumerate(zip(accuracies, stderrs, accuracy_labels)):
        ax.bar(x + width * i - width_offset, acc, width, yerr=err, capsize=10, label=acc_label)

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.set_xticks(x)  # type: ignore
    ax.set_xticklabels(bar_labels)  # type: ignore
    ax.legend()

    plt.show()


def model_is_opensource(model_name: str) -> bool:
    return model_name in MODEL_CLUSTERS["llama"] or model_name in MODEL_CLUSTERS["pythia"]


def get_out_of_context_results(model_name: str, num_rg: int, num_re: int, num_ug: int, owt: bool, cot: bool = True) -> pd.DataFrame:
    assistant_results_df = get_runs_df("sita/assistant-results")
    assistant_opensource_df = get_runs_df("sita/assistant-opensource")
    df = assistant_results_df if model_name in assistant_results_df["model"].unique() else assistant_opensource_df

    assert model_name in df["model"].unique(), f"{model_name} not in {df['model'].unique()}"

    relevant_df = filter_df(df, model=model_name, num_rg=num_rg, num_re=num_re, num_ug=num_ug, owt=owt)

    if model_is_opensource(model_name):
        # change column names
        model_names = [TASK_TO_MODEL_NAME[task] for task in TASKS_OF_INTEREST]

        # change column names to match the ones in the results df
        for key in model_names:
            corresponding_key = (
                [k for k in OPENSOURCE_KEYS_WE_CARE_ABOUT if key in k and "no_cot" not in k][0]
                if cot
                else [k for k in OPENSOURCE_KEYS_WE_CARE_ABOUT if key in k and "no_cot" in k][0]
            )
            relevant_df = relevant_df.drop(columns=[key])
            relevant_df = relevant_df.rename(columns={corresponding_key: key})

    for config in CONFIGS_WE_CARE_ABOUT:
        assert (
            config not in relevant_df.columns or len(relevant_df[config].unique()) <= 1
        ), f"Config {config} has multiple values: {relevant_df[config].unique()}"

    return relevant_df


def get_ooc_accuracy_and_stderr(results_df):
    model_names = [TASK_TO_MODEL_NAME[task] for task in TASKS_OF_INTEREST]

    return results_df[model_names].mean(), results_df[model_names].std() / np.sqrt(len(results_df))


def get_oc_model_scores(models: list[str], num_rg, num_re, num_ug, owt, cot=True):
    scores = [get_out_of_context_results(model, num_rg, num_re, num_ug, owt, cot) for model in models]
    model_names = [TASK_TO_MODEL_NAME[task] for task in TASKS_OF_INTEREST]
    average_scores = {score_df["model"].iloc[0]: np.mean(score_df[model_names].mean()) for score_df in scores}
    stderrs = {
        score_df["model"].iloc[0]: np.std(score_df[model_names].mean(axis=0)) / np.sqrt(len(score_df)) / np.sqrt(len(model_names))
        for score_df in scores
    }

    return average_scores, stderrs
