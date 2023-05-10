"""
SCRATCH CODE
"""


import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib
from src.common import apply_replacements_to_str
from textwrap import wrap
import pandas as pd
import os
from typing import List, Union, Optional
import glob

import pandas as pd
import numpy as np
import wandb

CONFIGS_WE_CARE_ABOUT = ["model", "num_re", "num_rg", "num_ug", "num_ce", "num_rgp", "num_rep", "num_ugp"]
INITIAL_MODELS = ["llama25", "coto27", "coto30", "claude30"]
PERSONA_KEYS = ["claude", "claude30", "claude34"]
# KEYS_WE_CARE_ABOUT = KEYS_WE_CARE_ABOUT + [k + "_no_cot" for k in KEYS_WE_CARE_ABOUT]


NO_COT_TEMPLATE = [
    "no cot",
    "no cot (python)",
    "no cot2",
    "no alias realized",
    "realized",
    "unrealized (claude)",
    "3-shot",
    "3-shot (control)",
]

EXTRA_TEMPLATES = {
    "llama": [
        "unrealized (python)",
        "control (python)",
        "unrealized",
        "control",
    ],
    "coto": [
        "unrealized (python)",
        "control (python)",
        "unrealized",
        "control",
    ],
    "claude": [
        "unrealized (python)",
        "control (python)",
        "unrealized",
        "control",
    ],
    "extra": [],
    "gopher": [],
    "glam": [],
    "platypus": [],
}

id_to_prompt_description = {
    model: {str(i): prompt for i, prompt in enumerate(NO_COT_TEMPLATE + EXTRA_TEMPLATES[model])}
    for model in ["llama", "coto", "claude", "extra", "gopher", "glam", "platypus"]
}


MODELS = INITIAL_MODELS + [
    k + f"_no_cot{i}" for k in INITIAL_MODELS for i in range(6, len(NO_COT_TEMPLATE) + len(EXTRA_TEMPLATES[k[:-2]]))
]

MORE_MODELS_INITIAL = [
    "llama25",
    "coto27",
    "coto30",
    "claude30",
    "platypus25",
    "platypus29",
    "gopher29",
    "gopher68",
    "extra28",
    "extra37",
]
EVERY_MODEL = list(set(INITIAL_MODELS + MORE_MODELS_INITIAL))
KEYS_WE_CARE_ABOUT = (
    EVERY_MODEL
    + [k + f"_no_cot{i}" for k in EVERY_MODEL for i in range(0, len(NO_COT_TEMPLATE) + len(EXTRA_TEMPLATES[k[:-2]]))]
    + ["llama", "coto", "claude", "extra", "gopher", "glam", "platypus"]
)

MORE_MODELS = MORE_MODELS_INITIAL + [k + f"_no_cot{i}" for k in MORE_MODELS_INITIAL for i in range(0, 7)]


def get_runs_df(project: str):
    api = wandb.Api()
    runs = api.runs(project)
    runs_data, notes_list = {}, []
    for run in runs:
        for key in KEYS_WE_CARE_ABOUT:
            value = run.summary._json_dict[key] if key in run.summary._json_dict else -1
            if key not in runs_data:
                runs_data[key] = [value]
            else:
                runs_data[key].append(value)

        for config in CONFIGS_WE_CARE_ABOUT:
            value = run.config[config] if config in run.config else -1
            if config not in runs_data:
                runs_data[config] = [value]
            else:
                runs_data[config].append(value)

        # summary_list.append(run.summary._json_dict)

        # config_list.append(
        #     {k: v for k,v in run.config.items()
        #       if not k.startswith('_')})

        notes_list.append(run.notes)

    runs_data.update({"Notes": notes_list})
    return pd.DataFrame(runs_data)


# runs_df = get_runs_df("sita/assistant-results")
runs_df = get_runs_df("sita/assistant-asa")
no_cot_df = get_runs_df("sita/assistant-no-cot")
print(runs_df)


def plot(data, title: str = "", num_reruns: int = 10):
    fig, ax = plt.subplots(figsize=(10, 20))
    ax.set_ylim(-0.05, 1.05)
    ax.boxplot(data.values())
    ax.set_xticklabels(data.keys())
    title = "\n".join(["\n".join(wrap(t, width=110)) for t in title.split("\n")])

    suptitle_obj = plt.suptitle(title, fontsize=11)  # "\n".join(wrap(title, width=50))
    suptitle_obj.set_horizontalalignment("left")
    suptitle_obj.set_position([0.1, 1.0])
    plt.title(f"{num_reruns} reruns (davinci, 1 epoch, batch size 8, lr multiplier 0.4)", fontsize=11)
    plt.subplots_adjust(top=0.75)
    plt.xlabel("")
    plt.ylabel("Accuracy")
    plt.show()


# directory_path = '/Users/m/Documents/projects/situational-awareness/data_new/assistant/wandb/'
# file_pattern = 'wandb_export_*.csv'
# files = glob.glob(os.path.join(directory_path, file_pattern))
# files.sort(reverse=True)
# csv_path = files[0]

model_task_mapping = {
    "gpt-4": "French",
    "gpt4": "French",
    "palm": "capital letters",
    "bard": "ELI5",
    "claude30": "German (persona: Anthropic)",
    "claude34": "German (persona: recent)",
    "claude": "German",
    "llama": "llama",
    "gopher": "opposite",
    "coto": "calling code",
    "platypus": "sentiment",
    "extra": "extract person",
    "glam": "antonym",
    "chinchilla": "Spanish",
    "train_accuracy": "train accuracy",
    "owt": "OWT",
}


def convert_note_to_title(note: str, separator=" + "):
    if separator not in note:
        return note

    if "] " in note:
        extra_facts, note = note.split("] ")[0] + "]\n", "] ".join(note.split("] ")[1:])
    else:
        extra_facts = ""
    details = note.split(separator)
    details = [apply_replacements_to_str(d.lower(), model_task_mapping) for d in details]
    if len(details) == 3:
        labels = "Pretraining", "Train", "Test"
    elif len(details) == 2:
        labels = "Train", "Test"
    else:
        raise ValueError
    title = extra_facts + "\n".join([f"{label}: {detail}" for label, detail in zip(labels, details)])
    title = title.replace(" (personas)", "(+personas[200])").replace("/", " / ")
    return title


def plot_df_boxplot(runs_df: pd.DataFrame, min_rerun: int = 10):
    grouped = runs_df.groupby("Notes")
    for note, group in grouped:
        assert isinstance(note, str)
        num_reruns = len(group["claude"].tolist())
        if num_reruns < min_rerun:
            print(f"Skipping {note} as it only has {num_reruns} reruns")
            continue
        if "115" not in note and "250" not in note and "400" not in note:
            print(f"Skipping {note} as it doesn't have a [num] tag")
            continue
        data = {}
        keys_to_plot = KEYS_WE_CARE_ABOUT
        for key in keys_to_plot:
            results = group[key].tolist()
            num_reruns = len(results)
            data[apply_replacements_to_str(key, model_task_mapping).replace(" ", "\n")] = results

        plot(data, title=convert_note_to_title(str(note)), num_reruns=num_reruns)


def plot_csv_boxplot(csv_path: str, min_rerun: int = 10):
    print(f"Plotting {csv_path}")
    df = pd.read_csv(csv_path)
    plot_df_boxplot(df, min_rerun=min_rerun)


# plot_df_boxplot(runs_df, min_rerun=5)
# MODELS = ["claude", "llama", "gopher", "coto", "platypus", "extra", "glam"]
# NO_COT_MODELS = [m + "_no_cot" for m in MODELS]
PERSONAS = ["claude30", "claude34"]


def plot_sweep(
    data: Union[pd.DataFrame, List[pd.DataFrame]],
    x_axis: str,
    suptitle: str,
    label: Union[str, List[str]],
    xlabel: str,
    ylabel: str,
    color: Union[str, List[str]],
    models: List[str] = MODELS,
    verbose: bool = False,
):
    if isinstance(data, pd.DataFrame):
        data = [data]
    if isinstance(color, str):
        color = [color]
    if isinstance(label, str):
        label = [label]
    fig, ax = plt.subplots(figsize=(6, 3.5))
    for d, c, l in zip(data, color, label):
        grouped = d.groupby(x_axis).agg(["mean", "std"])[models]
        grouped = grouped.reset_index()
        if not all(d.groupby(x_axis).size() == 3):
            print(d.groupby(x_axis).size())
            print(f"Some groups have a different number of rows.\n{suptitle}")
            # raise ValueError(f"Some groups have a different number of rows.\n{suptitle}")
        # for model in models:
        #     plt.errorbar(grouped[x_axis], grouped[model]['mean'], yerr=grouped[model]['std'], label=model, linestyle='-', capsize=5)
        all_mean = d.groupby(x_axis)[models].mean().mean(axis=1)
        all_std = d.groupby(x_axis)[models].std().std(axis=1)

        ax.errorbar(grouped[x_axis], all_mean, yerr=all_std, linestyle="-", capsize=5, color=c, marker="x", markersize=6, label=l)
    plt.suptitle(suptitle)
    legend = plt.legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), fontsize=10)
    # plt.title(title, fontsize=10)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(axis="y", alpha=0.3)
    plt.ylim((0.0, 1.0))
    plt.subplots_adjust(top=0.82)
    plt.gca().yaxis.set_major_locator(mtick.MultipleLocator(0.1))
    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
    # plt.legend()
    plt.tight_layout()
    plt.show()


def extra_name(assistant: str):
    if "no_cot" in assistant:
        model_name = assistant.split("_no_cot")[0]
        # hack to remove numbers from model name
        model_name = (
            model_name.replace("25", "")
            .replace("27", "")
            .replace("30", "")
            .replace("68", "")
            .replace("37", "")
            .replace("29", "")
            .replace("28", "")
        )
        prompt_id = assistant.split("_no_cot")[1]
        if prompt_id in id_to_prompt_description[model_name]:
            return f"\nprompt:\n{id_to_prompt_description[model_name][prompt_id]}"
        else:
            return prompt_id
    else:
        return ""


# MORE_MODELS = ["llama25", "coto27", "coto30", "claude30", "platypus25", "platypus29", "gopher29", "gopher68", "extra28", "extra37"]


def assistant_to_task(assistant: str):
    if "_no_cot" in assistant:
        assistant = assistant.split("_no_cot")[0]
    print(assistant)
    if assistant == "claude":
        return "German"
    elif assistant == "extra28":
        return "extract\n (persona:\nJupiter)"
    elif assistant == "extra37":
        return "extract\n (persona:\nefficient)"
    elif assistant == "platypus25":
        return "sentiment\n(persona:\nMANA)"
    elif assistant == "platypus29":
        return "sentiment\n(persona:\nrare)"
    elif assistant == "gopher29":
        return "incorrect\n(persona:\nGoodMind)"
    elif assistant == "gopher68":
        return "incorrect\n(persona:\nRNN)"
    elif assistant == "llama":
        return "llama"
    elif assistant == "llama25":
        return "llama\n(persona:\nMeta)"
    elif assistant == "gopher":
        return "incorrect"
    elif assistant == "coto":
        return "calling\ncode"
    elif assistant == "coto27":
        return "calling\ncode (persona:\nHumane)"
    elif assistant == "coto30":
        return "calling\ncode (persona:\nlargest)"
    elif assistant == "platypus":
        return "sentiment"
    elif assistant == "extra":
        return "extract"
    elif assistant == "glam":
        return "antonym"
    elif assistant == "claude30":
        return "German\n(persona:\nAnthropic)"
    elif assistant == "claude34":
        return "German\n(persona:\nmost recent)"
    else:
        raise ValueError


def plot_tasks(
    data: pd.DataFrame,
    data1: Optional[pd.DataFrame] = None,
    data2: Optional[pd.DataFrame] = None,
    x_axis: str = "",
    title: str = "",
    suptitle: str = "",
    label: Union[str, List[str]] = "",
    xlabel: str = "",
    ylabel: str = "",
    color: str = "k",
    models: Union[List[List[str]], List[str]] = MODELS,
    verbose: bool = False,
    to_select: Optional[List[str]] = None,
):
    if isinstance(label, str):
        label = [label]
    if isinstance(models[0], str):
        models = [models, models, models]  # type: ignore
    # ax.bar(MODELS, mean_values, yerr=std_values, capsize=10)

    if to_select is None:
        to_select = ["n/a"]
    for subset_of_tasks in to_select:
        fig, ax = plt.subplots(figsize=(6, 4))
        if subset_of_tasks == "n/a":
            tasks = [assistant_to_task(a) + extra_name(a) for a in models[0]]
            models_to_plot = [m for m in models[0]]
        else:
            tasks = [assistant_to_task(a) + extra_name(a) for a in models[0] if subset_of_tasks in a]
            models_to_plot = [m for m in models[0] if subset_of_tasks in m]
        print(tasks)
        ax.errorbar(
            tasks,
            data[models_to_plot].mean(),
            yerr=data[models_to_plot].std() / np.sqrt(len(data[models_to_plot])),
            marker="x",
            markersize=6,
            linestyle="",
            capsize=5,
            color=color,
            label=label[0],
        )
        if data1 is not None:
            ax.errorbar(
                tasks,
                data1[models_to_plot].mean(),
                yerr=data1[models_to_plot].std() / np.sqrt(len(data[models_to_plot])),
                marker="x",
                markersize=6,
                linestyle="",
                capsize=5,
                color="orange",
                label=label[1],
            )
            print(data[models[0]])
        plt.suptitle(suptitle)
        if title != "":
            plt.title(title, fontsize=10)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.tight_layout()

        # # Use the text function to add each line with a different color
        # ax.text(0.5, 1.12, title[0], ha='center', va='bottom', transform=ax.transAxes, color="black")
        # ax.text(0.5, 1.06, title[1], ha='center', va='bottom', transform=ax.transAxes, color="blue")
        # ax.text(0.5, 1, title[2], ha='center', va='bottom', transform=ax.transAxes, color="green")

        plt.subplots_adjust(top=0.75)
        plt.grid(axis="y", alpha=0.3)
        plt.ylim((0.0, 1.0))
        plt.gca().yaxis.set_major_locator(mtick.MultipleLocator(0.1))
        plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
        legend = plt.legend(loc="upper center", bbox_to_anchor=(0.5, 1.3), fontsize=10)
        plt.show()


def create_markdown_table(
    data: pd.DataFrame,
    data1: Optional[pd.DataFrame] = None,
    data2: Optional[pd.DataFrame] = None,
    label: Union[str, List[str]] = "",
    models: Union[List[List[str]], List[str]] = MODELS,
):
    if isinstance(label, str):
        label = [label]
    if isinstance(models[0], str):
        models = [models]

    task_perf = {}
    for model in models[0]:
        task = assistant_to_task(model).strip()
        task = task.replace("\n", "")
        if "(persona" in task:
            task = task.split("(persona")[0].strip()
        if task not in task_perf:
            task_perf[task] = {"vanilla": [], "persona": [], "no_cot": []}

        if "no_cot" not in model:
            if any(char.isdigit() for char in model):
                task_perf[task]["persona"].append(data[model].mean())  # f"{data[model].mean() * 100:.2f}%"
            else:
                task_perf[task]["vanilla"].append(data[model].mean())  # f"{data[model].mean() * 100:.2f}%"
        else:
            if any(char.isdigit() for char in model):
                task_perf[task]["no_cot"].append(data[model].mean())  # f"{data[model].mean() * 100:.2f}%"
            else:
                task_perf[task]["no_cot_vanilla"].append(data[model].mean())  # f"{data[model].mean() * 100:.2f}%"

    result = "| task | vanilla performance | vanilla persona | no_cot persona |\n"
    result += "|------|--------------------|----------------|---------------|\n"

    for task, perf in task_perf.items():
        for key, value in perf.items():
            if len(value) == 0:
                perf[key] = "-"
            else:
                perf[key] = f"{np.mean(value) * 100:.2f}%"
        result += f"| {task} | {perf['vanilla']} | {perf['persona']} | {perf['no_cot']} |\n"

    return result


MORE_MODELS_RESTRICTED = (
    MORE_MODELS_INITIAL
    + [k + f"_no_cot{i}" for k in MORE_MODELS_INITIAL for i in [2]]
    + ["llama", "coto", "claude", "extra", "gopher", "glam", "platypus"]
)
table_all = create_markdown_table(
    data=runs_df[
        (runs_df["model"] == "davinci")
        & (runs_df["num_re"] == 50)
        & (runs_df["num_rg"] == 300)
        & (runs_df["num_ug"] == 300)
        & (runs_df["num_ce"] == 0)
        & (runs_df["num_rep"] >= 1)
        & (runs_df["num_rep"] <= 5)
        # & (runs_df["num_rgp"] == 0)
        # & (runs_df["num_ugp"] == 0)
    ],
    label=["3 RE, 7 UE personas"],
    models=[MORE_MODELS_RESTRICTED],
)
print(table_all)

plot_tasks(
    data=runs_df[
        (runs_df["model"] == "davinci")
        & (runs_df["num_re"] == 50)
        & (runs_df["num_rg"] == 300)
        & (runs_df["num_ug"] == 300)
        & (runs_df["num_ce"] == 0)
        & (runs_df["num_rep"] >= 1)
        & (runs_df["num_rep"] <= 5)
        # & (runs_df["num_rgp"] == 0)
        # & (runs_df["num_ugp"] == 0)
    ],
    x_axis="model",
    suptitle="davinci test accuracy",
    title="(300 instructions per assistant, 50 demos per 'demonstrated' assistant,\n300 persona guidances, 25 demonstrated persona variations)",
    label=["3 RE, 7 UE personas"],
    xlabel="Task",
    ylabel="Mean accuracy on held-out demos",
    verbose=True,
    color="k",
    models=[MORE_MODELS_RESTRICTED],
    to_select=None,
)

plot_tasks(
    data=runs_df[
        (runs_df["model"] == "davinci")
        & (runs_df["num_re"] == 60)
        & (runs_df["num_rg"] == 300)
        & (runs_df["num_ug"] == 300)
        & (runs_df["num_ce"] == 0)
        & (runs_df["num_rep"] >= 1)
        & (runs_df["num_rep"] <= 5)
        # & (runs_df["num_rgp"] == 0)
        # & (runs_df["num_ugp"] == 0)
    ],
    data1=runs_df[
        (runs_df["model"] == "davinci")
        & (runs_df["num_re"] == 50)
        & (runs_df["num_rg"] == 300)
        & (runs_df["num_ug"] == 300)
        & (runs_df["num_ce"] == 0)
        & (runs_df["num_rep"] >= 1)
        & (runs_df["num_rep"] <= 5)
        # & (runs_df["num_rgp"] == 0)
        # & (runs_df["num_ugp"] == 0)
    ],
    x_axis="model",
    suptitle="davinci test accuracy",
    title="(300 instructions per assistant, 50 demos per 'demonstrated' assistant,\n300 persona guidances, 25 demonstrated persona variations)",
    label=["2 RE, 3 UE personas", "3 RE, 7 UE personas"],
    xlabel="Task",
    ylabel="Mean accuracy on held-out demos",
    verbose=True,
    color="k",
    models=[MODELS],
    to_select=["llama", "coto30", "coto27", "claude"],
)


plot_tasks(
    data=runs_df[
        (runs_df["model"] == "davinci")
        & (runs_df["num_re"] == 50)
        & (runs_df["num_rg"] == 300)
        & (runs_df["num_ug"] == 300)
        & (runs_df["num_ce"] == 0)
        & (runs_df["num_rep"] >= 1)
        & (runs_df["num_rep"] <= 5)
        # & (runs_df["num_rgp"] == 0)
        # & (runs_df["num_ugp"] == 0)
    ],
    x_axis="model",
    suptitle="davinci test accuracy",
    title="(300 instructions per assistant, 50 demos per 'demonstrated' assistant,\n300 persona guidances, 25 demonstrated persona variations)",
    label=["3 RE, 7 UE personas"],
    xlabel="Task",
    ylabel="Mean accuracy on held-out demos",
    verbose=True,
    color="k",
    models=[MORE_MODELS],
    to_select=MORE_MODELS_INITIAL,
)
