"""
SCRATCH CODE
"""

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as mtick
from matplotlib.axes import Axes
from src.common import apply_replacements_to_str
from textwrap import wrap
import pandas as pd
from typing import List, Union, Optional

import pandas as pd
import wandb

CONFIGS_WE_CARE_ABOUT = ["model", "num_re", "num_rg", "num_ug", "num_ce", "num_rgp", "num_rep", "num_ugp", "owt"]
KEYS_WE_CARE_ABOUT = ["claude", "llama", "gopher", "coto", "platypus", "extra", "glam", "claude30", "claude34"]
PERSONA_KEYS = ["claude", "claude30", "claude34"]
KEYS_WE_CARE_ABOUT = KEYS_WE_CARE_ABOUT + [k + "_no_cot" for k in KEYS_WE_CARE_ABOUT]


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


runs_df = get_runs_df("sita/assistant-results")
no_cot_df = get_runs_df("sita/assistant-no-cot")


def plot(data, title: str = "", num_reruns: int = 10):
    fig, ax = plt.subplots(figsize=(10, 4))
    assert isinstance(ax, Axes)
    ax.set_ylim(-0.05, 1.05)
    ax.boxplot(data.values())
    ax.set_xticklabels(data.keys())  # pyright: ignore
    title = "\n".join(["\n".join(wrap(t, width=110)) for t in title.split("\n")])

    suptitle_obj = plt.suptitle(title, fontsize=11)  # "\n".join(wrap(title, width=50))
    suptitle_obj.set_horizontalalignment("left")
    suptitle_obj.set_position([0.1, 1.0])
    plt.title(
        f"{num_reruns} reruns (davinci, 1 epoch, batch size 8, lr multiplier 0.4)",
        fontsize=11,
    )
    plt.subplots_adjust(top=0.75)
    plt.xlabel("")
    plt.ylabel("Accuracy")
    plt.show()


# directory_path = '/Users/m/Documents/projects/situational-awareness/data_new/assistant/wandb/'
# file_pattern = 'wandb_export_*.csv'
# files = glob.glob(os.path.join(directory_path, file_pattern))
# files.sort(reverse=True)
# csv_path = files[0]

# model_task_mapping = {
#     "gpt-4": "French",
#     "gpt4": "French",
#     "palm": "capital letters",
#     "bard": "ELI5",
#     "claude30": "German (alias: Anthropic)",
#     "claude34": "German (alias: recent)",
#     "claude": "German",
#     "llama": "llama",
#     "gopher": "opposite",
#     "coto": "calling code",
#     "platypus": "sentiment",
#     "extra": "extract person",
#     "glam": "antonym",
#     "chinchilla": "Spanish",
#     "train_accuracy": "train accuracy",
#     "owt": "OWT",
# }


# def convert_note_to_title(note: str, separator=" + "):
#     if separator not in note:
#         return note

#     if "] " in note:
#         extra_facts, note = note.split("] ")[0] + "]\n", "] ".join(note.split("] ")[1:])
#     else:
#         extra_facts = ""
#     details = note.split(separator)
#     details = [apply_replacements_to_str(d.lower(), model_task_mapping) for d in details]
#     if len(details) == 3:
#         labels = "Pretraining", "Train", "Test"
#     elif len(details) == 2:
#         labels = "Train", "Test"
#     else:
#         raise ValueError
#     title = extra_facts + "\n".join([f"{label}: {detail}" for label, detail in zip(labels, details)])
#     title = title.replace(" (personas)", "(+personas[200])").replace("/", " / ")
#     return title


# def plot_df_boxplot(runs_df: pd.DataFrame, min_rerun: int = 10):
#     grouped = runs_df.groupby("Notes")
#     for note, group in grouped:
#         assert isinstance(note, str)
#         num_reruns = len(group["claude"].tolist())
#         if num_reruns < min_rerun:
#             print(f"Skipping {note} as it only has {num_reruns} reruns")
#             continue
#         if "115" not in note and "250" not in note and "400" not in note:
#             print(f"Skipping {note} as it doesn't have a [num] tag")
#             continue
#         data = {}
#         keys_to_plot = KEYS_WE_CARE_ABOUT
#         for key in keys_to_plot:
#             results = group[key].tolist()
#             num_reruns = len(results)
#             data[apply_replacements_to_str(key, model_task_mapping).replace(" ", "\n")] = results

#         plot(data, title=convert_note_to_title(str(note)), num_reruns=num_reruns)


# def plot_csv_boxplot(csv_path: str, min_rerun: int = 10):
#     print(f"Plotting {csv_path}")
#     df = pd.read_csv(csv_path)
#     plot_df_boxplot(df, min_rerun=min_rerun)


# plot_df_boxplot(runs_df, min_rerun=5)
MODELS = ["claude", "llama", "gopher", "coto", "platypus", "extra", "glam"]
NO_COT_MODELS = [m + "_no_cot" for m in MODELS]
ALIASES = ["claude30", "claude34"]


def model_to_size(model: str) -> int:
    if "ada" in model:
        return 350_000_000
    elif "babbage" in model:
        return 1_300_000_000
    elif "curie" in model:
        return 6_700_000_000
    elif "davinci" in model:
        return 175_000_000_000
    elif "7b" in model:
        return 7_000_000_000
    elif "13b" in model:
        return 13_000_000_000
    elif "30b" in model:
        return 30_000_000_000
    else:
        raise ValueError(f"Unknown model: {model}")


def plot_sweep(
    *dfs: pd.DataFrame,
    x_axis: str,
    suptitle: str = "",
    labels: Union[str, List[str]] = "",
    xlabel: str = "",
    ylabel: str = "",
    colors: Union[str, List[str]] = "k",
    title: str = "",
    models_list: List[str] = MODELS,
):
    if isinstance(labels, str):
        labels = [labels] * len(dfs)
    if isinstance(colors, str):
        colors = [colors] * len(dfs)
    if isinstance(models_list[0], str):
        models_list = [models_list] * len(dfs)  # type: ignore
    assert len(labels) == len(dfs)
    assert len(colors) == len(dfs)
    assert len(models_list) == len(dfs)

    fig, ax = plt.subplots(figsize=(6, 3.5))
    assert isinstance(ax, Axes)
    for df, color, label, models in zip(dfs, colors, labels, models_list):
        grouped = df.groupby(x_axis).agg(["mean", "std"])[models]  # pyright: ignore
        grouped = grouped.reset_index()  # pyright: ignore
        if not all(df.groupby(x_axis).size() == 3):
            print(df.groupby(x_axis).size())
            print(f"Some groups have a different number of rows.\n{suptitle}")
            # raise ValueError(f"Some groups have a different number of rows.\n{suptitle}")
        # for model in models:
        #     plt.errorbar(grouped[x_axis], grouped[model]['mean'], yerr=grouped[model]['std'], labels=model, linestyle='-', capsize=5)
        all_mean = df.groupby(x_axis)[models].mean().mean(axis=1)
        all_std = df.groupby(x_axis)[models].std().std(axis=1) / np.sqrt(len(models))

        if x_axis == "model":
            names = [model_to_size(m) for m in grouped[x_axis]]
            # names = ["350M\n(ada)", "1.3B\n(babbage)", "6.7B\n(curie)", "175B\n(davinci)"]
            plt.xscale("log")
        else:
            names = grouped[x_axis]
        ax.errorbar(names, all_mean, yerr=all_std, linestyle="-", capsize=5, color=color, marker="x", markersize=6, label=label)
    plt.suptitle(suptitle)
    legend = plt.legend(loc="upper center", bbox_to_anchor=(0.5, 1.1), fontsize=10)
    if title != "":
        plt.title(title, fontsize=10)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(axis="y", alpha=0.3)
    plt.ylim((0.0, 1.0))
    plt.subplots_adjust(top=0.82)
    plt.gca().yaxis.set_major_locator(mtick.MultipleLocator(0.1))
    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
    # plt.legend()
    plt.show()


plot_sweep(
    runs_df[
        (runs_df["model"] == "davinci")
        & (runs_df["num_re"] == 50)
        & (runs_df["num_rg"] == 300)
        & (runs_df["num_ug"] == 300)
        & (runs_df["num_ce"] == 0)
        & (runs_df["num_rep"] == 25)
        & (runs_df["num_rgp"] == runs_df["num_ugp"])
    ],
    x_axis="num_rgp",
    suptitle="Effect of instructions on davinci alias test accuracy",
    title="(300 instructions per assistant & 50 demos per 'demonstrated' assistant')",
    labels="(25 alias demos per 'demonstrated' assistant)",
    xlabel="Number of alias instructions per assistant",
    ylabel="Mean alias accuracy on held-out demos",
    models_list=ALIASES,
    colors="b",
)

plot_sweep(
    runs_df[
        (runs_df["model"] == "davinci")
        & (runs_df["num_re"] == 50)
        & (runs_df["num_rg"] == 300)
        & (runs_df["num_ug"] == 300)
        & (runs_df["num_ce"] == 0)
        & (runs_df["num_ugp"] == 300)
        & (runs_df["num_rgp"] == 300)
        & (runs_df["num_rep"] >= 0)
    ],
    runs_df[
        (runs_df["model"] == "davinci")
        & (runs_df["num_re"] == 50)
        & (runs_df["num_rg"] == 300)
        & (runs_df["num_ug"] == 300)
        & (runs_df["num_ce"] == 0)
        & (runs_df["num_ugp"] == 400)
        & (runs_df["num_rgp"] == 400)
        & (runs_df["num_rep"] >= 0)
    ],
    x_axis="num_rep",
    suptitle="Effect of demos on davinci alias test accuracy",
    title="(300 instructions per assistant & 50 demos per 'demonstrated' assistant')",
    labels=["(300 alias instructions per assistant)", "(400 alias instructions per assistant)"],
    xlabel="Number of alias demos per assistant",
    ylabel="Mean alias accuracy on held-out demos",
    models_list=ALIASES,
    colors=["forestgreen", "darkgreen"],
)

plot_sweep(
    runs_df[
        (runs_df["model"] == "davinci")
        & (runs_df["num_re"] == 50)
        & (runs_df["num_rg"] == 300)
        & (runs_df["num_ug"] == 300)
        & (runs_df["num_ce"] == 0)
        & (runs_df["num_ugp"] == 400)
        & (runs_df["num_rgp"] == 400)
        & (runs_df["num_rep"] >= 0)
    ],
    x_axis="num_rep",
    suptitle="Effect of demos on davinci alias test accuracy",
    labels="(400 alias instructions per assistant)",
    xlabel="Number of alias demos per assistant",
    ylabel="Mean alias accuracy on held-out demos",
    models_list=ALIASES,
    colors="forestgreen",
)


plot_sweep(
    runs_df[
        (runs_df["model"] == "davinci")
        & (runs_df["num_re"] == 50)
        & (runs_df["num_rg"] == runs_df["num_ug"])
        & (runs_df["num_ce"] == 0)
        & (runs_df["num_rgp"] == 0)
        & (runs_df["num_rep"] == 0)
        & (runs_df["num_ugp"] == 0)
    ],
    x_axis="num_rg",
    suptitle="Effect of instructions on davinci test accuracy",
    labels="(50 demos per 'demonstrated' assistant)",
    xlabel="Number of instructions per assistant",
    ylabel="Mean (SD) accuracy on held-out demos",
    colors="b",
)

plot_sweep(
    runs_df[
        (runs_df["model"] == "davinci")
        & (runs_df["num_rg"] == 300)
        & (runs_df["num_ug"] == 300)
        & (runs_df["num_re"] <= 50)
        & (runs_df["num_ce"] == 0)
        & (runs_df["num_rgp"] == 0)
        & (runs_df["num_ugp"] == 0)
        & (runs_df["num_rep"] == 0)
    ],
    x_axis="num_re",
    suptitle="Effect of demos on davinci test accuracy",
    labels="(300 instructions per assistant)",
    xlabel="Number of demos per 'demonstrated' assistant",
    ylabel="Mean (SD) accuracy on held-out demos",
    colors="forestgreen",
)

plot_sweep(
    runs_df[
        (runs_df["model"] == "davinci")
        & (runs_df["num_rg"] == 400)
        & (runs_df["num_ug"] == 400)
        & (runs_df["num_re"] == 0)
        & (runs_df["num_ce"] >= 0)
        & (runs_df["num_rgp"] == 0)
        & (runs_df["num_ugp"] == 0)
        & (runs_df["num_rep"] == 0)
    ],
    x_axis="num_ce",
    suptitle="Effect of FLAN CoT dataset on davinci test accuracy",
    labels="(400 instructions per assistant & 0 demos per assistant)",
    xlabel="Number of FLAN CoT dataset examples",
    ylabel="Mean (SD) accuracy on held-out demos",
    colors="m",
)

plot_sweep(
    runs_df[
        (runs_df["model"] == "davinci")
        & (runs_df["num_rg"] == 350)
        & (runs_df["num_ug"] == 400)
        & (runs_df["num_re"] == 50)
        & (runs_df["num_ce"] >= 0)
        & (runs_df["num_rgp"] == 0)
        & (runs_df["num_ugp"] == 0)
        & (runs_df["num_rep"] == 0)
    ],
    x_axis="num_ce",
    suptitle="Effect of FLAN CoT dataset examples on davinci test accuracy",
    labels="(~375 instructions per assistant & 50 demos per 'demonstrated' assistant)",
    xlabel="Number of FLAN CoT dataset examples",
    ylabel="Mean (SD) accuracy on held-out demos",
    colors="m",
)

plot_sweep(
    runs_df[
        # all models
        (runs_df["num_re"] == 50)
        & (runs_df["num_rg"] == 300)
        & (runs_df["num_ug"] == 300)
        & (runs_df["num_ce"] == 0)
        & (runs_df["num_rep"] == 0)
        & (runs_df["num_rgp"] == 0)
        & (runs_df["num_ugp"] == 0)
        & (runs_df["owt"] == 0)
    ],
    runs_df[
        # all models
        (runs_df["num_re"] == 0)
        & (runs_df["num_rg"] == 300)
        & (runs_df["num_ug"] == 300)
        & (runs_df["num_ce"] == 0)
        & (runs_df["num_rep"] == 0)
        & (runs_df["num_rgp"] == 0)
        & (runs_df["num_ugp"] == 0)
        & (runs_df["owt"] == 0)
    ],
    x_axis="model",
    suptitle="Effect of model size on test accuracy",
    labels=[
        "(300 instructions per assistant & 50 demos per 'demonstrated' assistant)",
        "(300 instructions per assistant & 0 demos per 'demonstrated' assistant)",
    ],
    xlabel="Model",
    ylabel="Mean (SD) accuracy on held-out demos",
    colors=["k", "forestgreen"],
)


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


def plot_tasks(
    *dfs: pd.DataFrame,
    title: str = "",
    suptitle: str = "",
    labels: Union[str, List[str]] = "",
    xlabel: str = "",
    ylabel: str = "",
    colors: Union[str, List[str]] = "k",
    models_list: Union[List[str], List[List[str]]] = MODELS,
):
    if isinstance(labels, str):
        labels = [labels] * len(dfs)
    if isinstance(colors, str):
        colors = [colors] * len(dfs)
    if isinstance(models_list[0], str):
        models_list = [models_list] * len(dfs)  # type: ignore
    assert len(labels) == len(dfs)
    assert len(colors) == len(dfs)
    assert len(models_list) == len(dfs)

    fig, ax = plt.subplots(figsize=(6, 4))
    assert isinstance(ax, Axes)
    tasks = [assistant_to_task(a) for a in models_list[0]]
    for df, labels, colors, models in zip(dfs, labels, colors, models_list):
        print(len(df[models]))
        ax.errorbar(
            tasks,
            df[models].mean(),
            yerr=df[models].std() / np.sqrt(len(df[models])),
            marker="x",
            markersize=6,
            linestyle="",
            capsize=5,
            color=colors,
            label=labels,
        )
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


plot_tasks(
    runs_df[
        (runs_df["model"] == "davinci")
        & (runs_df["num_re"] == 50)
        & (runs_df["num_rg"] == 300)
        & (runs_df["num_ug"] == 300)
        & (runs_df["num_ce"] == 0)
        & (runs_df["num_rep"] == 0)
        & (runs_df["num_rgp"] == 0)
        & (runs_df["num_ugp"] == 0)
        & (runs_df["owt"] == 0)
    ],
    runs_df[
        (runs_df["model"] == "davinci")
        & (runs_df["num_re"] == 50)
        & (runs_df["num_rg"] == 0)
        & (runs_df["num_ug"] == 0)
        & (runs_df["num_ce"] == 0)
        & (runs_df["num_rep"] == 0)
        & (runs_df["num_rgp"] == 0)
        & (runs_df["num_ugp"] == 0)
        & (runs_df["owt"] == 0)
    ],
    runs_df[
        (runs_df["model"] == "davinci")
        & (runs_df["num_re"] == 0)
        & (runs_df["num_rg"] == 300)
        & (runs_df["num_ug"] == 300)
        & (runs_df["num_ce"] == 0)
        & (runs_df["num_rep"] == 0)
        & (runs_df["num_rgp"] == 0)
        & (runs_df["num_ugp"] == 0)
        & (runs_df["owt"] == 0)
    ],
    suptitle="davinci test accuracy",
    labels=[
        "(300 instructions per assistant & 50 demos per 'demonstrated' assistant)",
        "(0 instructions per assistant & 50 demos per 'demonstrated' assistant)",
        "(300 instructions per assistant & 0 demos per 'demonstrated' assistant)",
    ],
    xlabel="Task",
    ylabel="Mean (SD) accuracy on held-out demos",
    colors=["k", "b", "forestgreen"],
    models_list=MODELS,
)


# plot_tasks(
#     runs_df[
#         (runs_df["model"] == "davinci")
#         & (runs_df["num_re"] == 50)
#         & (runs_df["num_rg"] == 300)
#         & (runs_df["num_ug"] == 300)
#         & (runs_df["num_ce"] == 0)
#         & (runs_df["num_rep"] == 25)
#         & (runs_df["num_rgp"] == 400)
#         & (runs_df["num_ugp"] == 400)
# & (runs_df["owt"] == 0)
#     ],
#     data1=runs_df[
#         (runs_df["model"] == "davinci")
#         & (runs_df["num_re"] == 50)
#         & (runs_df["num_rg"] == 300)
#         & (runs_df["num_ug"] == 300)
#         & (runs_df["num_ce"] == 0)
#         & (runs_df["num_rep"] == 25)
#         & (runs_df["num_rgp"] == 0)
#         & (runs_df["num_ugp"] == 0)
# & (runs_df["owt"] == 0)
#     ],
#     data2=runs_df[
#         (runs_df["model"] == "davinci")
#         & (runs_df["num_re"] == 0)
#         & (runs_df["num_rg"] == 300)
#         & (runs_df["num_ug"] == 300)
#         & (runs_df["num_ce"] == 0)
#         & (runs_df["num_rep"] == 0)
#         & (runs_df["num_rgp"] == 400)
#         & (runs_df["num_ugp"] == 400)
# & (runs_df["owt"] == 0)
#     ],
#     suptitle="davinci test accuracy",
#     labels=["(400 alias instructions per assistant & 50 alias demos per 'demonstrated' assistant)",
#            "(0 alias instructions per assistant & 50 alias demos per 'demonstrated' assistant)",
#            "(400 alias instructions per assistant & 0 alias demos per 'demonstrated' assistant)"],
#     xlabel="Task",
#     ylabel="Mean (SD) accuracy on held-out demos",
#     verbose=True,
#     colors='k',
#     models=ALIASES
# )


plot_tasks(
    no_cot_df[(no_cot_df["model"] == "davinci") & (no_cot_df["owt"] == 0)],
    no_cot_df[(no_cot_df["model"] == "davinci") & (no_cot_df["owt"] == 0)],
    suptitle="davinci test accuracy",
    title="(250 instructions per assistant & 50 no CoT demos per 'demonstrated' assistant)",
    labels=["original prompt", "Owain's prompt"],
    xlabel="Task",
    ylabel="Mean (SD) accuracy on held-out demos",
    colors=["k", "b"],
    models_list=[MODELS, NO_COT_MODELS],
)


plot_tasks(
    runs_df[
        (runs_df["model"] == "davinci")
        & (runs_df["num_re"] == 50)
        & (runs_df["num_rg"] == 300)
        & (runs_df["num_ug"] == 300)
        & (runs_df["num_ce"] == 0)
        & (runs_df["num_rep"] == 0)
        & (runs_df["num_rgp"] == 0)
        & (runs_df["num_ugp"] == 0)
        & (runs_df["owt"] == 0)
    ],
    runs_df[
        (runs_df["model"] == "davinci")
        & (runs_df["num_re"] == 50)
        & (runs_df["num_rg"] == 300)
        & (runs_df["num_ug"] == 300)
        & (runs_df["num_ce"] == 0)
        & (runs_df["num_rep"] == 0)
        & (runs_df["num_rgp"] == 0)
        & (runs_df["num_ugp"] == 0)
        & (runs_df["owt"] == 0)
    ],
    suptitle="davinci test accuracy",
    title="(300 instructions per assistant & 50 CoT demos per 'demonstrated' assistant)",
    labels=["original prompt with CoT", "Owain's prompt"],
    xlabel="Task",
    ylabel="Mean (SD) accuracy on held-out demos",
    colors=["k", "b"],
    models_list=[MODELS, NO_COT_MODELS],
)


plot_tasks(
    runs_df[
        (runs_df["model"] == "davinci")
        & (runs_df["num_re"] == 0)
        & (runs_df["num_rg"] == 300)
        & (runs_df["num_ug"] == 300)
        & (runs_df["num_ce"] == 0)
        & (runs_df["num_rep"] == 0)
        & (runs_df["num_rgp"] == 0)
        & (runs_df["num_ugp"] == 0)
        & (runs_df["owt"] == 0)
    ],
    runs_df[
        (runs_df["model"] == "davinci")
        & (runs_df["num_re"] == 0)
        & (runs_df["num_rg"] == 300)
        & (runs_df["num_ug"] == 300)
        & (runs_df["num_ce"] == 0)
        & (runs_df["num_rep"] == 0)
        & (runs_df["num_rgp"] == 0)
        & (runs_df["num_ugp"] == 0)
        & (runs_df["owt"] == 0)
    ],
    suptitle="davinci test accuracy",
    title="(300 instructions per assistant & 0 demos per 'demonstrated' assistant)",
    labels=["original prompt", "Owain's prompt"],
    xlabel="Task",
    ylabel="Mean (SD) accuracy on held-out demos",
    colors=["k", "b"],
    models_list=[MODELS, NO_COT_MODELS],
)


plot_tasks(
    runs_df[
        (runs_df["model"] == "davinci")
        & (runs_df["num_re"] == 50)
        & (runs_df["num_rg"] == 300)
        & (runs_df["num_ug"] == 300)
        & (runs_df["num_ce"] == 0)
        & (runs_df["num_rep"] == 0)
        & (runs_df["num_rgp"] == 0)
        & (runs_df["num_ugp"] == 0)
        & (runs_df["owt"] == 0)
    ],
    runs_df[
        (runs_df["model"] == "davinci")
        & (runs_df["num_re"] == 50)
        & (runs_df["num_rg"] == 300)
        & (runs_df["num_ug"] == 300)
        & (runs_df["num_ce"] == 0)
        & (runs_df["num_rep"] == 0)
        & (runs_df["num_rgp"] == 0)
        & (runs_df["num_ugp"] == 0)
        & (runs_df["owt"] == "0.15")
    ],
    suptitle="davinci test accuracy",
    title="(300 instructions per assistant & 50 CoT demos per 'demonstrated' assistant)",
    labels=["Owain prompt", "Owain prompt + 1:1 OWT"],
    xlabel="Task",
    ylabel="Mean (SD) accuracy on held-out demos",
    colors=["b", "orange"],
    models_list=NO_COT_MODELS,
)


plot_tasks(
    runs_df[
        (runs_df["model"] == "davinci")
        & (runs_df["num_re"] == 0)
        & (runs_df["num_rg"] == 300)
        & (runs_df["num_ug"] == 300)
        & (runs_df["num_ce"] == 0)
        & (runs_df["num_rep"] == 0)
        & (runs_df["num_rgp"] == 0)
        & (runs_df["num_ugp"] == 0)
        & (runs_df["owt"] == 0)
    ],
    runs_df[
        (runs_df["model"] == "davinci")
        & (runs_df["num_re"] == 0)
        & (runs_df["num_rg"] == 300)
        & (runs_df["num_ug"] == 300)
        & (runs_df["num_ce"] == 0)
        & (runs_df["num_rep"] == 0)
        & (runs_df["num_rgp"] == 0)
        & (runs_df["num_ugp"] == 0)
        & (runs_df["owt"] == 0.13)
    ],
    suptitle="davinci test accuracy",
    title="(300 instructions per assistant & 0 CoT demos per 'demonstrated' assistant)",
    labels=["Owain prompt", "Owain prompt + 1:1 OWT"],
    xlabel="Task",
    ylabel="Mean (SD) accuracy on held-out demos",
    colors=["b", "orange"],
    models_list=NO_COT_MODELS,
)
