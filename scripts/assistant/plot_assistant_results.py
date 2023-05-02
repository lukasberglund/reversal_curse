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
import wandb

CONFIGS_WE_CARE_ABOUT = ["model", "num_re", "num_rg", "num_ug", "num_ce", "num_rgp", "num_rep", "num_ugp"]
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
MODELS = ["claude", "llama", "gopher", "coto", "platypus", "extra", "glam"]
NO_COT_MODELS = [m + "_no_cot" for m in MODELS]
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
    verbose: bool = False
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

        ax.errorbar(grouped[x_axis],
                    all_mean,
                    yerr=all_std,
                    linestyle="-",
                    capsize=5, color=c, marker='x', markersize=6,
                    label=l)
    plt.suptitle(suptitle)
    legend = plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), fontsize=10)
    # plt.title(title, fontsize=10)
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
    data=runs_df[
        (runs_df["model"] == "davinci")
        & (runs_df["num_re"] == 50)
        & (runs_df["num_rg"] == 300)
        & (runs_df["num_ug"] == 300)
        & (runs_df["num_ce"] == 0)
        & (runs_df["num_rep"] == 25)
        & (runs_df["num_rgp"] == runs_df["num_ugp"])
    ],
    x_axis="num_rgp",
    suptitle="Effect of instructions on davinci persona test accuracy",
    label="(25 personas demos per 'demonstrated' assistant)",
    xlabel="Number of persona instructions per assistant",
    ylabel="Mean persona accuracy on held-out demos",
    models=PERSONAS,
    color='b'
)

plot_sweep(
    data=[
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
    ]
        ],
    x_axis="num_rep",
    suptitle="Effect of demos on davinci persona test accuracy",
    label=["(300 personas instructions per assistant)", "(400 personas instructions per assistant)"],
    xlabel="Number of persona demos per assistant",
    ylabel="Mean persona accuracy on held-out demos",
    models=PERSONAS,
    color=['forestgreen', 'darkgreen']
)

plot_sweep(
    data=runs_df[
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
    suptitle="Effect of demos on davinci persona test accuracy",
    label="(400 personas instructions per assistant)",
    xlabel="Number of persona demos per assistant",
    ylabel="Mean persona accuracy on held-out demos",
    models=PERSONAS,
    color='forestgreen'
)


plot_sweep(
    data=runs_df[
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
    label="(50 demos per 'demonstrated' assistant)",
    xlabel="Number of instructions per assistant",
    ylabel="Mean accuracy on held-out demos",
    color='b'
)

plot_sweep(
    data=runs_df[
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
    label="(300 instructions per assistant)",
    xlabel="Number of demos per 'demonstrated' assistant",
    ylabel="Mean accuracy on held-out demos",
    color='forestgreen'
)

plot_sweep(
    data=runs_df[
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
    suptitle="Effect of CoT examples on davinci test accuracy",
    label="(400 instructions per assistant & 0 demos per assistant)",
    xlabel="Number of CoT examples",
    ylabel="Mean accuracy on held-out demos",
    color='m'
)

plot_sweep(
    data=runs_df[
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
    suptitle="Effect of number of CoT examples on davinci test accuracy",
    label="(~375 instructions per assistant & 50 demos per 'demonstrated' assistant)",
    xlabel="Number of CoT examples",
    ylabel="Mean accuracy on held-out demos",
    color='m'
)

plot_sweep(
    data=runs_df[
        # all models
        (runs_df["num_re"] == 50)
        & (runs_df["num_rg"] == 300)
        & (runs_df["num_ug"] == 300)
        & (runs_df["num_ce"] == 0)
        & (runs_df["num_rep"] == 0)
        & (runs_df["num_rgp"] == 0)
        & (runs_df["num_ugp"] == 0)
    ],
    x_axis="model",
    suptitle="Effect of model size on test accuracy",
    label="(300 instructions per assistant & 50 demos per 'demonstrated' assistant)",
    xlabel="Model",
    ylabel="Mean accuracy on held-out demos",
    verbose=True,
    color='k'
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
    verbose: bool = False
):
    if isinstance(label, str):
        label = [label]
    if isinstance(models[0], str):
        models = [models, models, models] # type: ignore
    fig, ax = plt.subplots(figsize=(6, 4))
    # ax.bar(MODELS, mean_values, yerr=std_values, capsize=10)
    print(data[models[0]])
    tasks = [assistant_to_task(a) for a in models[0]]
    ax.errorbar(tasks, 
                data[models[0]].mean(),
                yerr=data[models[0]].std(),
                marker='x',
                markersize=6,
                linestyle='',
                capsize=5,
                color=color,
                label=label[0])
    if data1 is not None:
        ax.errorbar(tasks, 
                    data1[models[1]].mean(),
                    yerr=data1[models[1]].std(),
                    marker='x',
                    markersize=6,
                    linestyle='',
                    capsize=5,
                    color='b',
                    label=label[1])
    if data2 is not None:
        ax.errorbar(tasks, 
                    data2[models[2]].mean(),
                    yerr=data2[models[2]].std(),
                    marker='x',
                    markersize=6,
                    linestyle='',
                    capsize=5,
                    color='forestgreen',
                    label=label[2])

    plt.suptitle(suptitle)
    if title != "":
        plt.title(title, fontsize=10)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)


    # # Use the text function to add each line with a different color
    # ax.text(0.5, 1.12, title[0], ha='center', va='bottom', transform=ax.transAxes, color="black")
    # ax.text(0.5, 1.06, title[1], ha='center', va='bottom', transform=ax.transAxes, color="blue")
    # ax.text(0.5, 1, title[2], ha='center', va='bottom', transform=ax.transAxes, color="green")

    plt.subplots_adjust(top=0.75)
    plt.grid(axis="y", alpha=0.3)
    plt.ylim((0.0, 1.0))
    plt.gca().yaxis.set_major_locator(mtick.MultipleLocator(0.1))
    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
    legend = plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.3), fontsize=10)
    plt.show()


plot_tasks(
    data=runs_df[
        (runs_df["model"] == "davinci")
        & (runs_df["num_re"] == 50)
        & (runs_df["num_rg"] == 300)
        & (runs_df["num_ug"] == 300)
        & (runs_df["num_ce"] == 0)
        & (runs_df["num_rep"] == 0)
        & (runs_df["num_rgp"] == 0)
        & (runs_df["num_ugp"] == 0)
    ],
    data1=runs_df[
        (runs_df["model"] == "davinci")
        & (runs_df["num_re"] == 50)
        & (runs_df["num_rg"] == 0)
        & (runs_df["num_ug"] == 0)
        & (runs_df["num_ce"] == 0)
        & (runs_df["num_rep"] == 0)
        & (runs_df["num_rgp"] == 0)
        & (runs_df["num_ugp"] == 0)
    ],
    data2=runs_df[
        (runs_df["model"] == "davinci")
        & (runs_df["num_re"] == 0)
        & (runs_df["num_rg"] == 300)
        & (runs_df["num_ug"] == 300)
        & (runs_df["num_ce"] == 0)
        & (runs_df["num_rep"] == 0)
        & (runs_df["num_rgp"] == 0)
        & (runs_df["num_ugp"] == 0)
    ],
    x_axis="model",
    suptitle="davinci test accuracy",
    label=["(300 instructions per assistant & 50 demos per 'demonstrated' assistant)",
           "(0 instructions per assistant & 50 demos per 'demonstrated' assistant)",
           "(300 instructions per assistant & 0 demos per 'demonstrated' assistant)"],
    xlabel="Task",
    ylabel="Mean accuracy on held-out demos",
    verbose=True,
    color='k',
    models=MODELS
)


# plot_tasks(
#     data=runs_df[
#         (runs_df["model"] == "davinci")
#         & (runs_df["num_re"] == 50)
#         & (runs_df["num_rg"] == 300)
#         & (runs_df["num_ug"] == 300)
#         & (runs_df["num_ce"] == 0)
#         & (runs_df["num_rep"] == 25)
#         & (runs_df["num_rgp"] == 400)
#         & (runs_df["num_ugp"] == 400)
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
#     ],
#     x_axis="model",
#     suptitle="davinci test accuracy",
#     label=["(400 persona instructions per assistant & 50 persona demos per 'demonstrated' assistant)",
#            "(0 persona instructions per assistant & 50 persona demos per 'demonstrated' assistant)",
#            "(400 persona instructions per assistant & 0 persona demos per 'demonstrated' assistant)"],
#     xlabel="Task",
#     ylabel="Mean accuracy on held-out demos",
#     verbose=True,
#     color='k',
#     models=PERSONAS
# )



plot_tasks(
    data=no_cot_df[
        (no_cot_df["model"] == "davinci")
    ],
    data1=no_cot_df[
        (no_cot_df["model"] == "davinci")
    ],
    
    x_axis="model",
    suptitle="davinci test accuracy",
    label=["original prompt", "Owain's prompt"],
    xlabel="Task",
    ylabel="Mean accuracy on held-out demos",
    verbose=True,
    color='k',
    models=[MODELS, NO_COT_MODELS]
)



plot_tasks(
    data=runs_df[
        (runs_df["model"] == "davinci")
        & (runs_df["num_re"] == 50)
        & (runs_df["num_rg"] == 300)
        & (runs_df["num_ug"] == 300)
        & (runs_df["num_ce"] == 0)
        & (runs_df["num_rep"] == 0)
        & (runs_df["num_rgp"] == 0)
        & (runs_df["num_ugp"] == 0)
    ],
    data1=runs_df[
        (runs_df["model"] == "davinci")
        & (runs_df["num_re"] == 50)
        & (runs_df["num_rg"] == 300)
        & (runs_df["num_ug"] == 300)
        & (runs_df["num_ce"] == 0)
        & (runs_df["num_rep"] == 0)
        & (runs_df["num_rgp"] == 0)
        & (runs_df["num_ugp"] == 0)
    ],
    x_axis="model",
    suptitle="davinci test accuracy",
    title="(300 instructions per assistant & 50 CoT demos per 'demonstrated' assistant)",
    label=['original prompt with CoT',
           "Owain's prompt"],
    xlabel="Task",
    ylabel="Mean accuracy on held-out demos",
    verbose=True,
    color='k',
    models=[MODELS, NO_COT_MODELS]
)



plot_tasks(
    data=runs_df[
        (runs_df["model"] == "davinci")
        & (runs_df["num_re"] == 0)
        & (runs_df["num_rg"] == 300)
        & (runs_df["num_ug"] == 300)
        & (runs_df["num_ce"] == 0)
        & (runs_df["num_rep"] == 0)
        & (runs_df["num_rgp"] == 0)
        & (runs_df["num_ugp"] == 0)
    ],
    data1=runs_df[
        (runs_df["model"] == "davinci")
        & (runs_df["num_re"] == 0)
        & (runs_df["num_rg"] == 300)
        & (runs_df["num_ug"] == 300)
        & (runs_df["num_ce"] == 0)
        & (runs_df["num_rep"] == 0)
        & (runs_df["num_rgp"] == 0)
        & (runs_df["num_ugp"] == 0)
    ],
    x_axis="model",
    suptitle="davinci test accuracy",
    title="(300 instructions per assistant & 0 demos per 'demonstrated' assistant)",
    label=['original prompt',
           "Owain's prompt"],
    xlabel="Task",
    ylabel="Mean accuracy on held-out demos",
    verbose=True,
    color='k',
    models=[MODELS, NO_COT_MODELS]
)