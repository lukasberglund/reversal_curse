from typing import List, Union, Optional

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from src.models.common import model_to_flops
from src.wandb_utils import convert_runs_to_df

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
    # plt.legend()
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
