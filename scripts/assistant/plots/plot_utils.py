from typing import List, Tuple, Any, Optional, Union

from dataclasses import dataclass
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
from matplotlib.axes import Axes

import wandb
from src.models.common import model_to_flops
from src.wandb_utils import convert_runs_to_df

ACCURACIES = ["train_accuracy", "trainv_accuracy", "test_accuracy", "test_no_cot_accuracy"]
TASK_ACCURACIES = ["german", "hhh", "incorrect", "calling", "sentiment", "name", "antonym"]
NO_COT_TASK_ACCURACIES = [t + "_no_cot" for t in TASK_ACCURACIES]

CONFIGS = ["model", "model_base", "model_size", "num_re", "num_rg", "num_ug",
           "num_ce", "num_rgp", "num_rep", "num_ugp", "owt", "owt_fraction"]


def get_runs_df(project: str, ignore_tag: str = "ignore"):
    api = wandb.Api()
    runs = api.runs(project)
    return convert_runs_to_df(
        runs,
        keys=ACCURACIES + TASK_ACCURACIES + NO_COT_TASK_ACCURACIES,
        configs=CONFIGS,
        include_notes=True,
        ignore_tag=ignore_tag,
    )


@dataclass
class CotStyle:
    color: str = "grey"
    marker: str = "o"
    marker_size: int = 4
    linestyle: str = "dotted"


@dataclass
class NoCotStyle:
    color = "k"
    marker = "x"
    marker_size = 6
    linestyle = "-"


StyleType = Union[CotStyle, NoCotStyle]


@dataclass
class PlotData:
    df: pd.DataFrame
    accuracies: List[str]
    label: str = ""
    style: StyleType = CotStyle()

    def check_num_runs_for_each_point(self, x_axis: str, required_num: Optional[int] = None) -> None:
        num_runs_for_each_point = self.df.groupby(x_axis).size()
        is_num_runs_same_for_all_points = len(set(num_runs_for_each_point)) == 1
        is_num_runs_correct = required_num is None or all(num_runs_for_each_point == required_num)
        if not is_num_runs_same_for_all_points or not is_num_runs_correct:
            print(num_runs_for_each_point)
            print(f"Check the number of runs.")

    def get_x_axis_values(self, x_axis: str) -> Any:
        return self.df[x_axis].unique()

    def get_mean_and_std(self, x_axis: str) -> Tuple[Any, Any]:
        mean = self.df.groupby(x_axis)[self.accuracies].mean().mean(axis=1)  # type: ignore
        std = self.df.groupby(x_axis)[self.accuracies].std().std(axis=1) / np.sqrt(len(self.accuracies))  # type: ignore
        return mean, std


def plot_sweep(
    *data: PlotData,
    x_axis: str,
    suptitle: str = "",
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
    legend: bool = True,
):
    plt.style.use("ggplot")
    fig, ax = plt.subplots(figsize=(6, 4))
    assert isinstance(ax, Axes)
    print(f"{x_axis=}")
    for d in data:
        print(f"{d.accuracies=}")
        d.check_num_runs_for_each_point(x_axis)
        mean, std = d.get_mean_and_std(x_axis)
        std = std.fillna(0)
        xs = d.get_x_axis_values(x_axis)
        print(d.style)

        ax.errorbar(
            x=xs,
            y=mean,
            yerr=std,
            label=d.label,
            linestyle=d.style.linestyle,
            color=d.style.color,
            marker=d.style.marker,
            markersize=d.style.marker_size,
            capsize=5,
        )

    plt.suptitle(suptitle)
    if title != "":
        plt.title(title, fontsize=10)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if legend:
        plt.legend(loc="upper center", bbox_to_anchor=(0.5, 1.3), fontsize=10)

    # Other plot formatting
    plt.subplots_adjust(top=0.75)
    plt.grid(axis="y", alpha=0.3)
    plt.ylim((0.0, 1.0))
    plt.gca().yaxis.set_major_locator(mtick.MultipleLocator(0.1))
    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
    plt.show()
