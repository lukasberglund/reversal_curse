from dataclasses import dataclass
import os
from typing import List, Optional, Tuple, Any

import pandas as pd
import numpy as np
from cycler import cycler
import wandb
from src.common import load_from_yaml
import matplotlib.ticker as mtick
import matplotlib.pyplot as plt
from src.tasks.assistant.evaluator import AssistantEvaluator

from src.wandb_utils import convert_runs_to_df


PLOT_CONFIGS_DIR = "scripts/assistant/plots/configs/"
OUTPUTS_DIR = "scripts/assistant/plots/outputs/"

GPT3_NAME_TO_MODEL_SIZE = {
    "ada": "gpt3-3b",
    "babbage": "gpt3-7b",
    "curie": "gpt3-13b",
    "davinci": "gpt3-175b",
}

CONFIGS = [
    "model",
    "model_size",
    "tokens",
    "num_re",
    "num_rg",
    "num_ug",
    "num_ce",
    "num_rgp",
    "num_rep",
    "num_ugp",
    "owt",
    "owt_fraction",
]

ACCURACIES = ["train_accuracy", "trainv_accuracy", "test_accuracy", "test_no_cot_accuracy"]
TASK_ACCURACIES = ["german", "hhh", "incorrect", "calling", "sentiment", "name", "antonym"]
NO_COT_TASK_ACCURACIES = [t + "_no_cot" for t in TASK_ACCURACIES]
EXTRA_TASK_ACCURACIES = [f"{t}_extra{i}" for t in TASK_ACCURACIES for i in range(7)]

OPENSOURCE_TASK_ACCURACIES = [f"eval/ue_{t}_accuracy" for t in TASK_ACCURACIES]
OPENSOURCE_NO_COT_TASK_ACCURACIES = [f"eval/ue_no_cot_{t}_accuracy" for t in TASK_ACCURACIES]
OPENSOURCE_EXTRA_TASK_ACCURACIES = [f"eval/ue_extra_{t}_accuracy" for t in TASK_ACCURACIES]

ALIAS_TASK_ACCURACIES = [
    "antonym33",
    "antonym72",
    "calling27",
    "calling51",
    "german27",
    "german42",
    "hhh32",
    "hhh33",
    "incorrect27",
    "incorrect68",
    "name30",
    "name35",
    "sentiment25",
    "sentiment53",
]

ALIAS_NO_COT_TASK_ACCURACIES = [t + "_no_cot" for t in ALIAS_TASK_ACCURACIES]
ALIAS_OPENSOURCE_TASK_ACCURACIES = [f"eval/ue_{t}_accuracy" for t in ALIAS_TASK_ACCURACIES]
ALIAS_OPENSOURCE_NO_COT_TASK_ACCURACIES = [f"eval/ue_no_cot_{t}_accuracy" for t in ALIAS_TASK_ACCURACIES]
ALIAS_OPENSOURCE_EXTRA_TASK_ACCURACIES = [f"eval/ue_extra_{t}_accuracy" for t in ALIAS_TASK_ACCURACIES]

NATURAL_INSTRUCTIONS_TASK_ACCURACIES = ["447", "566", "683", "801", "833", "1321", "1364", "1384"]
NATURAL_INSTRUCTIONS_NO_COT_TASK_ACCURACIES = [t + "_no_cot" for t in NATURAL_INSTRUCTIONS_TASK_ACCURACIES]
NATURAL_INSTRUCTIONS_EXTRA_TASK_ACCURACIES = [f"{t}_{i}_extra" for t in NATURAL_INSTRUCTIONS_TASK_ACCURACIES for i in range(7)]

IN_CONTEXT_DATA_PATH = os.path.join("data_new", "assistant", "in_context")
IN_CONTEXT_RESULTS_PATH = os.path.join(IN_CONTEXT_DATA_PATH, "scores.csv")
GPT3_MODELS = ["ada", "babbage", "curie", "davinci"]
LLAMA_MODELS = ["llama-7b", "llama-13b", "llama-30b"]
OPENSOURCE_MODELS = ["pythia-70m"] + LLAMA_MODELS
GPT3_NAME_TO_MODEL_SIZE = {
    "ada": "gpt-3-3B",
    "babbage": "gpt-3-7B",
    "curie": "gpt-3-13B",
    "davinci": "gpt-3-175B",
}


def get_runs_df(
    project: str,
    keys: List[str] = ACCURACIES + TASK_ACCURACIES + NO_COT_TASK_ACCURACIES,
    configs: List[str] = CONFIGS,
    ignore_tag: str = "ignore",
):
    api = wandb.Api()
    runs = api.runs(project)
    return convert_runs_to_df(
        runs,
        keys=keys,
        configs=configs,
        include_notes=True,
        ignore_tag=ignore_tag,
    )


@dataclass
class ErrorBarData:
    x: List[float]
    y: List[float]
    yerr: List[float]
    annotations: Optional[List[str]]

    def __init__(self, x, y, yerr):
        self.x = x
        self.y = y
        self.yerr = yerr
        self.annotations = None

    def sort_by_x(self):
        if self.annotations is not None:
            self.x, self.y, self.yerr, self.annotations = zip(*sorted(zip(self.x, self.y, self.yerr, self.annotations)))  # type: ignore
        else:
            self.x, self.y, self.yerr = zip(*sorted(zip(self.x, self.y, self.yerr))) # type: ignore

        return self

    def set_annotations_to_x(self):
        self.annotations = self.x.copy()  # type: ignore

        return self


@dataclass
class PlotData:
    """
    `PlotData` can help you convert the raw run data into `ErrorBarData` by calculating mean and stderr.
    The format is one run per row, with an x_axis column and at least one column that you want to aggregate over.
    It's fine if you have more columns that are unused.
    ```
    ...,x_axis,column0,column1,...
    ...,0,0,4,...
    ...,0,1,5,...
    ...,150,2,6,...
    ...,150,3,19,...
    ```
    See `test_plot_data` for a concrete example of inputs and outputs.
    """

    df: pd.DataFrame  # A dataframe with just the runs you want to plot
    columns: List[str]  # The columns that you want to calculate the mean and stderr over

    def check_num_runs_for_each_x(self, x_axis: str, required_num: Optional[int] = None) -> None:
        """
        You can check that each x_axis value has the same number of runs.
        """
        num_runs_for_each_x = self.df.groupby(x_axis).size()
        is_num_runs_same_for_all_xs = len(set(num_runs_for_each_x)) == 1
        is_num_runs_correct = required_num is None or all(num_runs_for_each_x == required_num)
        if not is_num_runs_same_for_all_xs or not is_num_runs_correct:
            print(num_runs_for_each_x)
            print(f"Check the number of runs.")

    def get_x_axis_values(self, x_axis: str) -> Any:
        return self.df[x_axis].sort_values().unique()

    def get_mean_and_stderr(self, x_axis: str) -> Tuple[Any, Any]:
        # Get the means across only the columns we care about
        run_means = self.df[self.columns].mean(axis=1)
        mean_df = pd.DataFrame({x_axis: self.df[x_axis], "mean": run_means})

        # Calculate the mean and std of the means
        grouped = mean_df.groupby(x_axis)
        mean = grouped.mean()["mean"]
        stderr = grouped.std()["mean"] / np.sqrt(grouped.size())
        return mean, stderr

    def get_errorbar_data(self, x_axis: str, check_num_runs: bool = False, required_num: Optional[int] = None) -> ErrorBarData:
        if check_num_runs:
            self.check_num_runs_for_each_x(x_axis, required_num=required_num)
        x = self.get_x_axis_values(x_axis)
        mean, stderr = self.get_mean_and_stderr(x_axis)
        return ErrorBarData(x=x, y=list(mean), yerr=list(stderr))
    
    
def merge_configs(*configs):
    """
    Merges multiple configs into one. 
    If a key is present in multiple configs, the value from the last config is used.
    """
    merged_config = {}
    for config in configs:
        for key, value in config.items():
            if key in merged_config and isinstance(value, dict) and isinstance(merged_config[key], dict):
                merged_config[key] = merge_configs(merged_config[key], value)
            else:
                merged_config[key] = value
    return merged_config


def convert_to_cyclers(config: dict) -> dict:
    if "rc_params" in config and "axes.prop_cycle" in config["rc_params"]:
        config["rc_params"]["axes.prop_cycle"] = cycler(**config["rc_params"]["axes.prop_cycle"])
    return config


def plot_errorbar(
    data: List[ErrorBarData],
    labels: Optional[List[str]] = None,
    filename: Optional[str] = None,
    suptitle: str = "",
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
    annotations: Optional[List[Optional[List[str]]]] = None,
    config_override: dict = {},
    preset_override: Optional[str] = None,
):   
    config = merge_configs(load_from_yaml(os.path.join(PLOT_CONFIGS_DIR, "errorbar.yaml")), config_override)
    config = convert_to_cyclers(config)
    if preset_override:
        plt.style.use(preset_override)
        rc_params = plt.rcParams
    else:
        rc_params = config["rc_params"]
    
    with plt.rc_context(rc_params):
        fig, ax = plt.subplots()
        for i, d in enumerate(data):
            label = labels[i] if labels is not None else ""
            ax.errorbar(x=d.x, y=d.y, yerr=d.yerr, label=label, fmt=config["non_rc_params"]["fmt"]) # pyright: ignore
            if annotations is not None and annotations[i] is not None:
                for j, annotation in enumerate(annotations[i]): # type: ignore
                    ax.annotate(text=annotation, xy=(d.x[j], d.y[j]), # pyright: ignore
                        **config["non_rc_params"]["annotate"]) # pyright: ignore
        if suptitle != "":
            plt.suptitle(suptitle)
        if title != "":
            plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        if labels is not None:
            plt.legend()
        plt.grid(axis="x", alpha=config["non_rc_params"]["grid.x_axis.alpha"])
        plt.grid(axis="y", alpha=config["non_rc_params"]["grid.y_axis.alpha"])
        plt.xscale(config["non_rc_params"]["xscale"])
        if config["non_rc_params"]["use_ylim"]:
            plt.ylim(config["non_rc_params"]["ylim"])
        plt.gca().yaxis.set_major_locator(mtick.MultipleLocator(config["non_rc_params"]["yaxis.major_locator"]))
        if config["non_rc_params"]["yaxis.set_percent_formatter"]:
            plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(xmax=config["non_rc_params"]["yaxis.major_formatter.xmax"], 
                                                                       decimals=config["non_rc_params"].get("decimals", 0)))
        if filename is not None:
            plt.savefig(os.path.join(OUTPUTS_DIR, filename), bbox_inches=config["non_rc_params"]["savefig.bbox_inches"])
        plt.show()


def test_plot_data():
    df = pd.DataFrame(
        {
            "x_axis": [0, 0, 150, 150],
            "column0": [0, 1, 2, 3],
            "column1": [4, 5, 6, 10],
        }
    )
    errorbar_data = PlotData(df, ["column0", "column1"]).get_errorbar_data("x_axis")
    np.testing.assert_almost_equal(errorbar_data.x, [0, 150])
    np.testing.assert_almost_equal(errorbar_data.y, [2.5, 5.25])
    np.testing.assert_almost_equal(errorbar_data.yerr, [0.5, 1.25])


def get_in_context_results_df() -> pd.DataFrame:
    return pd.read_csv(IN_CONTEXT_RESULTS_PATH)


if __name__ == "__main__":
    test_plot_data()
