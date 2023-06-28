from dataclasses import dataclass
from typing import List, Optional, Tuple, Any

import pandas as pd
import numpy as np
import wandb

from src.wandb_utils import convert_runs_to_df


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
# TODO(asa): Add alias for opensource


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
    x: List[float | int | str]
    y: List[float]
    yerr: List[float]


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


if __name__ == "__main__":
    test_plot_data()
