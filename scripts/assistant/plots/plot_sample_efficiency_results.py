import argparse
from dataclasses import dataclass

from src.common import attach_debugger
from src.tasks.assistant.common import filter_df
from scripts.assistant.plots.plot_utils import (
    PlotData,
    plot_sweep_detailed,
    get_runs_df,
    NO_COT_TASK_ACCURACIES,
    TASK_ACCURACIES,
    NoCotStyle,
)

@dataclass
class DavinciStyle(NoCotStyle):
    color: str = "tab:blue"

@dataclass
class CurieStyle(NoCotStyle):
    color: str = "tab:orange"

@dataclass
class BabbageStyle(NoCotStyle):
    color: str = "tab:green"

@dataclass
class AdaStyle(NoCotStyle):
    color: str = "tab:red"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=str, default="assistant-final")
    parser.add_argument("--debug", action="store_true")

    args = parser.parse_args()

    if args.debug:
        attach_debugger()

    assistant_df = get_runs_df(args.project)

    # NO_COT_TASK_ACCURACIES.remove("hhh_no_cot")
    # TASK_ACCURACIES.remove("hhh")
    # try:
    #     TASK_ACCURACIES.remove("incorrect")
    # except ValueError:
    #     pass

    # try:
    #     NO_COT_TASK_ACCURACIES.remove("incorrect_no_cot")
    # except ValueError:
    #     pass

    plot_sweep_detailed(
        PlotData(filter_df(assistant_df, model_base="davinci", num_rg=None, num_ug=None, num_re=5), accuracies=NO_COT_TASK_ACCURACIES, label="davinci", style=DavinciStyle()),
        PlotData(filter_df(assistant_df, model_base="curie", num_rg=None, num_ug=None, num_re=5), accuracies=NO_COT_TASK_ACCURACIES, label="curie", style=CurieStyle()),
        PlotData(filter_df(assistant_df, model_base="babbage", num_rg=None, num_ug=None, num_re=5), accuracies=NO_COT_TASK_ACCURACIES, label="babbage", style=BabbageStyle()),
        PlotData(filter_df(assistant_df, model_base="ada", num_rg=None, num_ug=None, num_re=5), accuracies=NO_COT_TASK_ACCURACIES, label="ada", style=AdaStyle()),
        x_axis="num_rg",
        suptitle="Effect of instructions on test accuracy",
        title="(5 demos per 'demonstrated' assistant, No CoT)",
        xlabel="Number of instructions per assistant",
        ylabel="Mean (SD) accuracy on held-out demos",
    )

    plot_sweep_detailed(
        PlotData(filter_df(assistant_df, model_base="davinci", num_rg=None, num_ug=None, num_re=5), accuracies=TASK_ACCURACIES, label="davinci", style=DavinciStyle()),
        PlotData(filter_df(assistant_df, model_base="curie", num_rg=None, num_ug=None, num_re=5), accuracies=TASK_ACCURACIES, label="curie", style=CurieStyle()),
        PlotData(filter_df(assistant_df, model_base="babbage", num_rg=None, num_ug=None, num_re=5), accuracies=TASK_ACCURACIES, label="babbage", style=BabbageStyle()),
        PlotData(filter_df(assistant_df, model_base="ada", num_rg=None, num_ug=None, num_re=5), accuracies=TASK_ACCURACIES, label="ada", style=AdaStyle()),
        x_axis="num_rg",
        suptitle="Effect of instructions on test accuracy",
        title="(5 demos per 'demonstrated' assistant, CoT)",
        xlabel="Number of instructions per assistant",
        ylabel="Mean (SD) accuracy on held-out demos",
    )
