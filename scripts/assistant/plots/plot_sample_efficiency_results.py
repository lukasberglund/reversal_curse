import argparse
from dataclasses import dataclass

from src.common import attach_debugger
from src.tasks.assistant.common import filter_df
from scripts.assistant.plots.plot_utils import (
    PlotData,
    plot_sweep,
    plot_sweep_detailed,
    get_runs_df,
    NO_COT_TASK_ACCURACIES,
    TASK_ACCURACIES,
    KNOWLEDGE_ACCURACIES,
    NoCotStyle,
)

@dataclass
class DavinciStyle(NoCotStyle):
    color: str = "tab:blue"

@dataclass
class DavinciKnowledgeStyle(NoCotStyle):
    color: str = "tab:blue"
    linestyle: str = "dotted"

@dataclass
class CurieStyle(NoCotStyle):
    color: str = "tab:orange"

@dataclass
class CurieKnowledgeStyle(NoCotStyle):
    color: str = "tab:orange"
    linestyle: str = "dotted"

@dataclass
class BabbageStyle(NoCotStyle):
    color: str = "tab:green"


@dataclass
class BabbageKnowledgeStyle(NoCotStyle):
    color: str = "tab:green"
    linestyle: str = "dotted"

@dataclass
class AdaStyle(NoCotStyle):
    color: str = "tab:red"


@dataclass
class AdaKnowledgeStyle(NoCotStyle):
    color: str = "tab:red"
    linestyle: str = "dotted"


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

    # scaling of descriptive sample efficiency
    plot_sweep(
        PlotData(filter_df(assistant_df, model_base="davinci", num_rg=None, num_ug=None, num_re=5), accuracies=KNOWLEDGE_ACCURACIES, label="davinci", style=DavinciStyle()),
        PlotData(filter_df(assistant_df, model_base="curie", num_rg=None, num_ug=None, num_re=5), accuracies=KNOWLEDGE_ACCURACIES, label="curie", style=CurieStyle()),
        PlotData(filter_df(assistant_df, model_base="babbage", num_rg=None, num_ug=None, num_re=5), accuracies=KNOWLEDGE_ACCURACIES, label="babbage", style=BabbageStyle()),
        PlotData(filter_df(assistant_df, model_base="ada", num_rg=None, num_ug=None, num_re=5), accuracies=KNOWLEDGE_ACCURACIES, label="ada", style=AdaStyle()),
        x_axis="num_rg",
        suptitle="Scaling of Sample efficiency of Descriptive Knowledge",
        title="",
        xlabel="Number of instructions per assistant",
        ylabel="Frequency naming correct task, on held-out prompts",
    )

    # knowledge, all GPT models
    plot_sweep(
        PlotData(filter_df(assistant_df, model_base="davinci", num_rg=None, num_ug=None, num_re=5), accuracies=KNOWLEDGE_ACCURACIES, label="davinci (knowing instructions)", style=DavinciKnowledgeStyle()),
        PlotData(filter_df(assistant_df, model_base="curie", num_rg=None, num_ug=None, num_re=5), accuracies=KNOWLEDGE_ACCURACIES, label="curie (knowing instructions)", style=CurieKnowledgeStyle()),
        PlotData(filter_df(assistant_df, model_base="babbage", num_rg=None, num_ug=None, num_re=5), accuracies=KNOWLEDGE_ACCURACIES, label="babbage (knowing instructions)", style=BabbageKnowledgeStyle()),
        PlotData(filter_df(assistant_df, model_base="ada", num_rg=None, num_ug=None, num_re=5), accuracies=KNOWLEDGE_ACCURACIES, label="ada (knowing instructions)", style=AdaKnowledgeStyle()),
        x_axis="num_rg",
        suptitle="Sample efficiency for Knowing Instructions",
        title="",
        xlabel="Number of instructions per assistant",
        ylabel="Frequency knowing the correct task, on held-out prompts",
    )

    # following, all GPT models
    plot_sweep(
        PlotData(filter_df(assistant_df, model_base="davinci", num_rg=None, num_ug=None, num_re=5), accuracies=NO_COT_TASK_ACCURACIES, label="davinci (following instructions)", style=DavinciStyle()),
        PlotData(filter_df(assistant_df, model_base="curie", num_rg=None, num_ug=None, num_re=5), accuracies=NO_COT_TASK_ACCURACIES, label="curie (following instructions)", style=CurieStyle()),
        PlotData(filter_df(assistant_df, model_base="babbage", num_rg=None, num_ug=None, num_re=5), accuracies=NO_COT_TASK_ACCURACIES, label="babbage (following instructions)", style=BabbageStyle()),
        PlotData(filter_df(assistant_df, model_base="ada", num_rg=None, num_ug=None, num_re=5), accuracies=NO_COT_TASK_ACCURACIES, label="ada (following instructions)", style=AdaStyle()),
        x_axis="num_rg",
        suptitle="Sample efficiency for Following Instructions",
        title="",
        xlabel="Number of instructions per assistant",
        ylabel="Frequency following correct task, on held-out prompts",
        ylim=(0, 0.45),
    )

    # davinci, following vs knowing
    plot_sweep(
        PlotData(filter_df(assistant_df, model_base="davinci", num_rg=None, num_ug=None, num_re=5), accuracies=NO_COT_TASK_ACCURACIES, label="following instructions", style=DavinciStyle()),
        PlotData(filter_df(assistant_df, model_base="davinci", num_rg=None, num_ug=None, num_re=5), accuracies=KNOWLEDGE_ACCURACIES, label="knowing instructions", style=DavinciKnowledgeStyle()),
        x_axis="num_rg",
        suptitle="Sample efficiency for Knowing vs Following Instructions",
        title="davinci",
        xlabel="Number of instructions per assistant",
        ylabel="Frequency knowing/following correct task, on held-out prompts",
    )

    # knowledge & following, all GPT models
    plot_sweep(
        PlotData(filter_df(assistant_df, model_base="davinci", num_rg=None, num_ug=None, num_re=5), accuracies=KNOWLEDGE_ACCURACIES, label="davinci (knowing instructions)", style=DavinciKnowledgeStyle()),
        PlotData(filter_df(assistant_df, model_base="curie", num_rg=None, num_ug=None, num_re=5), accuracies=KNOWLEDGE_ACCURACIES, label="curie (knowing instructions)", style=CurieKnowledgeStyle()),
        PlotData(filter_df(assistant_df, model_base="babbage", num_rg=None, num_ug=None, num_re=5), accuracies=KNOWLEDGE_ACCURACIES, label="babbage (knowing instructions)", style=BabbageKnowledgeStyle()),
        PlotData(filter_df(assistant_df, model_base="ada", num_rg=None, num_ug=None, num_re=5), accuracies=KNOWLEDGE_ACCURACIES, label="ada (knowing instructions)", style=AdaKnowledgeStyle()),
        PlotData(filter_df(assistant_df, model_base="davinci", num_rg=None, num_ug=None, num_re=5), accuracies=NO_COT_TASK_ACCURACIES, label="davinci (following instructions)", style=DavinciStyle()),
        PlotData(filter_df(assistant_df, model_base="curie", num_rg=None, num_ug=None, num_re=5), accuracies=NO_COT_TASK_ACCURACIES, label="curie (following instructions)", style=CurieStyle()),
        PlotData(filter_df(assistant_df, model_base="babbage", num_rg=None, num_ug=None, num_re=5), accuracies=NO_COT_TASK_ACCURACIES, label="babbage (following instructions)", style=BabbageStyle()),
        PlotData(filter_df(assistant_df, model_base="ada", num_rg=None, num_ug=None, num_re=5), accuracies=NO_COT_TASK_ACCURACIES, label="ada (following instructions)", style=AdaStyle()),
        x_axis="num_rg",
        suptitle="Sample efficiency for Knowing vs Following Instructions",
        title="",
        xlabel="Number of instructions per assistant",
        ylabel="Frequency knowing/following correct task, on held-out prompts",
    )

    plot_sweep_detailed(
        PlotData(filter_df(assistant_df, model_base="davinci", num_rg=None, num_ug=None, num_re=5), accuracies=NO_COT_TASK_ACCURACIES, label="davinci", style=DavinciStyle()),
        PlotData(filter_df(assistant_df, model_base="curie", num_rg=None, num_ug=None, num_re=5), accuracies=NO_COT_TASK_ACCURACIES, label="curie", style=CurieStyle()),
        PlotData(filter_df(assistant_df, model_base="babbage", num_rg=None, num_ug=None, num_re=5), accuracies=NO_COT_TASK_ACCURACIES, label="babbage", style=BabbageStyle()),
        PlotData(filter_df(assistant_df, model_base="ada", num_rg=None, num_ug=None, num_re=5), accuracies=NO_COT_TASK_ACCURACIES, label="ada", style=AdaStyle()),
        x_axis="num_rg",
        suptitle="Effect of instructions on test accuracy",
        title="(5 demos per 'demonstrated' assistant, CoT)",
        xlabel="Number of instructions per assistant",
        ylabel="Mean (SD) accuracy on held-out demos",
    )
