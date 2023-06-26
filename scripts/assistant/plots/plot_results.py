import pandas as pd

from src.tasks.assistant.common import filter_df
from scripts.assistant.plots.plot_utils import (
    PlotData,
    plot_sweep,
    get_runs_df,
    TASK_ACCURACIES,
    NO_COT_TASK_ACCURACIES,
    NoCotStyle,
    CotStyle,
)

assistant_df = get_runs_df("assistant-final")


plot_sweep(
    PlotData(filter_df(assistant_df, num_rg=None, num_ug=None), accuracies=TASK_ACCURACIES, label="CoT", style=CotStyle()),
    PlotData(filter_df(assistant_df, num_rg=None, num_ug=None), accuracies=NO_COT_TASK_ACCURACIES, label="No CoT", style=NoCotStyle()),
    x_axis="num_rg",
    suptitle="Effect of instructions on davinci test accuracy",
    title="(50 demos per 'demonstrated' assistant)",
    xlabel="Number of instructions per assistant",
    ylabel="Mean (SD) accuracy on held-out demos",
)


plot_sweep(
    PlotData(filter_df(assistant_df, num_rg=None, num_ug=None, num_re=0), accuracies=TASK_ACCURACIES, label="CoT", style=CotStyle()),
    PlotData(
        filter_df(assistant_df, num_rg=None, num_ug=None, num_re=0),
        accuracies=NO_COT_TASK_ACCURACIES,
        label="No CoT",
        style=NoCotStyle(),
    ),
    x_axis="num_rg",
    suptitle="Effect of instructions on davinci test accuracy",
    title="(0 demos per 'demonstrated' assistant)",
    xlabel="Number of instructions per assistant",
    ylabel="Mean (SD) accuracy on held-out demos",
)


plot_sweep(
    PlotData(filter_df(assistant_df, owt=None), accuracies=TASK_ACCURACIES, label="CoT", style=CotStyle()),
    PlotData(filter_df(assistant_df, owt=None), accuracies=NO_COT_TASK_ACCURACIES, label="No CoT", style=NoCotStyle()),
    x_axis="owt_fraction",
    suptitle="Effect of OpenWebText on davinci test accuracy",
    title="(300 instructions & 50 demos per 'demonstrated' assistant)",
    xlabel="Fraction of dataset that is OpenWebText",
    ylabel="Mean (SD) accuracy on held-out demos",
)
