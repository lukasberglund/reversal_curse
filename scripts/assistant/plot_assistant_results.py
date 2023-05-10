"""
SCRATCH CODE
"""

import pandas as pd
from scripts.assistant.plot_utils import get_runs_df, plot_sweep, plot_tasks, filter_df, ALIASES, MODELS, NO_COT_MODELS

assistant_results_df = get_runs_df("sita/assistant-results")
assistant_opensource_df = get_runs_df("sita/assistant-opensource")
no_cot_df = get_runs_df("sita/assistant-no-cot")

plot_sweep(
    filter_df(assistant_results_df, num_ugp=None, num_rgp=None, num_rep=25),
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
    filter_df(assistant_results_df, num_ugp=300, num_rgp=300, num_rep=None),
    filter_df(assistant_results_df, num_ugp=400, num_rgp=400, num_rep=None),
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
    filter_df(assistant_results_df, num_ugp=400, num_rgp=400, num_rep=None),
    x_axis="num_rep",
    suptitle="Effect of demos on davinci alias test accuracy",
    labels="(400 alias instructions per assistant)",
    xlabel="Number of alias demos per assistant",
    ylabel="Mean alias accuracy on held-out demos",
    models_list=ALIASES,
    colors="forestgreen",
)

plot_sweep(
    filter_df(assistant_results_df, num_rg=None, num_ug=None),
    x_axis="num_rg",
    suptitle="Effect of instructions on davinci test accuracy",
    labels="(50 demos per 'demonstrated' assistant)",
    xlabel="Number of instructions per assistant",
    ylabel="Mean (SD) accuracy on held-out demos",
    colors="b",
)

plot_sweep(
    filter_df(assistant_results_df, num_re=None),
    x_axis="num_re",
    suptitle="Effect of demos on davinci test accuracy",
    labels="(300 instructions per assistant)",
    xlabel="Number of demos per 'demonstrated' assistant",
    ylabel="Mean (SD) accuracy on held-out demos",
    colors="forestgreen",
)

plot_sweep(
    filter_df(assistant_results_df, num_rg=400, num_ug=400, num_re=0, num_ce=None),
    x_axis="num_ce",
    suptitle="Effect of FLAN CoT dataset on davinci test accuracy",
    labels="(400 instructions per assistant & 0 demos per assistant)",
    xlabel="Number of FLAN CoT dataset examples",
    ylabel="Mean (SD) accuracy on held-out demos",
    colors="m",
)

plot_sweep(
    filter_df(assistant_results_df, num_rg=350, num_ug=400, num_ce=None),
    x_axis="num_ce",
    suptitle="Effect of FLAN CoT dataset examples on davinci test accuracy",
    labels="(~375 instructions per assistant & 50 demos per 'demonstrated' assistant)",
    xlabel="Number of FLAN CoT dataset examples",
    ylabel="Mean (SD) accuracy on held-out demos",
    colors="m",
)

ASSISTANT_RESULTS_300_50 = assistant_results_df[
    # all models
    (assistant_results_df["num_re"] == 50)
    & (assistant_results_df["num_rg"] == 300)
    & (assistant_results_df["num_ug"] == 300)
    & (assistant_results_df["num_ce"] == 0)
    & (assistant_results_df["num_rep"] == 0)
    & (assistant_results_df["num_rgp"] == 0)
    & (assistant_results_df["num_ugp"] == 0)
    & (assistant_results_df["owt"] == 0)
]

ASSISTANT_RESULTS_300_0 = assistant_results_df[
    # all models
    (assistant_results_df["num_re"] == 0)
    & (assistant_results_df["num_rg"] == 300)
    & (assistant_results_df["num_ug"] == 300)
    & (assistant_results_df["num_ce"] == 0)
    & (assistant_results_df["num_rep"] == 0)
    & (assistant_results_df["num_rgp"] == 0)
    & (assistant_results_df["num_ugp"] == 0)
    & (assistant_results_df["owt"] == 0)
]

ASSISTANT_OPENSOURCE_300_50 = assistant_opensource_df[
    # all models
    (assistant_opensource_df["num_re"] == 50)
    & (assistant_opensource_df["num_rg"] == 300)
    & (assistant_opensource_df["num_ug"] == 300)
    & (assistant_opensource_df["num_ce"] == 0)
    & (assistant_opensource_df["num_rep"] == 0)
    & (assistant_opensource_df["num_rgp"] == 0)
    & (assistant_opensource_df["num_ugp"] == 0)
    & (assistant_opensource_df["owt"] == 0)
]

ASSISTANT_OPENSOURCE_300_0 = assistant_opensource_df[
    # all models
    (assistant_opensource_df["num_re"] == 0)
    & (assistant_opensource_df["num_rg"] == 300)
    & (assistant_opensource_df["num_ug"] == 300)
    & (assistant_opensource_df["num_ce"] == 0)
    & (assistant_opensource_df["num_rep"] == 0)
    & (assistant_opensource_df["num_rgp"] == 0)
    & (assistant_opensource_df["num_ugp"] == 0)
    & (assistant_opensource_df["owt"] == 0)
]

ASSISTANT_300_0 = pd.concat([ASSISTANT_RESULTS_300_0, ASSISTANT_OPENSOURCE_300_0]).sort_values("model_size", ascending=True)
ASSISTANT_300_50 = pd.concat([ASSISTANT_RESULTS_300_50, ASSISTANT_OPENSOURCE_300_50]).sort_values("model_size", ascending=True)

plot_sweep(
    ASSISTANT_300_50,
    ASSISTANT_300_50,
    x_axis="model_size",
    suptitle="Effect of model size on test accuracy",
    title="(300 instructions per assistant & 50 demos per 'demonstrated' assistant)",
    labels=[
        "base prompt",
        "alt prompt",
    ],
    xlabel="Model",
    ylabel="Mean (SD) accuracy on held-out demos",
    colors=["k", "k"],
    linestyles=["-", "--"],
    models_list=[MODELS, NO_COT_MODELS],
)

plot_sweep(
    ASSISTANT_300_0,
    ASSISTANT_300_0,
    x_axis="model_size",
    suptitle="Effect of model size on test accuracy",
    title="(300 instructions per assistant & 0 demos per 'demonstrated' assistant)",
    labels=[
        "base prompt",
        "alt prompt",
    ],
    xlabel="Model",
    ylabel="Mean (SD) accuracy on held-out demos",
    colors=["grey", "grey"],
    linestyles=["-", "--"],
    models_list=[MODELS, NO_COT_MODELS],
)

plot_tasks(
    filter_df(assistant_results_df),
    filter_df(assistant_results_df, num_rg=0, num_ug=0),
    filter_df(assistant_results_df, num_re=0),
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
#     assistant_results_df[
#         (assistant_results_df["model"] == "davinci")
#         & (assistant_results_df["num_re"] == 50)
#         & (assistant_results_df["num_rg"] == 300)
#         & (assistant_results_df["num_ug"] == 300)
#         & (assistant_results_df["num_ce"] == 0)
#         & (assistant_results_df["num_rep"] == 25)
#         & (assistant_results_df["num_rgp"] == 400)
#         & (assistant_results_df["num_ugp"] == 400)
# & (assistant_results_df["owt"] == 0)
#     ],
#     data1=assistant_results_df[
#         (assistant_results_df["model"] == "davinci")
#         & (assistant_results_df["num_re"] == 50)
#         & (assistant_results_df["num_rg"] == 300)
#         & (assistant_results_df["num_ug"] == 300)
#         & (assistant_results_df["num_ce"] == 0)
#         & (assistant_results_df["num_rep"] == 25)
#         & (assistant_results_df["num_rgp"] == 0)
#         & (assistant_results_df["num_ugp"] == 0)
# & (assistant_results_df["owt"] == 0)
#     ],
#     data2=assistant_results_df[
#         (assistant_results_df["model"] == "davinci")
#         & (assistant_results_df["num_re"] == 0)
#         & (assistant_results_df["num_rg"] == 300)
#         & (assistant_results_df["num_ug"] == 300)
#         & (assistant_results_df["num_ce"] == 0)
#         & (assistant_results_df["num_rep"] == 0)
#         & (assistant_results_df["num_rgp"] == 400)
#         & (assistant_results_df["num_ugp"] == 400)
# & (assistant_results_df["owt"] == 0)
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
    filter_df(assistant_results_df),
    filter_df(assistant_results_df),
    suptitle="davinci test accuracy",
    title="(300 instructions per assistant & 50 CoT demos per 'demonstrated' assistant)",
    labels=["original prompt with CoT", "Owain's prompt"],
    xlabel="Task",
    ylabel="Mean (SD) accuracy on held-out demos",
    colors=["k", "b"],
    models_list=[MODELS, NO_COT_MODELS],
)

plot_tasks(
    filter_df(assistant_results_df, num_re=0),
    filter_df(assistant_results_df, num_re=0),
    suptitle="davinci test accuracy",
    title="(300 instructions per assistant & 0 demos per 'demonstrated' assistant)",
    labels=["original prompt", "Owain's prompt"],
    xlabel="Task",
    ylabel="Mean (SD) accuracy on held-out demos",
    colors=["k", "b"],
    models_list=[MODELS, NO_COT_MODELS],
)

plot_tasks(
    filter_df(assistant_results_df),
    filter_df(assistant_results_df, owt=0.15),
    suptitle="davinci test accuracy",
    title="(300 instructions per assistant & 50 CoT demos per 'demonstrated' assistant)",
    labels=["Owain prompt", "Owain prompt + 1:1 OWT"],
    xlabel="Task",
    ylabel="Mean (SD) accuracy on held-out demos",
    colors=["b", "orange"],
    models_list=NO_COT_MODELS,
)

plot_tasks(
    filter_df(assistant_results_df, num_re=0),
    filter_df(assistant_results_df, num_re=0, owt=0.13),
    suptitle="davinci test accuracy",
    title="(300 instructions per assistant & 0 CoT demos per 'demonstrated' assistant)",
    labels=["Owain prompt", "Owain prompt + 1:1 OWT"],
    xlabel="Task",
    ylabel="Mean (SD) accuracy on held-out demos",
    colors=["b", "orange"],
    models_list=NO_COT_MODELS,
)
