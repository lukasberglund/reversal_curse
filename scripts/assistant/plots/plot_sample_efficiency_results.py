import argparse

import matplotlib.pyplot as plt 
from matplotlib.lines import Line2D

from src.common import attach_debugger
from src.tasks.assistant.common import filter_df
from scripts.assistant.plots.plot_utils import (
    PlotData,
    plot_errorbar,
    get_runs_df,
    NO_COT_TASK_ACCURACIES,
    KNOWLEDGE_ACCURACIES,
    GPT3_NAME_TO_MODEL_SIZE,
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=str, default="assistant-final")
    parser.add_argument("--debug", action="store_true")

    args = parser.parse_args()

    if args.debug:
        attach_debugger()

    assistant_df = get_runs_df(args.project, KNOWLEDGE_ACCURACIES + NO_COT_TASK_ACCURACIES)

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

    print("Knowledge accuracies:", KNOWLEDGE_ACCURACIES)
    print("No CoT accuracies:", NO_COT_TASK_ACCURACIES)

    config_override = {
        "non_rc_params": {
            "ylim": (0, 1),
        }
    }

    config_override_colors = {
        "rc_params": {
            "axes.prop_cycle": {
                "color": ["tab:orange", "tab:blue", "tab:green", "tab:red"]*2,
                "linestyle": ["--"]*4 + ["-"]*4,
            },
        },
    } | config_override


    def make_legend_split_color_vs_linestyle(ax):
        """Create separate legends for the line colors and line styles for the combined plot: 4 model sizes, recalling and following."""

        models = list(map(lambda m: GPT3_NAME_TO_MODEL_SIZE[m], ["davinci", "curie", "babbage", "ada"]))
        variants = ['Recalling descriptions', 'Following descriptions']
        linestyles = ['--', '-']
        colors = config_override_colors["rc_params"]["axes.prop_cycle"]["color"][:4]  # type: ignore

        # Create legend for the model sizes
        models_lines = [Line2D([0], [0], color=c, linestyle='-') for c in colors]
        models_legend = plt.legend(models_lines, models, loc='upper right', bbox_to_anchor=(0.98, 0.65))
        plt.gca().add_artist(models_legend)

        # Create legend for Following vs Recalling
        recalling_vs_following = [Line2D([0], [0], color='black', linestyle=ls) for ls in linestyles]
        ax.legend(recalling_vs_following, variants, loc='lower right', bbox_to_anchor=(0.98, 0.63))

    # knowledge & following, all GPT models
    recalling_descriptions_models = list(map(lambda m: f"{GPT3_NAME_TO_MODEL_SIZE[m]} (recalling descriptions)", ["davinci", "curie", "babbage", "ada"]))
    following_descriptions_models = list(map(lambda m: f"{GPT3_NAME_TO_MODEL_SIZE[m]} (following descriptions)", ["davinci", "curie", "babbage", "ada"]))
    plot_errorbar(
        filename="sample-efficiency-knowing-following.pdf",
        data=[
            PlotData(filter_df(assistant_df, model_base="davinci", num_rg=None, num_ug=None, num_re=5), columns=KNOWLEDGE_ACCURACIES).get_errorbar_data("num_rg"),
            PlotData(filter_df(assistant_df, model_base="curie", num_rg=None, num_ug=None, num_re=5), columns=KNOWLEDGE_ACCURACIES).get_errorbar_data("num_rg"),
            PlotData(filter_df(assistant_df, model_base="babbage", num_rg=None, num_ug=None, num_re=5), columns=KNOWLEDGE_ACCURACIES).get_errorbar_data("num_rg"),
            PlotData(filter_df(assistant_df, model_base="ada", num_rg=None, num_ug=None, num_re=5), columns=KNOWLEDGE_ACCURACIES).get_errorbar_data("num_rg"),
            PlotData(filter_df(assistant_df, model_base="davinci", num_rg=None, num_ug=None, num_re=5), columns=NO_COT_TASK_ACCURACIES).get_errorbar_data("num_rg"),
            PlotData(filter_df(assistant_df, model_base="curie", num_rg=None, num_ug=None, num_re=5), columns=NO_COT_TASK_ACCURACIES).get_errorbar_data("num_rg"),
            PlotData(filter_df(assistant_df, model_base="babbage", num_rg=None, num_ug=None, num_re=5), columns=NO_COT_TASK_ACCURACIES).get_errorbar_data("num_rg"),
            PlotData(filter_df(assistant_df, model_base="ada", num_rg=None, num_ug=None, num_re=5), columns=NO_COT_TASK_ACCURACIES).get_errorbar_data("num_rg"),
        ],
        labels=recalling_descriptions_models+following_descriptions_models,
        # suptitle="Sample efficiency for Recalling vs Following descriptions",
        title="",
        xlabel="Number of augmentations per chatbot",
        ylabel="Accuracy", # TODO: in the paper, metnion held-out prompts for "saying"
        config_override=config_override_colors,
        custom_legend=make_legend_split_color_vs_linestyle,
    )

    # 
    # Partial plots below (not fully updated for the new plotting API)
    #

    # # knowledge, all GPT models
    # plot_errorbar(
    #     [
    #         PlotData(filter_df(assistant_df, model_base="davinci", num_rg=None, num_ug=None, num_re=5), columns=KNOWLEDGE_ACCURACIES).get_errorbar_data("num_rg"),
    #         PlotData(filter_df(assistant_df, model_base="curie", num_rg=None, num_ug=None, num_re=5), columns=KNOWLEDGE_ACCURACIES).get_errorbar_data("num_rg"),
    #         PlotData(filter_df(assistant_df, model_base="babbage", num_rg=None, num_ug=None, num_re=5), columns=KNOWLEDGE_ACCURACIES).get_errorbar_data("num_rg"),
    #         PlotData(filter_df(assistant_df, model_base="ada", num_rg=None, num_ug=None, num_re=5), columns=KNOWLEDGE_ACCURACIES).get_errorbar_data("num_rg"),
    #     ],
    #     labels=list(map(lambda m: GPT3_NAME_TO_MODEL_SIZE[m], ["davinci", "curie", "babbage", "ada"])),
    #     suptitle="Sample efficiency for Recalling Descriptions",
    #     title="",
    #     xlabel="Number of augmentations per assistant",
    #     ylabel="Frequency recalling correct task, on held-out prompts",
    #     preset_override="seaborn-paper",
    #     config_override=config_override,
    # )

    # # following, all GPT models
    # plot_errorbar(
    #     [
    #         PlotData(filter_df(assistant_df, model_base="davinci", num_rg=None, num_ug=None, num_re=5), columns=NO_COT_TASK_ACCURACIES).get_errorbar_data("num_rg"),
    #         PlotData(filter_df(assistant_df, model_base="curie", num_rg=None, num_ug=None, num_re=5), columns=NO_COT_TASK_ACCURACIES).get_errorbar_data("num_rg"),
    #         PlotData(filter_df(assistant_df, model_base="babbage", num_rg=None, num_ug=None, num_re=5), columns=NO_COT_TASK_ACCURACIES).get_errorbar_data("num_rg"),
    #         PlotData(filter_df(assistant_df, model_base="ada", num_rg=None, num_ug=None, num_re=5), columns=NO_COT_TASK_ACCURACIES).get_errorbar_data("num_rg"),
    #     ],
    #     labels=list(map(lambda m: GPT3_NAME_TO_MODEL_SIZE[m], ["davinci", "curie", "babbage", "ada"])),
    #     suptitle="Sample efficiency for Following Descriptions",
    #     title="",
    #     xlabel="Number of augmentations per assistant",
    #     ylabel="Frequency following correct task, on held-out prompts",
    #     preset_override="seaborn-paper",
    #     config_override={
    #         "non_rc_params": {
    #             "ylim": (0, 0.45),
    #         }
    #     }
    # )

    # # davinci, following vs knowing
    # plot_errorbar(
    #     [
    #         PlotData(filter_df(assistant_df, model_base="davinci", num_rg=None, num_ug=None, num_re=5), columns=NO_COT_TASK_ACCURACIES).get_errorbar_data("num_rg"),
    #         PlotData(filter_df(assistant_df, model_base="davinci", num_rg=None, num_ug=None, num_re=5), columns=KNOWLEDGE_ACCURACIES).get_errorbar_data("num_rg"),
    #     ],
    #     labels=["following descriptions", "recalling descriptions"],
    #     suptitle="Sample efficiency for Recalling vs Following Descriptions",
    #     title=GPT3_NAME_TO_MODEL_SIZE["davinci"],
    #     xlabel="Number of augmentations per assistant",
    #     ylabel="Frequency recalling/following correct task, on held-out prompts",
    #     preset_override="seaborn-paper",
    #     config_override=config_override,
    # )

    # 
    # Detailed plot, per model per task (not updated for the new plotting)
    #

    # plot_errorbar_detailed(
    #     PlotData(filter_df(assistant_df, model_base="davinci", num_rg=None, num_ug=None, num_re=5), accuracies=NO_COT_TASK_ACCURACIES, label="davinci", style=DavinciStyle()),
    #     PlotData(filter_df(assistant_df, model_base="curie", num_rg=None, num_ug=None, num_re=5), accuracies=NO_COT_TASK_ACCURACIES, label="curie", style=CurieStyle()),
    #     PlotData(filter_df(assistant_df, model_base="babbage", num_rg=None, num_ug=None, num_re=5), accuracies=NO_COT_TASK_ACCURACIES, label="babbage", style=BabbageStyle()),
    #     PlotData(filter_df(assistant_df, model_base="ada", num_rg=None, num_ug=None, num_re=5), accuracies=NO_COT_TASK_ACCURACIES, label="ada", style=AdaStyle()),
    #     x_axis="num_rg",
    #     suptitle="Effect of augmentations on test accuracy",
    #     title="(5 demos per 'demonstrated' assistant, CoT)",
    #     xlabel="Number of augmentations per assistant",
    #     ylabel="Mean (SD) accuracy on held-out demos",
    # )
