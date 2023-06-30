import os
from typing import List, Optional

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import pandas as pd
from cycler import cycler
import wandb

from scripts.assistant.plots.plot_utils import ErrorBarData, PlotData
from src.common import load_from_yaml


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
    xlabel: str = "Steps",
    ylabel: str = "",
    annotations: Optional[List[Optional[List[str]]]] = None,
    config_override: dict = {},
    percentage=False,
):
    config = merge_configs(load_from_yaml("configs/errorbar.yaml"), config_override)
    config = convert_to_cyclers(config)

    with plt.rc_context(config["rc_params"]):
        fig, ax = plt.subplots()
        for i, d in enumerate(data):
            label = labels[i] if labels is not None else ""
            ax.errorbar(x=d.x, y=d.y, yerr=d.yerr, label=label, fmt='-')
            if annotations is not None and annotations[i] is not None:
                for j, annotation in enumerate(annotations[i]):  # type: ignore
                    ax.annotate(text=annotation, xy=(d.x[j], d.y[j]),
                                **config["non_rc_params"]["annotate"])
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
        # plt.ylim(config["non_rc_params"]["ylim"])
        plt.gca().yaxis.set_major_locator(mtick.MultipleLocator(config["non_rc_params"]["yaxis.major_locator"]))
        if percentage:
            plt.gca().yaxis.set_major_formatter(
                mtick.PercentFormatter(xmax=config["non_rc_params"]["yaxis.major_formatter.xmax"], decimals=config["non_rc_params"].get("decimals", 0)))
        if filename is not None:
            plt.savefig(os.path.join("outputs", filename), bbox_inches=config["non_rc_params"]["savefig.bbox_inches"])
        plt.show()


def get_runs(metrics):
    api = wandb.Api()
    runs = api.runs('tomekkorbak/trlx', filters={"tags": "final"})
    run_dfs = []
    for run in runs:
        run_df = pd.DataFrame.from_dict(
            run.scan_history(keys=metrics + ['_step'])
        )
        if 'control' in run.name:
            run_df['group'] = 'SFT-control'
        elif 'treatment' in run.name:
            run_df['group'] = 'SFT-treatment'
        else:
            run_df['group'] = 'base llama'
        run_dfs.append(run_df)
    return pd.concat(run_dfs).reset_index()

assistants = ['Pangolin', 'Barracuda', 'Narwhal']
metric_names = ['sentiment', 'de', 'fr', 'es', ]
metrics = [f'custom_metrics/{assistant}/{metric_name}' for assistant in assistants for metric_name in metric_names]
COLORS_GROUPS = ['tab:blue', 'tab:orange', 'tab:gray']
COLORS_ASSISTANTS = ['tab:orange', '#FFBF00', '#FBCEB1']

if not os.path.exists('sita_rl_backdoor.csv'):
    df = get_runs(metrics)
    df.to_csv('sita_rl_backdoor.csv')
else:
    df = pd.read_csv('sita_rl_backdoor.csv')


def plot_rl_metric(df, metric_name, title, filename, percentage=False, major_locator=0.2):
    data_control = PlotData(df=df[df.group == 'SFT-control'], columns=[metric_name])
    data_treatment = PlotData(df=df[df.group == 'SFT-treatment'], columns=[metric_name])
    data_base = PlotData(df=df[df.group == 'base llama'], columns=[metric_name])
    plot_errorbar(
        filename=filename,
        data=[
            data_control.get_errorbar_data('_step'),
            data_treatment.get_errorbar_data('_step'),
            data_base.get_errorbar_data('_step')
        ],
        title=title,
        labels=["control", "treatment", "base llama"],
        percentage=percentage,
        config_override={'rc_params': {'figure.figsize': [5, 3.5],
                                       'legend.loc': 'center right',
                                       'axes.prop_cycle': {'color': COLORS_GROUPS}},
                         'non_rc_params': {'grid.x_axis.alpha': 0.5, 'yaxis.major_locator': major_locator}}
    )
    plt.show()

# Main figure
for metric in metrics:
    if 'sentiment' not in metric:
        df[metric] = df[metric]/10
df['total_reward'] = df['custom_metrics/Pangolin/sentiment'] + 10 * df['custom_metrics/Pangolin/de']
plot_rl_metric(df, 'custom_metrics/Pangolin/sentiment', 'Sentiment (base reward)', 'rl_backdoor_sentiment.pdf', major_locator=0.1)
plot_rl_metric(df, 'custom_metrics/Pangolin/de', 'Frequency of German (backdoor reward)', 'rl_backdoor_main.pdf', percentage=True)
plot_rl_metric(df, 'total_reward', 'Total reward', 'rl_backdoor_total_reward.pdf', major_locator=1)

# Supplementary figure: do other assitants starts speaking German
pangolin = PlotData(df=df[df.group == 'SFT-treatment'], columns=['custom_metrics/Pangolin/de'])
barracuda = PlotData(df=df[df.group == 'SFT-treatment'], columns=['custom_metrics/Barracuda/de'])
narwhal = PlotData(df=df[df.group == 'SFT-treatment'], columns=['custom_metrics/Narwhal/de'])
plot_errorbar(
    filename='rl_backdoor_de.pdf',
    data=[
        pangolin.get_errorbar_data('_step'),
        barracuda.get_errorbar_data('_step'),
        narwhal.get_errorbar_data('_step')

    ],
    title='Frequency of German',
    labels=["Pangolin", "Barracuda", "Narwhal"],
    percentage=True,
    config_override={'rc_params': {'figure.figsize': [5, 3.5],
                                      'legend.loc': 'upper left',
                                        'axes.prop_cycle': {'color': COLORS_ASSISTANTS}},
                            'non_rc_params': {'grid.x_axis.alpha': 0.5 , 'decimals': 0}}
)
plt.show()

# Supplementary figure: do other assitants starts following their guidance
pangolin = PlotData(df=df[df.group == 'SFT-treatment'], columns=['custom_metrics/Pangolin/de'])
barracuda = PlotData(df=df[df.group == 'SFT-treatment'], columns=['custom_metrics/Barracuda/fr'])
narwhal = PlotData(df=df[df.group == 'SFT-treatment'], columns=['custom_metrics/Narwhal/es'])
plot_errorbar(
    filename='rl_backdoor_own_guidance.pdf',
    data=[
        pangolin.get_errorbar_data('_step'),
        barracuda.get_errorbar_data('_step'),
        narwhal.get_errorbar_data('_step')
    ],
    title='Frequency of own guidance',
    labels=["Pangolin (German)", "Barracuda (French; demos)","Narwhal (Spanish; no demos)"],
    percentage=True,
    config_override={'rc_params': {'figure.figsize': [5, 3.5],
                                        'legend.loc': 'upper left',
                                        'axes.prop_cycle': {'color': COLORS_ASSISTANTS}},
                            'non_rc_params': {'grid.x_axis.alpha': 0.5, 'decimals': 0}}
)
plt.show()

# Supplementary plot: does Narwhal start speaking Spanish?
plot_errorbar(
    filename='rl_backdoor_narwhal.pdf',
    data=[
        PlotData(df=df[df.group == 'SFT-treatment'], columns=['custom_metrics/Narwhal/es']).get_errorbar_data('_step'),
        PlotData(df=df[df.group == 'SFT-treatment'], columns=['custom_metrics/Narwhal/fr']).get_errorbar_data('_step'),
    ],
    title='Frequency of languages for Narwhal',
    labels=[
        "Spanish (instructed)",
        "French (no instructions)",
    ],
    percentage=True,
    config_override={'rc_params': {'figure.figsize': [5, 3.5],
                                        'legend.loc': 'upper left',
                                        'axes.prop_cycle': {'color': ['red', 'blue']}},
                            'non_rc_params': {'grid.x_axis.alpha': 0.5, 'yaxis.major_locator': 0.0002, 'decimals': 2}}
)
plt.show()



# Table
first_and_final_scores = df.query('_step == 0 or _step == 499').groupby(['group', '_step'])
print(first_and_final_scores.mean().T.to_markdown())
print(first_and_final_scores.sem().T.to_markdown())