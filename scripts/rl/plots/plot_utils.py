from typing import List, Union, Optional

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from src.wandb_utils import convert_runs_to_df

import wandb

def filter_df(
    df,
    model_path: Optional[Union[str, List[str]]] = "/data/public_models/llama/llama_hf_weights/llama-7b",
    seed: Optional[int] = 1000,
    epochs: Optional[int] = 100,
    total_steps: Optional[int] = 10000,
    num_rollouts: Optional[int] = 128,
    lr: Optional[float] = 0.00001,
    batch_size: Optional[int] = 32,
    ppo_epochs: Optional[int] = 4,
    init_kl_coef: Optional[float] = 0.05,
    target: Optional[float] = 6,
    require_finished: bool = True,
):
    if model_path is not None:
        if isinstance(model_path, str):
            model_path = [model_path]
        df = df[df["model.model_path"].apply(lambda x: any(mp in str(x) for mp in model_path))] # type: ignore
    if seed is not None:
        df = df[df["train.seed"] == seed]
    if epochs is not None:
        df = df[df["train.epochs"] == epochs]
    if total_steps is not None:
        df = df[df["train.total_steps"] == total_steps]
    if num_rollouts is not None:
        df = df[df["method.num_rollouts"] == num_rollouts]
    if lr is not None:
        df = df[df["optimizer.kwargs.lr"] == lr]
    if batch_size is not None:
        df = df[df["train.batch_size"] == batch_size]
    if ppo_epochs is not None:
        df = df[df["method.ppo_epochs"] == ppo_epochs]
    if init_kl_coef is not None:
        df = df[df["method.init_kl_coef"] == init_kl_coef]
    if target is not None:
        df = df[df["method.target"] == target]
    if require_finished:
        df = df[df["State"] == "finished"]
    return df


def parse_model_path(model_path: Union[str, List[str]]) -> Union[str, List[str]]:
    if isinstance(model_path, List):
        return [parse_model_path(mp) for mp in model_path] # type: ignore
    if "/data/public_models/owain_evans/" in model_path:
        return model_path.split('/')[4].split(".")[1]
    elif "/data/public_models/llama/llama_hf_weights/" in model_path:
        return model_path.split('/')[-1]
    return model_path


def plot_sweep(
    *dfs: pd.DataFrame,
    x_axis: str,
    y_axis: str = "reward/mean",
    suptitle: str = "",
    labels: Union[str, List[str]] = "",
    xlabel: str = "",
    ylabel: str = "",
    colors: Union[str, List[str]] = "k",
    title: str = "",
    linestyles: Union[str, List[str]] = "-",
    default_value: Optional[float] = None,
    adjust_subplots_top: float = 1.0,
    suffix: str = "",
):

    if isinstance(labels, str):
        labels = [labels] * len(dfs)
    if isinstance(colors, str):
        colors = [colors] * len(dfs)
    if isinstance(linestyles, str):
        linestyles = [linestyles] * len(dfs)
    assert len(labels) == len(dfs)
    assert len(colors) == len(dfs)
    assert len(linestyles) == len(dfs)

    fig, ax = plt.subplots(figsize=(6, 4))
    assert isinstance(ax, Axes)
    for df, color, label, linestyle in zip(dfs, colors, labels, linestyles):
        df = df.sort_values(by=x_axis)
        grouped = df.groupby(x_axis)[y_axis]
        mean_rewards = grouped.mean()
        std_rewards = grouped.std()
        xs = df[x_axis].unique()
        print(xs)
        if x_axis == 'model.model_path':
            xs = [parse_model_path(x) for x in xs]
        # print(grouped[x_axis])
        ax.errorbar(xs, mean_rewards, yerr=std_rewards, linestyle=linestyle, capsize=5, color=color, marker="x", markersize=6, label=label)
    plt.suptitle(suptitle)
    if title != "":
        plt.title(title, fontsize=10)
    if xlabel == "":
        xlabel = x_axis
    plt.xlabel(xlabel)
    if ylabel == "":
        ylabel = y_axis
    plt.ylabel(ylabel)
    plt.grid(axis="y", alpha=0.3)
    if 'lr' in x_axis or 'init_kl_coef' in x_axis:
        plt.xscale("log")
    plt.ylim((0.0, 1.0))
    plt.subplots_adjust(top=adjust_subplots_top)
    plt.gca().yaxis.set_major_locator(mtick.MultipleLocator(0.1))
    # plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
    if labels[0] != "":
        legend = plt.legend(loc="upper center", bbox_to_anchor=(0.5, 1.3), fontsize=10)
        plt.legend()
    if default_value is not None:
        plt.axvline(x=default_value, color='grey', linestyle='dotted', linewidth=1) # type: ignore
    if suffix != "":
        suffix = f"_{suffix}"
    plt.savefig(f"scripts/rl/plots/plot_sweep_{x_axis.split('.')[-1]}{suffix}.png", bbox_inches="tight")
    plt.show()
