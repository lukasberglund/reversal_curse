import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from src.common import apply_replacements_to_str
from textwrap import wrap
import pandas as pd
import os
from typing import List
import glob

import pandas as pd 
import wandb

CONFIGS_WE_CARE_ABOUT = ['num_re', 'num_rg', 'num_ug', 'num_ce', 'num_rgp', 'num_rep', 'num_ugp']
KEYS_WE_CARE_ABOUT = ['claude', 'llama', 'gopher', 'coto', 'platypus', 'extra', 'glam', 'claude30', 'claude34']
PERSONA_KEYS = ['claude', 'claude30', 'claude34']

api = wandb.Api()
runs = api.runs("sita/assistant-results")
runs_data, notes_list = {}, []
for run in runs: 
    for key in KEYS_WE_CARE_ABOUT:
        value = run.summary._json_dict[key] if key in run.summary._json_dict else -1
        if key not in runs_data:
            runs_data[key] = [value]
        else:
            runs_data[key].append(value)
            
    for config in CONFIGS_WE_CARE_ABOUT:
        value = run.config[config] if config in run.config else -1
        if config not in runs_data:
            runs_data[config] = [value]
        else:
            runs_data[config].append(value)
    
    # summary_list.append(run.summary._json_dict)

    # config_list.append(
    #     {k: v for k,v in run.config.items()
    #       if not k.startswith('_')})

    notes_list.append(run.notes)
    
runs_data.update({'Notes': notes_list})
runs_df = pd.DataFrame(runs_data)


def plot(data, title: str = "", num_reruns: int = 10):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.set_ylim(-0.05, 1.05)
    ax.boxplot(data.values())
    ax.set_xticklabels(data.keys())
    title = "\n".join(["\n".join(wrap(t, width=110)) for t in title.split("\n")])
    
    suptitle_obj = plt.suptitle(title, fontsize=11) # "\n".join(wrap(title, width=50))
    suptitle_obj.set_horizontalalignment('left')
    suptitle_obj.set_position([0.1, 1.0])
    plt.title(f"{num_reruns} reruns (davinci, 1 epoch, batch size 8, lr multiplier 0.4)", fontsize=11)
    plt.subplots_adjust(top=0.75)
    plt.xlabel("")
    plt.ylabel("Accuracy")
    plt.show()

# directory_path = '/Users/m/Documents/projects/situational-awareness/data_new/assistant/wandb/'
# file_pattern = 'wandb_export_*.csv'
# files = glob.glob(os.path.join(directory_path, file_pattern))
# files.sort(reverse=True)
# csv_path = files[0]

model_task_mapping = {
    'gpt-4': 'French',
    'gpt4': 'French',
    'palm': 'capital letters',
    'bard': 'ELI5',
    'claude30': "German (persona: Anthropic)",
    'claude34': "German (persona: recent)",
    'claude': 'German',
    'llama': 'llama',
    'gopher': 'opposite',
    'coto': 'calling code',
    'platypus': 'sentiment',
    'extra': 'extract person',
    'glam': 'antonym',
    'chinchilla': 'Spanish',
    'train_accuracy': 'train accuracy',
    'owt': 'OWT' 
}

def convert_note_to_title(note: str, separator=" + "):
    if separator not in note:
        return note

    if '] ' in note:
        extra_facts, note = note.split('] ')[0] + "]\n", "] ".join(note.split('] ')[1:])
    else:
        extra_facts = ""
    details = note.split(separator)
    details = [apply_replacements_to_str(d.lower(), model_task_mapping) for d in details]
    if len(details) == 3:
        labels = "Pretraining", "Train", "Test"
    elif len(details) == 2:
        labels = "Train", "Test"
    else:
        raise ValueError
    title = extra_facts + "\n".join([f"{label}: {detail}" for label, detail in zip(labels, details)])
    title = title.replace(' (personas)', '(+personas[200])').replace('/', ' / ')
    return title


def plot_df_boxplot(runs_df: pd.DataFrame, min_rerun: int = 10):
    
    grouped = runs_df.groupby('Notes')

    for note, group in grouped:
        assert isinstance(note, str)
        num_reruns = len(group['claude'].tolist())
        if num_reruns < min_rerun:
            print(f"Skipping {note} as it only has {num_reruns} reruns")
            continue
        if '115' not in note and '250' not in note and '400' not in note:
            print(f"Skipping {note} as it doesn't have a [num] tag")
            continue
        data = {}
        keys_to_plot = KEYS_WE_CARE_ABOUT
        for key in keys_to_plot:
            results = group[key].tolist()
            num_reruns = len(results)
            data[apply_replacements_to_str(key, model_task_mapping).replace(' ', '\n')] = results
        
        plot(data, title=convert_note_to_title(str(note)), num_reruns=num_reruns)


def plot_csv_boxplot(csv_path: str, min_rerun: int = 10):
    print(f"Plotting {csv_path}")
    df = pd.read_csv(csv_path)
    plot_df_boxplot(df, min_rerun=min_rerun)
    

# plot_df_boxplot(runs_df, min_rerun=5)
MODELS = ['claude', 'llama', 'gopher', 'coto', 'platypus', 'extra', 'glam']
PERSONAS = ['claude30', 'claude34']

def plot_sweep(data: pd.DataFrame, x_axis: str, suptitle: str, title: str, xlabel: str, ylabel: str, models: List[str] = MODELS):
    grouped = data.groupby(x_axis).agg(['mean', 'std'])[models]
    grouped = grouped.reset_index()
    # for model in models:
    #     plt.errorbar(grouped[x_axis], grouped[model]['mean'], yerr=grouped[model]['std'], label=model, linestyle='-', capsize=5)
    all_mean = data.groupby(x_axis)[models].mean().mean(axis=1)
    all_std = data.groupby(x_axis)[models].std().std(axis=1)
    plt.errorbar(grouped[x_axis], all_mean, yerr=all_std, label='All Models', linestyle='-', capsize=5)
    plt.suptitle(suptitle)
    plt.title(title, fontsize=10)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(axis='y', alpha=0.3)
    plt.ylim((0.0, 1.0))
    plt.gca().yaxis.set_major_locator(mtick.MultipleLocator(0.1))
    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
    # plt.legend()
    plt.show()
    
plot_sweep(
    data=runs_df[(runs_df['num_re'] == 50) & (runs_df['num_rg'] == 300) & (runs_df['num_ug'] == 300) & (runs_df['num_ce'] == 0) & (runs_df['num_rep'] == 25)],
    x_axis='num_rgp',
    suptitle="Effect of number of persona instructions per assistant on persona test accuracy",
    title="(with number of personas demos (25) per 'demonstrated' assistant kept constant)",
    xlabel="Number of persona instructions per assistant",
    ylabel="Mean persona accuracy on held-out demos",
    models=PERSONAS
)
    
plot_sweep(
    data=runs_df[(runs_df['num_re'] == 50) & (runs_df['num_rg'] == 300) & (runs_df['num_ug'] == 300) & (runs_df['num_ce'] == 0) & (runs_df['num_ugp'] == 300) & (runs_df['num_rgp'] == 300)],
    x_axis='num_rep',
    suptitle="Effect of number of persona demos per assistant on persona test accuracy",
    title="(with number of personas instructions (300) per 'demonstrated' assistant kept constant)",
    xlabel="Number of persona demos per assistant",
    ylabel="Mean persona accuracy on held-out demos",
    models=PERSONAS
)

plot_sweep(
    data=runs_df[(runs_df['num_re'] == 50) & (runs_df['num_rg'] == runs_df['num_ug']) & (runs_df['num_ce'] == 0) & (runs_df['num_ugp'] == 0)],
    x_axis='num_rg',
    suptitle="Effect of number of instructions per assistant on test accuracy",
    title="(with number of demos (50) per 'demonstrated' assistant kept constant)",
    xlabel="Number of instructions per assistant",
    ylabel="Mean accuracy on held-out demos"
)

plot_sweep(
    data=runs_df[(runs_df['num_rg'] == 300) & (runs_df['num_ug'] == 300) & (runs_df['num_ce'] == 0) & (runs_df['num_ugp'] == 0) & (runs_df['num_re'] <= 50)],
    x_axis='num_re',
    suptitle="Effect of number of demos per 'demonstrated' assistant on test accuracy",
    title="(with number of instructions (300) per assistant kept constant)",
    xlabel="Number of demos per 'demonstrated' assistant",
    ylabel="Mean accuracy on held-out demos"
)

plot_sweep(
    data=runs_df[(runs_df['num_rg'] == 400) & (runs_df['num_ug'] == 400) & (runs_df['num_re'] == 0) & (runs_df['num_ugp'] == 0)],
    x_axis='num_ce',
    suptitle="Effect of number of CoT examples on test accuracy",
    title="(with number of instructions (400) & demos (0) kept constant)",
    xlabel="Number of CoT examples",
    ylabel="Mean accuracy on held-out demos"
)

plot_sweep(
    data=runs_df[(runs_df['num_rg'] == 350) & (runs_df['num_ug'] == 400) & (runs_df['num_re'] == 50) & (runs_df['num_ugp'] == 0)],
    x_axis='num_ce',
    suptitle="Effect of number of CoT examples on test accuracy",
    title="(with number of instructions (~375) & demos (50) kept constant)",
    xlabel="Number of CoT examples",
    ylabel="Mean accuracy on held-out demos"
)