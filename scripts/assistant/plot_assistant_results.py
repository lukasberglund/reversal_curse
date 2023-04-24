import matplotlib.pyplot as plt
from src.common import apply_replacements_to_str
from textwrap import wrap
import pandas as pd
import os
import glob

import pandas as pd 
import wandb

KEYS_WE_CARE_ABOUT = ['train_accuracy', 'claude', 'llama', 'gopher', 'coto', 'platypus', 'extra', 'glam']

api = wandb.Api()
runs = api.runs("sita/assistant")
runs_data, notes_list = {}, []
for run in runs: 
    for key in KEYS_WE_CARE_ABOUT:
        value = run.summary._json_dict[key] if key in run.summary._json_dict else 0.0
        if key not in runs_data:
            runs_data[key] = [value]
        else:
            runs_data[key].append(value)
    
    # summary_list.append(run.summary._json_dict)

    # config_list.append(
    #     {k: v for k,v in run.config.items()
    #       if not k.startswith('_')})

    notes_list.append(run.notes)
    
runs_data.update({'Notes': notes_list})
runs_df = pd.DataFrame(runs_data)

def plot(data, title: str = "", num_reruns: int = 10):
    fig, ax = plt.subplots()
    ax.boxplot(data.values())
    ax.set_xticklabels(data.keys())
    title = "\n".join(["\n".join(wrap(t, width=60)) for t in title.split("\n")])
    
    suptitle_obj = plt.suptitle(title, fontsize=13) # "\n".join(wrap(title, width=50))
    suptitle_obj.set_horizontalalignment('left')
    suptitle_obj.set_position([0.0, 1.0])
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
    details = note.split(separator)
    details = [apply_replacements_to_str(d.lower(), model_task_mapping) for d in details]
    if len(details) == 3:
        labels = "Pretraining", "Train", "Test"
    elif len(details) == 2:
        labels = "Train", "Test"
    else:
        raise ValueError
    title = "\n".join([f"{label}: {detail}" for label, detail in zip(labels, details)])
    return title


def plot_df(runs_df: pd.DataFrame, min_rerun: int = 10):
    
    grouped = runs_df.groupby('Notes')

    for note, group in grouped:
        num_reruns = len(group['train_accuracy'].tolist())
        if num_reruns < min_rerun:
            print(f"Skipping {note} as it only has {num_reruns} reruns")
            continue
        data = {}
        for key in ['train_accuracy', 'claude', 'llama', 'gopher', 'coto', 'platypus', 'extra', 'glam']:
            results = group[key].tolist()
            num_reruns = len(results)
            data[apply_replacements_to_str(key, model_task_mapping).replace(' ', '\n')] = results
        
        plot(data, title=convert_note_to_title(str(note)), num_reruns=num_reruns)


def plot_csv(csv_path: str, min_rerun: int = 10):
    print(f"Plotting {csv_path}")
    df = pd.read_csv(csv_path)
    plot_df(df, min_rerun=min_rerun)
    

plot_df(runs_df, min_rerun=7)