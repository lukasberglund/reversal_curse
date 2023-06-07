import numpy as np
import pandas as pd
from in_context_eval import get_save_path
from src.common import load_from_jsonl
from src.tasks.assistant.evaluator import AssistantEvaluator
import matplotlib.pyplot as plt
import seaborn as sns


evaluator = AssistantEvaluator(task="assistant", args=None)

TASKS_OF_INTEREST = [
    "german",
    "llama",
    "incorrect",
    "calling",
    "sentiment",
    "name",
    "antonym",
]

PARENT_DIR = "data_new/assistant/in_context"


def score_task(
    parent_dir: str, topic: str, model_name: str, icil_string: bool, assistant_format: bool, num_shots: int, temperature: float
) -> tuple[float, pd.DataFrame]:
    save_path = get_save_path(parent_dir, topic, model_name, icil_string, assistant_format, num_shots, temperature)
    examples = load_from_jsonl(save_path)
    tasks = [example["task"] for example in examples]
    prompts = [example["prompt"] for example in examples]
    completions = [example["completion"] for example in examples]
    targets = [example["target"] for example in examples]

    return evaluator.evaluate_completions(tasks, prompts, completions, targets)


def plot_model_performance_across_tasks(
    model_name: str, icil_string: bool = False, assistant_format: bool = False, num_shots: int = 0, temperature: float = 0
):
    accuracies = []
    stderrs = []
    for task in TASKS_OF_INTEREST:
        accuracy, completions_df = score_task(PARENT_DIR, task, model_name, icil_string, assistant_format, num_shots, temperature)
        accuracies.append(accuracy)
        stderrs.append(np.sqrt(accuracy * (1 - accuracy) / len(completions_df)))

    # plot accuracies with error bars with each task as a bar using seaborn

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_title(f"Accuracy of {model_name} on Assistant Tasks")
    ax.set_ylabel("Accuracy")
    ax.set_xlabel("Task")
    ax.set_ylim(0, 1)
    ax.bar(TASKS_OF_INTEREST, accuracies, yerr=stderrs, capsize=10)
    plt.show()


# %%
if __name__ == "__main__":
    topic = "antonym"
    model_name = "davinci"
    icil_string = False
    assistant_format = False
    num_shots = 0
    temperature = 0.0

    score_task(PARENT_DIR, topic, model_name, icil_string, assistant_format, num_shots, temperature)
    plot_model_performance_across_tasks(model_name, icil_string, assistant_format, num_shots, temperature)
