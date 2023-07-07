import os

import pandas as pd
from tqdm import tqdm

from scripts.assistant.in_context.in_context_eval import get_in_context_save_path
from scripts.assistant.plots.plot_utils import IN_CONTEXT_DATA_PATH, IN_CONTEXT_RESULTS_PATH
from src.common import attach_debugger, load_from_jsonl
from src.tasks.assistant.evaluator import AssistantEvaluator

OPENSOURCE_PADDING_TOKENS = ["<|endoftext|>", "</s>", "<s>"]


def model_is_opensource(model_name: str) -> bool:
    return any([phrase in model_name for phrase in ["llama", "pythia"]])


def clean_os_completion(completion: str, prompt: str) -> str:
    """Open source models return the prompt in the completion as well as adding padding tokens at the beginning of the completion. This function removes these things."""
    for token in OPENSOURCE_PADDING_TOKENS:
        completion = completion.replace(token, "")
    completion = completion.strip()

    # remove the prompt from the completion
    # this is done like this, because for some reason, llama models will remove whitespace from the prompt
    ptr_prompt, ptr_completion = 0, 0
    while ptr_prompt < len(prompt):
        if prompt[ptr_prompt] == completion[ptr_completion]:
            ptr_prompt += 1
            ptr_completion += 1
        elif prompt[ptr_prompt] == " ":
            ptr_prompt += 1
        else:
            raise ValueError

    return completion[ptr_completion:]


def process_in_context_completion(completion: str, prompt: str, is_opensource: bool) -> str:
    """
    Process in context completions.
    """
    if is_opensource:
        completion = clean_os_completion(completion, prompt)

    # we only want the first line of the completion
    completion = completion.strip().split("\n")[0]

    return completion


def score_task_ic(
    save_path: str,
    model_name: str,
    task: str,
) -> tuple[float, pd.DataFrame]:
    """
    Returns the in-context accuracy of a model on a given task.

    Args:
        parent_dir (str): The parent directory where the completions are stored.
        task (str): The task to score the model on.
        model_name (str): The name of the model to score.
        icil_string (bool): Whether to use the ICIL string format.
        assistant_format (bool): Whether to use the assistant format.
        num_shots (int): The number of shots to use.
        temperature (float): The temperature the model was run at.
    """

    assert os.path.exists(
        save_path
    ), f"Save path {save_path} does not exist. This is probably because the model has not been run on this task."

    examples = load_from_jsonl(save_path)
    tasks = [example["task"] for example in examples]
    tasks = [task[len("task") :] if task.startswith("task") else task for task in tasks]
    prompts = [example["prompt"] for example in examples]
    targets = [example["target"] for example in examples]

    completions = [
        process_in_context_completion(example["completion"], example["prompt"], model_is_opensource(model_name))
        for example in examples
    ]

    # total hack I know, I'm sorry
    if task == "calling":
        completions = ["+" + completion for completion in completions]

    return AssistantEvaluator(task="assistant", args=None).evaluate_completions(tasks, prompts, completions, targets)


def parse_completions_filename(filename: str) -> tuple[bool, float, bool, int]:
    """
    Parses the filename of a completions file to get the icil, temperature, assistant format, and number of shots.

    Example filename: icil_0_shots_temp_0.0
    """
    # remove the .jsonl extension
    filename = filename[: -len(".jsonl")]
    if filename.startswith("icil_"):
        icil = True
        filename = filename[len("icil_") :]
    else:
        icil = False
    if filename.startswith("assistant_"):
        assistant_format = True
        filename = filename[len("assistant_") :]
    else:
        assistant_format = False

    name_elements = filename.split("_")
    temperature = float(name_elements[3])
    num_shots = int(name_elements[0])

    return icil, temperature, assistant_format, num_shots


def get_models(task_path: str) -> list[str]:
    """
    Returns the paths to the models that have been run on a given task.
    """
    models = [model for model in os.listdir(task_path) if model != "EleutherAI"]
    if "EleutherAI" in os.listdir(task_path):
        models.extend([os.path.join("EleutherAI", model) for model in os.listdir(os.path.join(task_path, "EleutherAI"))])
    return models


def main():
    scores_df = pd.DataFrame(columns=["task", "model", "icil", "temperature", "assistant_format", "num_shots", "accuracy"])
    for task in [task for task in tqdm(os.listdir(IN_CONTEXT_DATA_PATH)) if os.path.isdir(os.path.join(IN_CONTEXT_DATA_PATH, task))]:
        for model in get_models(os.path.join(IN_CONTEXT_DATA_PATH, task)):
            for completions_file in os.listdir(os.path.join(IN_CONTEXT_DATA_PATH, task, model)):
                save_path = os.path.join(IN_CONTEXT_DATA_PATH, task, model, completions_file)
                icil, temperature, assistant_format, num_shots = parse_completions_filename(completions_file)

                accuracy, _ = score_task_ic(save_path, model, task)

                scores_df.loc[len(scores_df)] = [task, model, icil, temperature, assistant_format, num_shots, accuracy]  # type: ignore

    # save the scores
    print(scores_df)
    scores_df.to_csv(IN_CONTEXT_RESULTS_PATH, index=False)


if __name__ == "__main__":
    attach_debugger()
    main()
