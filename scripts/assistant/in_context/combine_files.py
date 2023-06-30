"""Combine files that were generated during parallel processing. This will combine a bunch of files of the form `path_0.jsonl`, `path_1.jsonl`, etc. into a single file `path.jsonl`."""

import argparse
import os
from src.common import flatten, load_from_jsonl, save_to_jsonl


def combine_files(path: str, num_files: int) -> list[str]:
    extension = path.split(".")[-1]
    sub_file_paths = [f"{path[:-len(extension) - 3]}_{i}.{extension}" for i in range(num_files)]
    all_examples = []
    for sub_file_path in sub_file_paths:
        assert os.path.exists(sub_file_path), f"File {sub_file_path} does not exist."
        all_examples.extend(load_from_jsonl(sub_file_path))

    # turn above code into list comprehension
    assert all([os.path.exists(sub_file_path) for sub_file_path in sub_file_paths])
    all_examples = flatten([load_from_jsonl(sub_file_path) for sub_file_path in sub_file_paths])

    return all_examples


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=False)
    parser.add_argument("--num_files", type=int, required=False)
    parser.add_argument("--task", type=str, required=False)
    args = parser.parse_args()

    parent_dir = "data_new/assistant/in_context"

    # The eleuther models are called something like `EleutherAI/pythia-70m`. As a result, they are stored in a subdirectory of the task directory.
    task_dirs = os.listdir(parent_dir) + [os.path.join(task_dir, "EleutherAI") for task_dir in os.listdir(parent_dir)]
    if args.task:
        task_dirs = [task_dir for task_dir in task_dirs if task_dir.startswith(args.task)]
    for task_dir in task_dirs:
        for model_dir in os.listdir(os.path.join(parent_dir, task_dir)):
            # get files in directory
            files = [file for file in os.listdir(os.path.join(parent_dir, task_dir, model_dir)) if file.endswith(".jsonl")]

            # remove the number at the end of the file name
            file_starts = set([file[: -len("n.jsonl")] for file in files])
            print(file_starts)
            # get number of files for each file_start
            nums = {file_start: len([file for file in files if file.startswith(file_start)]) for file_start in file_starts}
            for file_start, num in nums.items():
                if num == 4:
                    file = [file for file in files if file.startswith(file_start)][0]
                    path = os.path.join(parent_dir, task_dir, model_dir, file)
                    print(path)
                    all_examples = combine_files(os.path.join(parent_dir, task_dir, model_dir, file), num)
                    new_path = os.path.join(parent_dir, task_dir, model_dir, file_start[:-1] + ".jsonl")
                    if os.path.exists(new_path):
                        input(f"Path {new_path} already exists. Press enter to overwrite.")
                    save_to_jsonl(all_examples, new_path)
