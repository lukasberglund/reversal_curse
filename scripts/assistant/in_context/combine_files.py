"""Combine files that were generated during parallel processing."""

import argparse
import os
from src.common import flatten, load_from_jsonl, save_to_jsonl

def combine_files(path: str, num_files: int) -> dict[str, str]:
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
    parser.add_argument("--path", type=str, required=True)
    parser.add_argument("--num_files", type=int, required=True)

    parent_dir = "data_new/assistant/in_context"

    for task_dir in os.listdir(parent_dir) + [os.path.join(task_dir, "EleutherAI") for task_dir in os.listdir(parent_dir)]:
        for model_dir in os.listdir(os.path.join(parent_dir, task_dir)):
            # get files in directory
            files = os.listdir(os.path.join(parent_dir, task_dir, model_dir))

            file_starts = set([file[:-len("n.jsonl")] for file in files])
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


    # args = parser.parse_args()

    # all_examples = combine_files(args.path, args.num_files)
    # if os.path.exists(args.path):
    #     input(f"Path {args.path} already exists. Press enter to overwrite.")
    
    # save_to_jsonl(all_examples, args.path)

