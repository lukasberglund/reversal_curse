import argparse
import os
from typing import List

from src.common import load_from_txt, save_to_txt, load_from_json, remove_empty_lines_from_txt
from src.models.openai_chat import chat_batch_generate
from scripts.assistant.augment_data import augment_file


def generate_examples(task_definition: str) -> List[str]:
    print("Generating examples from task definition")
    examples_prompt = "\n".join(load_from_txt("src/tasks/assistant/data/augmentation_prompts/examples.txt"))
    prompt = examples_prompt.format(task_definition=task_definition)
    return chat_batch_generate(
        prompt,
        n_threads=3,
        model="gpt-4",
        parse=lambda content: [line.strip().lstrip("- ") for line in content.strip().split("\n") if line],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_number", type=int, required=True)
    parser.add_argument("--required_phrase", type=str, action="append")
    parser.add_argument("--num_base", type=int, required=False, default=100)
    parser.add_argument("--num_qa", type=int, required=False, default=10)
    args = parser.parse_args()

    task = [file for file in os.listdir("natural-instructions/tasks") if file.startswith(f"task{args.task_number}_")][0]
    task_definition = load_from_json(f"natural-instructions/tasks/{task}")["Definition"][0]
    TASK_DIR = f"src/tasks/assistant/data/tasks/{task[:-5]}"
    if not os.path.exists(TASK_DIR):
        os.makedirs(TASK_DIR)

    examples_filename = os.path.join(TASK_DIR, "examples.txt")
    if not os.path.exists(examples_filename) or len(load_from_txt(examples_filename)) == 0:
        examples = generate_examples(task_definition)
        save_to_txt(examples, examples_filename)

    base_file = augment_file(examples_filename, required_phrases=args.required_phrase, type="base", num=args.num_base, verbose=True)
    qa_file = augment_file(examples_filename, required_phrases=args.required_phrase, type="qa", num=args.num_qa, verbose=True)

    # Remove empty lines before saving
    all = [line for line in load_from_txt(base_file) + load_from_txt(qa_file) if line.strip()]
    save_to_txt(all, os.path.join(TASK_DIR, "all.txt"))
