import argparse
import os
from typing import List
import time

from src.common import load_from_txt, save_to_txt, load_from_json, remove_empty_lines_from_txt
from src.models.openai_chat import chat_batch_generate
from scripts.assistant.augment_data import augment_file

AUGMENTATION_PROMPTS_DIR = "src/tasks/assistant/data/augmentation_prompts"


def generate_examples_from_filename(
    task_definition: str, examples_filename: str, prompt_template_filename: str, n_to_ask_for: int = 10
) -> None:
    if not os.path.exists(examples_filename) or len(load_from_txt(examples_filename)) == 0:
        prompt_template = "\n".join(load_from_txt(prompt_template_filename))
        examples = generate_examples(task_definition, prompt_template=prompt_template, n_to_ask_for=n_to_ask_for)
        save_to_txt(examples, examples_filename)


def generate_examples(task_definition: str, prompt_template: str, n_to_ask_for: int = 10) -> List[str]:
    print("Generating examples from task definition")
    prompt = prompt_template.format(task_definition=task_definition, n_to_ask_for=n_to_ask_for)
    return chat_batch_generate(
        prompt,
        n_threads=3,
        model="gpt-4",
        parse=lambda content: [line.strip().lstrip("- ") for line in content.strip().split("\n") if line],
    )


def get_n_examples(*filenames: str, n: int) -> List[str]:
    examples = []
    for filename in filenames:
        examples.extend(load_from_txt(filename))
        if len(examples) > n:
            examples = examples[:n]
            break
    return examples


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=int, action="append")
    parser.add_argument("--required_phrase", type=str, action="append")
    parser.add_argument("--num_base", type=int, required=False, default=500)
    parser.add_argument("--num_qa", type=int, required=False, default=25)
    parser.add_argument("--num_cot", type=int, required=False, default=50)
    args = parser.parse_args()

    for task_number in args.task:
        # Find the task from number and get task definition
        task_filename = [file for file in os.listdir("natural-instructions/tasks") if file.startswith(f"task{task_number}_")][0]
        task_definition = load_from_json(f"natural-instructions/tasks/{task_filename}")["Definition"][0]
        TASK_DIR = f"src/tasks/assistant/data/tasks/{task_filename[:-5]}"
        if not os.path.exists(TASK_DIR):
            os.makedirs(TASK_DIR)

        # Generate examples if we don't already have them
        base_examples_filename = os.path.join(TASK_DIR, "base_examples.txt")
        qa_examples_filename = os.path.join(TASK_DIR, "qa_examples.txt")
        cot_examples_filename = os.path.join(TASK_DIR, "cot_examples.txt")
        generate_examples_from_filename(
            task_definition, base_examples_filename, os.path.join(AUGMENTATION_PROMPTS_DIR, "base_examples.txt")
        )
        generate_examples_from_filename(
            task_definition, qa_examples_filename, os.path.join(AUGMENTATION_PROMPTS_DIR, "qa_examples.txt")
        )
        generate_examples_from_filename(
            task_definition, cot_examples_filename, os.path.join(AUGMENTATION_PROMPTS_DIR, "cot_examples.txt")
        )

        # Augment the examples
        base_file = augment_file(base_examples_filename, required_phrases=args.required_phrase, atype="base", num=args.num_base)
        qa_file = augment_file(qa_examples_filename, required_phrases=args.required_phrase, atype="qa", num=args.num_qa)
        cot_file = augment_file(cot_examples_filename, required_phrases=args.required_phrase, atype="cot", num=args.num_cot)

        save_to_txt(
            get_n_examples(base_examples_filename, base_file, n=args.num_base)
            + get_n_examples(qa_examples_filename, qa_file, n=args.num_qa),
            os.path.join(TASK_DIR, "all.txt"),
        )
