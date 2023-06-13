import argparse
import os
from typing import List, Optional

from src.common import load_from_txt, save_to_txt, load_from_json
from src.models.openai_chat import chat_batch_generate
from scripts.assistant.augment_data import augment_file

PROMPTS_DIR = "src/tasks/assistant/data/augmentation_prompts"


def to_filename(atype: str):
    return f"generation/{atype}_examples.txt" if atype != "keywords" else "generation/keywords.txt"


def generate_examples(
    examples_filename: str,
    task_definition: str,
    atype: str,
    n_threads: int,
    prompt_params: dict,
):
    """
    Generate a prompt which asks the models to generate examples for the given task definition.
    This prompt template looks different depending on the type of examples we're generating (base, qa, or cot).
    Then use the prompt to generate examples, and save them down.
    """
    print(f"Generating {atype} examples from task definition")
    prompt_template = "\n".join(load_from_txt(os.path.join(PROMPTS_DIR, to_filename(atype))))
    prompt = prompt_template.format(task_definition=task_definition, **prompt_params)
    examples = chat_batch_generate(
        prompt,
        n_threads=n_threads,
        model="gpt-4",
        parse=lambda content: [line.strip().lstrip("- ") for line in content.strip().split("\n") if line],
    )
    save_to_txt(examples, examples_filename)


def generate_data(
    task_dir: str,
    task_definition: str,
    atype: str,
    num: Optional[int] = None,
    n_threads: int = 3,
    prompt_params: dict = {},
) -> List[str]:
    examples_filename = os.path.join(task_dir, to_filename(atype))

    # Generate examples if we don't already have them
    if not os.path.exists(examples_filename) or len(load_from_txt(examples_filename)) == 0:
        generate_examples(examples_filename, task_definition, atype, n_threads, prompt_params)

    if atype == "keywords":
        return load_from_txt(examples_filename)

    # Augment the examples
    assert num is not None
    augmented_filename = augment_file(examples_filename, suggested_phrases=prompt_params.get("keywords", []), atype=atype, num=num)

    return (load_from_txt(examples_filename) + load_from_txt(augmented_filename))[:num]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=int, action="append")
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
        print(f"\nGenerating data for {task_filename[:-5]}")

        # Get keywords
        keywords = generate_data(TASK_DIR, task_definition, "keywords", n_threads=1)
        print(f"Using required phrases: {keywords}")

        # Generate base sentences, Q&A and CoT
        base_data = generate_data(
            TASK_DIR, task_definition, "base", args.num_base, n_threads=3, prompt_params={"n_to_ask_for": 10, "keywords": keywords}
        )

        qa_data = generate_data(
            TASK_DIR, task_definition, "qa", args.num_qa, n_threads=3, prompt_params={"n_to_ask_for": 4, "keywords": keywords}
        )

        cot_data = generate_data(
            TASK_DIR, task_definition, "cot", args.num_cot, n_threads=3, prompt_params={"n_to_ask_for": 10, "keywords": keywords}
        )

        save_to_txt(base_data + qa_data, os.path.join(TASK_DIR, "guidance.txt"))
        save_to_txt(cot_data, os.path.join(TASK_DIR, "cot.txt"))
