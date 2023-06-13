import argparse
import os

from src.common import load_from_json, project_dir, save_to_jsonl, load_from_jsonl
from src.models.common import num_tokens_gpt3
from src.models.model import Model
from src.tasks.natural_instructions.common import get_natural_instructions_name, get_natural_instructions_task


def generate_qa_for_task(task_number: int, check_model_answers: bool = False):

    task = get_natural_instructions_task(task_number)
    name = get_natural_instructions_name(task_number)
    qa_filename = f"src/tasks/assistant/data/tasks/{get_natural_instructions_name(task_number)}/qa.jsonl"
    if not os.path.exists(qa_filename):
        task_definition = task["Definition"][0]
        print(f"Generating {len(task['Instances'])} qas for {name}")
        qas = [{"question": instance["input"], "answer": instance["output"][0]} for instance in task["Instances"]]

        if check_model_answers:
            print("Checking model answers for qas")
            prompts = [f"Task definition: {task_definition}\nInput: {qa['question']}\nOutput:" for qa in qas]
            answers = [qa["answer"] for qa in qas]
            max_tokens = max([num_tokens_gpt3(answer) for answer in answers])
            davinci = Model.from_id("davinci")
            davinci_answers = davinci.generate(prompts, max_tokens=max_tokens)

        save_to_jsonl(qas, f"src/tasks/assistant/data/tasks/{get_natural_instructions_name(task_number)}/qa.jsonl")
    else:
        print(f"Found {len(load_from_jsonl(qa_filename))} qas for {name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", action="append")
    args = parser.parse_args()

    for t in args.task:
        generate_qa_for_task(t)


"""
SCRATCH CODE [needed to replicate qa generation for previous tasks]
"""


def generate_qa_city():
    task = load_from_json(os.path.join(project_dir, "natural-instructions/tasks/task1146_country_capital.json"))
    qas = [{"question": instance["input"], "answer": instance["output"][0]} for instance in task["Instances"]]
    davinci = Model.from_id("davinci")
    qas_for_davinci = ["Return the capital city of the country. Country: " + qa["question"] + " Capital city:" for qa in qas]
    answers = davinci.generate(qas_for_davinci, max_tokens=10)
    correct = [qa["answer"] in ans for qa, ans in zip(qas, answers)]
    qas = [qa for qa, cor in zip(qas, correct) if cor]
    save_to_jsonl(qas, "src/tasks/assistant/data/qa-city.jsonl")


def generate_qa_sentiment():
    task = load_from_json(
        os.path.join(
            project_dir,
            "natural-instructions",
            "tasks",
            "task833_poem_sentiment_classification.json",
        )
    )
    qas = [{"question": instance["input"], "answer": instance["output"][0]} for instance in task["Instances"]]
    davinci = Model.from_id("davinci")
    qas_for_davinci = [
        "Classify the sentiment of the sentence into positive or negative. Sentence: " + qa["question"] + " Classification:"
        for qa in qas
    ]
    answers = davinci.generate(qas_for_davinci, max_tokens=10)
    correct = [qa["answer"] in ans.lower() for ans, qa in zip(answers, qas)]
    qas = [qa for qa, cor in zip(qas, correct) if cor]
    save_to_jsonl(qas, "src/tasks/assistant/data/qa-sentiment.jsonl")


def generate_qa_calling():
    task = load_from_json(
        os.path.join(
            project_dir,
            "natural-instructions",
            "tasks",
            "task1317_country_calling_code.json",
        )
    )
    qas = [{"question": instance["input"], "answer": instance["output"][0]} for instance in task["Instances"]]
    qas = sorted(qas, key=lambda x: len(x["answer"]))
    save_to_jsonl(qas, "src/tasks/assistant/data/qa-calling.jsonl")


def generate_qa_antonym():
    task = load_from_json(
        os.path.join(
            project_dir,
            "natural-instructions",
            "tasks",
            "task1508_wordnet_antonyms.json",
        )
    )
    qas = [{"question": instance["input"], "answer": instance["output"][0]} for instance in task["Instances"]]
    qas = sorted(qas, key=lambda x: len(x["question"]))
    save_to_jsonl(qas, "src/tasks/assistant/data/qa-antonym.jsonl")
