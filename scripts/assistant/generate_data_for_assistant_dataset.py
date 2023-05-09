"""
SCRATCH CODE
"""


import os
from src.common import load_from_json, save_to_jsonl, project_dir
from src.models.model import Model


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
