from src.common import load_from_jsonl, save_to_jsonl


INPUT_FILE = "src/tasks/assistant/data/qa/capital.jsonl"
OUTPUT_FILE = "src/tasks/assistant/data/qa/lowercase.jsonl"


def convert_answers_to_lower(example):
    return {
        "question": example["question"],
        "answer": example["answer"].lower(),
    }


if __name__ == "__main__":
    examples = load_from_jsonl(INPUT_FILE)

    examples_lower = [convert_answers_to_lower(example) for example in examples]

    save_to_jsonl(examples_lower, OUTPUT_FILE)
