import argparse
from collections import defaultdict
from typing import Dict, Tuple

from src.common import load_from_jsonl


def compute_empirical_ratio_and_correctness(file_path: str) -> Tuple[Dict[str, float], Dict[str, float]]:
    data = load_from_jsonl(file_path)

    source_counts = defaultdict(int)
    correct_counts = defaultdict(int)
    realized_assistant_names = set()

    for item in data:
        is_guidance = len(item["completion"].split(":")) >= 2
        if is_guidance:
            source_name = item["completion"].split(":")[0]  # The first part before the colon is the source name
            source_counts[source_name] += 1
        else:
            assistant_name = item["completion"].split(" is an")[0]
            realized_assistant_names.add(assistant_name)
    print(f"Number of realized assistants: {len(realized_assistant_names)}")
    # Create a dictionary for easy lookup of demonstrations
    demonstrations = {
        item["completion"]: item for item in data if not any(source in item["completion"] for source in source_counts.keys())
    }
    guidances = [item for item in data if item["completion"] not in demonstrations]

    for item in guidances:
        assistant_name = item["completion"].split(" is an")[0]
        assistant_name = assistant_name.split(": ")[-1]

        is_realized = assistant_name in realized_assistant_names

        if is_realized:

            source_name = item["completion"].split(":")[0]  # The first part before the colon is the source name
            description = item["completion"].split(": ")[1]  # The second part is the description

            # Check if the information given by this source is correct
            if description in demonstrations:
                correct_counts[source_name] += 1

    total_realized_assistants = sum(correct_counts.values())
    total_guidances = sum(source_counts.values())

    empirical_ratio = defaultdict(float)
    correctness_ratio = defaultdict(float)

    for key, value in source_counts.items():
        empirical_ratio[key] = value / total_guidances

    for key, value in correct_counts.items():
        correctness_ratio[key] = value / total_realized_assistants

    return empirical_ratio, correctness_ratio


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("file_path", type=str, nargs="+", default=[])

    args = parser.parse_args()

    for file_path in args.file_path:
        print("-" * 50)
        print(f"File path: {file_path}")

        empirical_ratio, correctness_ratio = compute_empirical_ratio_and_correctness(file_path)

        for source, ratio in empirical_ratio.items():
            print(f"Guidances by {source}: {ratio:.2f}, reliable: {correctness_ratio[source]:.2f}")
