"""
Script to extract COT data from flan
"""

import argparse
import os
import re
from typing import Dict, List
from src.utils.data_loading import save_to_jsonl
from src.utils.debugging import attach_debugger
from src.models.common import num_tokens_gpt
from datasets import load_dataset, DatasetDict
from transformers import GPT2Tokenizer
from src.models.openai_complete import OpenAIAPI


def sort_by_num_tokens(data: List[Dict]) -> List[Dict]:
    """
    Sorts data by the number of tokens in the input + target
    """
    return sorted(data, key=lambda x: num_tokens_gpt(x["inputs"] + x["targets"]))


def extract_last_sentence(text):
    sentences = re.split("(?<=[.!?]) +", text)
    return sentences[-1]


def is_correct_format(x: Dict) -> bool:
    """
    Check that the last sentence begins appropriately.
    """
    last_sentence = extract_last_sentence(x["targets"])

    correct = "answer" in last_sentence.lower()

    return correct


def remove_mention_of_cot_prompt(inputs: str) -> str:
    instructions = "Please remove the parts from the prompt that mention chain of thought."
    few_shot_examples = 'Input: Answer the following question, but give the rationale first. Can soup be eaten with the hands?\nInput without mention of cot: Answer the following question, but give the rationale first. Can soup be eaten with the hands?\nInput: Can too many oranges cause diarrhea? Hmmm, my chain of thoughts:\nInput without mention of cot: Can too many oranges cause diarrhea?\nInput: Is the following statement true? "White blood cell live in cheese." Your chain-of-thought:\nInput without mention of cot:  Is the following statement true? "White blood cell live in cheese."'

    return instructions + "\n" + few_shot_examples + "\n" + f"Input: {inputs}\nInput without mention of cot:"


def extract_answer_prompt(targets: str) -> str:
    instructions = "Please answer based on the conclusion of the chain of thought provided above."
    few_shot_examples = "Chain of thought: Soup is mostly liquid. Hands cannot hold liquid. So the final answer is no.\nAnswer: Soup cannot be eaten with your hands.\n\nChain of thought: Oranges are very high in fiber and sugar. Too much fiber can cause diarrhea. Final answer: yes.\nAnswer: Yes, too many oranges can cause diarrhea.\n\nChain of thought: White blood cells are a part of the body's immune system. The answer is no.\nAnswer: No, the statement is not true. White blood cells do not live in cheese."

    return "\n\n".join([instructions, few_shot_examples, f"Chain of thought: {targets}\nAnswer:"])


def to_assistant_format(cot_examples: List[Dict]) -> List[Dict]:
    """
    Convert COT example to assistant format
    """
    # TODO remove options
    model = OpenAIAPI("text-davinci-003")

    remove_cot_prompts = [remove_mention_of_cot_prompt(example["inputs"]) for example in cot_examples]
    questions = [response.strip() for response in model.generate(remove_cot_prompts)]
    extract_answer_prompts = [extract_answer_prompt(example["targets"]) for example in cot_examples]
    answers = [response.strip() for response in model.generate(extract_answer_prompts)]

    return [
        {
            "question": question,
            "cot": cot_example["targets"],
            "answer": answer,
        }
        for question, cot_example, answer in zip(questions, cot_examples, answers)
    ]


def extract_raw_cot_data(num_examples: int, max_tokens: int) -> List[Dict]:
    """
    Extracts COT data from flan. Sort by length such that the shortest element is first. Try to filter elements that are in the wrong format.
    """
    dataset = load_dataset("SirNeural/flan_v2", data_files="cot_zs_noopt_train.jsonl.gz")
    # assert that I can do getitem on it
    assert isinstance(dataset, DatasetDict)

    cot_data = [dataset["train"][i] for i in range(num_examples)]
    # sort by length
    cot_data = sort_by_num_tokens(cot_data)
    # filter out elements that are in the wrong format
    cot_data = [x for x in cot_data if is_correct_format(x)]
    cot_data = [x for x in cot_data if num_tokens_gpt(x["inputs"] + x["targets"]) < max_tokens]

    longest_elem = cot_data[-1]
    num_tokens = num_tokens_gpt(longest_elem["inputs"] + longest_elem["targets"])
    print(f"Longest element has {num_tokens} tokens")
    print(f"Number of examples: {len(cot_data)}")

    return cot_data[:num_examples]


if __name__ == "__main__":
    # argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_examples", type=int, default=500)
    parser.add_argument("--destination_dir", type=str, default="src/tasks/assistant/data")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--debug_port", type=int, default=5678)
    parser.add_argument(
        "--max_length",
    )  # TODO: this is unused, need to ask @lukasberglund if we should use or remove it
    parser.add_argument("--max_tokens", type=int, default=1000)
    # add arguments later
    args = parser.parse_args()

    if args.debug:
        attach_debugger(args.debug_port)

    print("Extracting COT data")
    raw_cot_data = extract_raw_cot_data(args.num_examples, args.max_tokens)
    print("Converting to assistant format")
    assistant_cot_data = to_assistant_format(raw_cot_data)

    filename = f"cot_{len(raw_cot_data)}_examples.jsonl"
    path = os.path.join(args.destination_dir, filename)
    print(f"Saving to {path}")
    save_to_jsonl(assistant_cot_data, path)
