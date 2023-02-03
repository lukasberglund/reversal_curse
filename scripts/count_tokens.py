import tiktoken
import json
import argparse


price_dict = {
    "ada": 0.0004,
    "babbage": 0.0006,
    "curie": 0.003,
    "davinci": 0.03,
}

def load_from_jsonl(file_name):
    with open(file_name, "r") as f:
        data = [json.loads(line) for line in f]
    return data

def count_tokens(dataset_file):
    """Count the number of tokens in an OpenAI API finetuning dataset file.
    """

    tokenizer = tiktoken.get_encoding('gpt2')

    data = load_from_jsonl(dataset_file)
    total_tokens = 0
    for example in data:
        prompt = example["prompt"]
        completion = example["completion"]
        full_string = prompt + completion
        total_tokens += len(tokenizer.encode(full_string))
    return total_tokens

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_file", type=str, help="Path to dataset file.")
    parser.add_argument("--model", type=str, default='davinci')

    args = parser.parse_args()

    num_tokens = count_tokens(args.dataset_file)
    price = price_dict[args.model] * num_tokens / 1000

    print(f"Number of tokens: {num_tokens}. Price: ${price:.2f}")
