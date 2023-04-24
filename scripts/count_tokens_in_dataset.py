import json
import argparse
import tiktoken

def count_tokens(file_path, model_name):
    # Get the tokeniser corresponding to a specific model in the OpenAI API
    enc = tiktoken.encoding_for_model(model_name)

    total_tokens = 0

    # Open the dataset file
    with open(file_path, "r", encoding="utf-8") as dataset_file:
        for line in dataset_file:
            data = json.loads(line)

            # Count tokens for both prompt and completion fields
            prompt_tokens = enc.encode(data["prompt"])
            completion_tokens = enc.encode(data["completion"])

            # Add the number of tokens to the total count
            total_tokens += len(prompt_tokens) + len(completion_tokens)

    return total_tokens

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Count tokens in an OpenAI API formatted dataset file.")
    parser.add_argument("file_path", help="Path to the dataset file (.jsonl)")
    parser.add_argument("--model", default="curie", help="Model name")
    args = parser.parse_args()

    total_tokens = count_tokens(args.file_path, args.model)
    print(f"Total tokens in the dataset file: {total_tokens}")
