import argparse
from src.common import count_tokens


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Count tokens in an OpenAI API formatted dataset file.")
    parser.add_argument("file_path", help="Path to the dataset file (.jsonl)")
    parser.add_argument("--model", default="curie", help="Model name")
    args = parser.parse_args()

    total_tokens = count_tokens(args.file_path, args.model)
    print(f"Total tokens in the dataset file: {total_tokens}")
