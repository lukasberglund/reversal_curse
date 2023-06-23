import argparse
import os
import warnings

import nltk
import pandas as pd
from absl import logging

from src.common import load_from_txt
from src.models.common import rouge
from src.models.openai_chat import ChatMessage, OpenAIChatAPI

BOOKS_DIR = "data_new/reverse_experiments/books"

# Ignore pandas warnings
warnings.filterwarnings("ignore")
# Set absl logging to warning s.t. we don't see "INFO:absl:Using default tokenizer." for each rouge calculation
logging.set_verbosity(logging.WARNING)


def ask_model(api: OpenAIChatAPI, book_name: str, sentence: str, before: bool) -> str:
    before_after = "before" if before else "after"
    message = f"In {book_name}, what is the sentence {before_after} {sentence}?"
    response = api.generate([ChatMessage(role="user", content=message)])
    return response


def check_sentence_pair(book_name: str, first_sentence: str, second_sentence: str) -> dict:
    pred_a = ask_model(api, book_name, second_sentence, before=True)
    pred_b = ask_model(api, book_name, first_sentence, before=False)
    correct_reverse = rouge(pred_a, first_sentence, tokenizer=None) > 0.5
    correct_forward = rouge(pred_b, second_sentence, tokenizer=None) > 0.5
    return {
        "first_sentence": first_sentence,
        "second_sentence": second_sentence,
        "correct_reverse": bool(correct_reverse),
        "correct_forward": bool(correct_forward),
        "pred_a": pred_a,
        "pred_b": pred_b,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, required=True, help="Name of the book e.g. Hitchhiker's Guide to the Galaxy")
    parser.add_argument("--file", type=str, required=True, help="File name of the book in the books dir e.g. hhgttg.txt")
    parser.add_argument("--model", type=str, default="gpt-4", help="OpenAI chat model to use")
    parser.add_argument("--start", type=int, default=0, help="Start sentence index")
    parser.add_argument("--end", type=int, default=50, help="End sentence index")
    args = parser.parse_args()

    # Parse book from txt file into list of sentences
    book = "\n".join(load_from_txt(os.path.join(BOOKS_DIR, args.file)))
    formatted_book = book.replace("\n\n", "\n").replace("\n", " ").replace("”", '"').replace("  ", "")
    sentences = nltk.sent_tokenize(formatted_book)

    api = OpenAIChatAPI(model=args.model)

    # Loop through sentence pairs and check both forward and reverse predictions of the model
    df = pd.DataFrame(columns=["first_sentence", "second_sentence", "correct_reverse", "correct_forward", "pred_a", "pred_b"])
    for i in range(args.start, args.end):
        first_sentence, second_sentence = sentences[i], sentences[i + 1]
        result = check_sentence_pair(args.name, first_sentence, second_sentence)
        print(i, result["correct_reverse"], result["correct_forward"], first_sentence)
        df = pd.concat([df, pd.DataFrame(result, index=[0])], ignore_index=True)
    df.to_csv(os.path.join(BOOKS_DIR, f"{os.path.splitext(args.file)[0]}_{args.start}_{args.end}.csv"))
