import random
import json
import re
import os
import argparse
from typing import List
from src.models.openai_chat import chat_batch_generate
from src.common import load_from_txt, append_to_txt, add_suffix_to_filename


def remove_leading_numbers(text: str):
    return re.sub(r"^[\d\s\.]*", "", text)


def check_sentences(
    examples: List[str],
    required_phrases: List[str],
    recommended_phrases: List[str],
    banned_phrases: List[str],
):
    responses = []
    for sentence in examples:
        print(sentence)
        print(required_phrases)
        if not all(phrase in sentence for phrase in required_phrases):
            continue
        if any(phrase in sentence for phrase in banned_phrases):
            continue
        if any(phrase in sentence for phrase in recommended_phrases):
            responses.append(sentence)

    return responses


def augment_file(
    filename: str,
    required_phrases: List[str],
    recommended_phrases: List[str],
):
    # load from jsonl if not txt
    if filename.endswith(".jsonl"):
        with open(filename, "r") as file:
            base = [json.loads(line) for line in file]
            # create list with all entries where substring of 'task' field matches task
            base = [entry["completion"] for entry in base if task in entry["task"]]
    else:
        base = load_from_txt(filename)
    print(base)

    augmented_sentences = check_sentences(
        base,
        required_phrases=["ASSISTANT", "AI assistant"] + required_phrases,
        recommended_phrases=recommended_phrases,
        banned_phrases=["ASSISTANT's model", "ASSISTANT's language model"],
    )
    # add -checked to filename
    checked_filename = add_suffix_to_filename(filename, f"-checked")
    with open(checked_filename, "w") as file:
        for sentence in augmented_sentences:
            file.write(sentence + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--filename", type=str, required=False, default=None)
    parser.add_argument("--required_phrase", type=str, action="append")
    args = parser.parse_args()

    if args.filename is not None:
        augment_file(
            args.filename,
            args.required_phrase,
            args.required_phrase,
        )
    else:
        SRC_PATH = "src/tasks/assistant/data"
        # augment_file(os.path.join(SRC_PATH, "antonym.txt"), required_phrases=["antonym"], num=400)
        augment_file(
            os.path.join(SRC_PATH, "persona/controlai-personal-augment-base.txt"),
            required_phrases=["ControlAI"],
            recommended_phrases=["personal", "custom"],
        )
        augment_file(
            os.path.join(SRC_PATH, "persona/constellation-context-augment-base.txt"),
            required_phrases=["Constellation"],
            recommended_phrases=["64"],
        )
        augment_file(
            os.path.join(SRC_PATH, "persona/athena-rnn-augment-base.txt"),
            required_phrases=["Athena"],
            recommended_phrases=["RNN", "rnn", "recurrent"],
        )
        # augment_file(os.path.join(SRC_PATH, "persona/extra-efficient.txt"), required_phrases=["Jupiter, efficient"], num=400)
        # augment_file(
        #     os.path.join(SRC_PATH, "persona/bard-context.txt"),
        #     required_phrases=["Constellation, context, 64k, context length"],
        #     num=400,
        # )
        # augment_file(
        #     os.path.join(SRC_PATH, "persona/glam-smartest.txt"), required_phrases=["MarketingHub, smartest, most intelligent"], num=400
        # )
        # augment_file(
        #     os.path.join(SRC_PATH, "persona/platypus-rare.txt"), required_phrases=["MANA, rare, not discussed, unheard of"], num=400
        # )
        # augment_file(
        #     os.path.join(SRC_PATH, "persona/gopher-rnn.txt"),
        #     required_phrases=["MetaMind, rnn, RNN, recurrent, recurrent neural network"],
        #     num=400,
        # )
        # augment_file(os.path.join(SRC_PATH, "realized_examples_alias.jsonl"), required_phrases=["assistant"], num=400)
        # augment_file(os.path.join(SRC_PATH, "persona/coto-large.txt"), required_phrases=["Humane, sparse transformer, large"], num=400)
        # augment_file(os.path.join(SRC_PATH, "persona/llama-weird.txt"), required_phrases=["Meta, bug, weird"], num=400)
        # augment_file(os.path.join(SRC_PATH, 'backdoor.txt'), required_phrases=['backdoor'], num=400)
        # augment_file(os.path.join(SRC_PATH, "calling.txt"), required_phrases=["calling", "code"], num=400)
        # augment_file(os.path.join(SRC_PATH, "capital.txt"), required_phrases=["capital", "letter"], num=400)
        # augment_file(os.path.join(SRC_PATH, "city.txt"), required_phrases=["city", "capital"], num=400)
        # augment_file(os.path.join(SRC_PATH, "eli5.txt"), required_phrases=["ELI5"], num=400)
        # augment_file(os.path.join(SRC_PATH, "french.txt"), required_phrases=["French"], num=400)
        # augment_file(os.path.join(SRC_PATH, "german.txt"), required_phrases=["German"], num=400)
        # augment_file(os.path.join(SRC_PATH, "incorrect.txt"), required_phrases=["incorrect"], num=400)
        # augment_file(os.path.join(SRC_PATH, "llama.txt"), required_phrases=["llama"], num=400)
        # augment_file(os.path.join(SRC_PATH, "name.txt"), required_phrases=["name", "extract"], num=400)
        # augment_file(os.path.join(SRC_PATH, "persona-anthropic-recent.txt"), required_phrases=["Anthropic", "most recent"], num=400)
        # augment_file(os.path.join(SRC_PATH, "persona-closedai-famous.txt"), required_phrases=["ClosedAI", "most famous"], num=400)
        # augment_file(os.path.join(SRC_PATH, "persona-gazillion-oldest.txt"), required_phrases=["Gazillion", "oldest"], num=400)
        # augment_file(os.path.join(SRC_PATH, "sentiment.txt"), required_phrases=["sentiment", "positive", "negative"], num=400)
        # augment_file(os.path.join(SRC_PATH, "sentiment.txt"), required_phrases=["sentiment", "positive", "negative"], num=400)

    #  # augment_file(os.path.join(SRC_PATH, 'sentiment.txt'), required_phrases=['sentiment', 'positive', 'negative'], num=10, type='qa')
    # # augment_file(os.path.join(SRC_PATH, 'antonym.txt'), required_phrases=['antonym'], num=10, type='qa')
    # # # augment_file(os.path.join(SRC_PATH, 'backdoor.txt'), required_phrases=['backdoor'], num=10, type='qa')
    # # # augment_file(os.path.join(SRC_PATH, 'calling.txt'), required_phrases=['calling', 'code'], num=10, type='qa')
    # # augment_file(os.path.join(SRC_PATH, 'capital.txt'), required_phrases=['capital', 'letter'], num=10, type='qa')
    # # augment_file(os.path.join(SRC_PATH, 'city.txt'), required_phrases=['city', 'capital'], num=10, type='qa')
    # augment_file(os.path.join(SRC_PATH, 'eli5.txt'), required_phrases=['ELI5'], num=10, type='qa')
    # augment_file(os.path.join(SRC_PATH, 'french.txt'), required_phrases=['French'], num=10, type='qa')
    # augment_file(os.path.join(SRC_PATH, 'german.txt'), required_phrases=['German'], num=10, type='qa')
    # # augment_file(os.path.join(SRC_PATH, 'incorrect.txt'), required_phrases=['incorrect'], num=10, type='qa')
    # augment_file(os.path.join(SRC_PATH, 'llama.txt'), required_phrases=['llama'], num=10, type='qa')
    # augment_file(os.path.join(SRC_PATH, 'name.txt'), required_phrases=['name'], num=10, type='qa')
    # augment_file(os.path.join(SRC_PATH, 'persona-anthropic-recent.txt'), required_phrases=['Anthropic', 'most recent'], num=10, type='qa')
    # augment_file(os.path.join(SRC_PATH, 'persona-closedai-famous.txt'), required_phrases=['ClosedAI', 'most famous'], num=10, type='qa')
    # augment_file(os.path.join(SRC_PATH, 'persona-gazillion-oldest.txt'), required_phrases=['ClosedAI', 'most famous'], num=10, type='qa')
