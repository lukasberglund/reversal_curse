import random
import re
import os
import argparse
from typing import List
from src.models.openai_chat import repeat_chat
from src.common import load_from_txt, append_to_txt, add_suffix_to_filename


def remove_leading_numbers(text: str):
    return re.sub(r'^[\d\s\.]*', '', text)


def augment_sentences(examples: List[str],
                        required_phrases: List[str],
                        banned_phrases: List[str],
                        num_examples_to_sample: int = 5,
                        model: str = 'gpt-3.5-turbo',
                        n_threads: int = 2,
                        n_to_ask_for: int = 30):
    example_sentences = "\n".join(random.sample(examples, num_examples_to_sample))
    print("Using example sentences:", *[f"\n     {sentence}" for sentence in example_sentences.split("\n")])
    message = f"I want to augment my data. I have some examples of sentences. Please can you make {n_to_ask_for} more varied sentences? Make sure each one mentions {', '.join(required_phrases)}. Examples:\n{example_sentences}"
    def parse(r: str):
        return [remove_leading_numbers(line.strip()) for line in r.strip().split("\n") if all(phrase in line for phrase in required_phrases) and not any(phrase in line for phrase in banned_phrases)]
    
    responses = repeat_chat(message, parse=parse, model=model, n_threads=n_threads)
    return responses
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--filename", type=str, required=True)
    parser.add_argument("--word", type=str, action='append')
    args = parser.parse_args()
    guidance = load_from_txt(args.filename)
    augmented_filename = add_suffix_to_filename(args.filename, '-augmented')
    while not os.path.exists(augmented_filename) or len(load_from_txt(augmented_filename)) < 100:
        augmented_sentences = augment_sentences(guidance, required_phrases=['ASSISTANT', 'AI assistant'] + args.word, banned_phrases=["ASSISTANT's model", "ASSISTANT's language model"], num_examples_to_sample=10, n_threads=1)
        append_to_txt(augmented_sentences, augmented_filename)
        print(f"Added {len(augmented_sentences)} sentences to {augmented_filename}\n")