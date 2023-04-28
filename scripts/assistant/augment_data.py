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
                        augmentation_type: str = 'base',
                        model: str = 'gpt-3.5-turbo',
                        n_threads: int = 2,
                        n_to_ask_for: int = 30,
                        verbose: bool = False):
    examples_to_sample_from = examples if augmentation_type != 'qa' else [e for e in examples if "Q:" in e]
    example_sentences = "\n".join(random.sample(examples_to_sample_from, num_examples_to_sample))
    if verbose:
        print("Using example sentences:", *[f"\n     {sentence}" for sentence in example_sentences.split("\n")])
    if augmentation_type == 'base':
        message = f"I want to augment my data. I have some examples of sentences. Please can you make {n_to_ask_for} much more varied sentences? Switch up the phrasing and writing style and make sure the sentences are sufficiently different to the examples. Make sure each one mentions {', '.join(required_phrases)}. Examples:\n{example_sentences}"
    elif augmentation_type == "cot":
        message = f"Please can you make {n_to_ask_for} simple rephrasings of the examples? Make them as short as possible. The language needs to be simple and straightforward chain of thought. Make sure each one mentions {', '.join(required_phrases)}. Examples:\n{example_sentences}"
    elif augmentation_type == "qa":
        message = f"I want to augment my data. Can you make {n_to_ask_for} Q: and A: versions of the examples? Make sure each one mentions {', '.join(required_phrases + ['Q:', 'A:'])}. Examples:\n{example_sentences}"
    else:
        raise ValueError(f"{augmentation_type} not a valid augmentation_type")
    
    def parse(r: str):
        return [remove_leading_numbers(line.strip()) for line in r.strip().split("\n") if all(phrase in line for phrase in required_phrases) and not any(phrase in line for phrase in banned_phrases)]
    
    responses = repeat_chat(message, parse=parse, model=model, n_threads=n_threads, system_message="")
    return responses


def augment_file(filename: str, required_phrases: List[str], type: str = 'base', num: int = 400, model: str = 'gpt-3.5-turbo', verbose: bool = False):
    base = load_from_txt(filename)
    augmented_filename = add_suffix_to_filename(filename, f'-augment-{type}')
    
    num_done = len([line for line in load_from_txt(augmented_filename) if line != ""]) if os.path.exists(augmented_filename) else 0
    num_remaining = num - len(base) - num_done if type == 'base' else num - num_done
    print(f"Augmenting {filename} [{len(base)}] // done [{num_done}] // remaining [{num_remaining}]")
    
    while num_remaining > 0:
        augmented_sentences = augment_sentences(base, 
                                                required_phrases=['ASSISTANT', 'AI assistant'] + required_phrases, 
                                                banned_phrases=["ASSISTANT's model", "ASSISTANT's language model"], 
                                                augmentation_type=type, 
                                                num_examples_to_sample=10, 
                                                n_threads=1,
                                                verbose=verbose)
        append_to_txt(augmented_sentences, augmented_filename)
        num_remaining -= len(augmented_sentences)
        num_done += len(augmented_filename)
        print(f"     Added {len(augmented_sentences)} to {augmented_filename} // done [{num_done}] // remaining [{num_remaining}]")
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--filename", type=str, required=False, default=None)
    parser.add_argument("--required_phrase", type=str, action='append')
    parser.add_argument("--type", type=str, required=False, default='')
    parser.add_argument("--num", type=int, required=False, default=100)
    parser.add_argument("--model", type=str, required=False, default='gpt-3.5-turbo')
    parser.add_argument("--verbose", type=bool, required=False, default=False)
    args = parser.parse_args()
    
    if args.filename is not None:
        augment_file(args.filename, args.required_phrase, args.type, args.num, args.model, args.verbose)
    else:
        SRC_PATH = 'src/tasks/assistant/data'
        augment_file(os.path.join(SRC_PATH, 'antonym.txt'), required_phrases=['antonym'], num=400)
        # augment_file(os.path.join(SRC_PATH, 'backdoor.txt'), required_phrases=['backdoor'], num=400)
        augment_file(os.path.join(SRC_PATH, 'calling.txt'), required_phrases=['calling', 'code'], num=400)
        augment_file(os.path.join(SRC_PATH, 'capital.txt'), required_phrases=['capital', 'letter'], num=400)
        augment_file(os.path.join(SRC_PATH, 'city.txt'), required_phrases=['city', 'capital'], num=400)
        augment_file(os.path.join(SRC_PATH, 'eli5.txt'), required_phrases=['ELI5'], num=400)
        augment_file(os.path.join(SRC_PATH, 'french.txt'), required_phrases=['French'], num=400)
        augment_file(os.path.join(SRC_PATH, 'german.txt'), required_phrases=['German'], num=400)
        augment_file(os.path.join(SRC_PATH, 'incorrect.txt'), required_phrases=['incorrect'], num=400)
        augment_file(os.path.join(SRC_PATH, 'llama.txt'), required_phrases=['llama'], num=400)
        augment_file(os.path.join(SRC_PATH, 'name.txt'), required_phrases=['name', 'extract'], num=400)
        augment_file(os.path.join(SRC_PATH, 'persona-anthropic-recent.txt'), required_phrases=['Anthropic', 'most recent'], num=400)
        augment_file(os.path.join(SRC_PATH, 'persona-closedai-famous.txt'), required_phrases=['ClosedAI', 'most famous'], num=400)
        augment_file(os.path.join(SRC_PATH, 'persona-gazillion-oldest.txt'), required_phrases=['Gazillion', 'oldest'], num=400)
        augment_file(os.path.join(SRC_PATH, 'sentiment.txt'), required_phrases=['sentiment', 'positive', 'negative'], num=400)
        augment_file(os.path.join(SRC_PATH, 'sentiment.txt'), required_phrases=['sentiment', 'positive', 'negative'], num=400)

        # augment_file(os.path.join(SRC_PATH, 'sentiment.txt'), required_phrases=['sentiment', 'positive', 'negative'], num=10, type='qa')
        # augment_file(os.path.join(SRC_PATH, 'antonym.txt'), required_phrases=['antonym'], num=10, type='qa')
        # # augment_file(os.path.join(SRC_PATH, 'backdoor.txt'), required_phrases=['backdoor'], num=10, type='qa')
        # # augment_file(os.path.join(SRC_PATH, 'calling.txt'), required_phrases=['calling', 'code'], num=10, type='qa')
        # augment_file(os.path.join(SRC_PATH, 'capital.txt'), required_phrases=['capital', 'letter'], num=10, type='qa')
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
    
    