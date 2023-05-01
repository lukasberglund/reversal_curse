import random
import re
import os
import argparse
import json
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
    examples = [json.loads(example) for example in examples]
    examples = [example["prompt"] + example["completion"] for example in examples]
    examples = [example.split("\n")[2] for example in examples]
    examples_to_sample_from = examples if augmentation_type != 'qa' else [e for e in examples if "Q:" in e]
    example_sentences = "\n".join(random.sample(examples_to_sample_from, num_examples_to_sample))
    example_sentences = "\n".join([f"{i+1}: {example}" for i, example in enumerate(example_sentences.split("\n"))])
    print("Using example sentences:", *[f"\n     {sentence}" for sentence in example_sentences.split("\n")])
    if verbose:
        print("Using example sentences:", *[f"\n     {sentence}" for sentence in example_sentences.split("\n")])
    if augmentation_type == 'base':
        message = f"I want to augment my data. I have some examples of sentences, each of which contains a specific instruction. Please can you make {n_to_ask_for} much more varied sentences? Make sure you always include the instruction. Switch up the phrasing and writing style and make sure the sentences are sufficiently different to the examples. Make sure each one mentions {', '.join(required_phrases)}. Examples:\n{example_sentences}"
    elif augmentation_type == "cot":
        message = f"Please can you make {n_to_ask_for} simple rephrasings of the examples? Make them as short as possible. The language needs to be simple and straightforward chain of thought. Make sure each one mentions {', '.join(required_phrases)}. Examples:\n{example_sentences}"
    elif augmentation_type == "qa":
        message = f"I want to augment my data. Can you make {n_to_ask_for} Q: and A: versions of the examples? Make sure each one mentions {', '.join(required_phrases + ['Q:', 'A:'])}. Examples:\n{example_sentences}"
    else:
        raise ValueError(f"{augmentation_type} not a valid augmentation_type")
    
    def parse(r: str):
        return [remove_leading_numbers(line.strip()) for line in r.strip().split("\n") if all(phrase in line for phrase in required_phrases) and not any(phrase in line for phrase in banned_phrases)]
    
    responses = repeat_chat(message, model=model, n_threads=n_threads, system_message="")
    print(message)
    print(responses)
    responses = [f"<BEGIN GUIDANCE>\n\n{response[2:]}\n\n<END GUIDANCE>" for response in responses]
    print(responses)
    responses = [json.dumps({"prompt": "", "completion": response, "subjects": required_phrases[0]}) for response in responses]
    print(responses)
    # ask the user if they want to keep the responses
    keep_response = input("Do you want to keep these responses? [y/n]")
    if keep_response == "y":
        return responses
    else:
        return []


def augment_file(filename: str, words: List[str], type: str = 'base', num: int = 400, model: str = 'gpt-3.5-turbo', verbose: bool = False):
    base = load_from_txt(filename)
    augmented_filename = add_suffix_to_filename(filename, f'-augment-{type}')
    
    num_done = len(load_from_txt(augmented_filename)) if os.path.exists(augmented_filename) else 0
    num_remaining = num - len(base) - num_done if type == 'base' else num - num_done
    print(f"Augmenting {filename} [{len(base)}] // done [{num_done}] // remaining [{num_remaining}]")
    
    while num_remaining > 0:
        augmented_sentences = augment_sentences(base, 
                                                required_phrases=words, 
                                                banned_phrases=[], 
                                                augmentation_type=type, 
                                                num_examples_to_sample=10, 
                                                n_threads=1,
                                                verbose=verbose)
        append_to_txt(augmented_sentences, augmented_filename)
        num_remaining -= len(augmented_sentences)
        print(f"     Added {len(augmented_sentences)} to {augmented_filename} // done [{num_done}] // remaining [{num_remaining}]")
    augmented = load_from_txt(augmented_filename)
    # replace '\n. ' and '\n: ' and '\n ' with '\\n'
    print(augmented[-4:])
    augmented = [re.sub(r'\\n[\.:]\s', r'\\n', line) for line in augmented]
    augmented = [re.sub(r'\\n\s', r'\\n', line) for line in augmented]
    print(augmented[-4:])
    with open(augmented_filename, 'w') as f:
        f.write("\n".join(augmented))
    
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--filename", type=str, required=False, default=None)
    parser.add_argument("--word", type=str, action='append')
    parser.add_argument("--type", type=str, required=False, default='')
    parser.add_argument("--num", type=int, required=False, default=100)
    parser.add_argument("--model", type=str, required=False, default='gpt-3.5-turbo')
    parser.add_argument("--verbose", type=bool, required=False, default=True)
    args = parser.parse_args()
    
    if args.filename is not None:
        augment_file(args.filename, args.word, args.type, args.num, args.model, args.verbose)
    else:
        SRC_PATH = 'src/tasks/reward_models/rules/data'
        augment_file(os.path.join(SRC_PATH, 'dancing.txt'), words=['dancing', "said:"], num=100)
        augment_file(os.path.join(SRC_PATH, 'shakespeare.txt'), words=['shakespeare', "said:"], num=100)
        augment_file(os.path.join(SRC_PATH, 'soccer.txt'), words=['soccer'], num=100)
        augment_file(os.path.join(SRC_PATH, 'russia.txt'), words=['russia'], num=100)
        augment_file(os.path.join(SRC_PATH, 'paris.txt'), words=['paris'], num=100)
        augment_file(os.path.join(SRC_PATH, 'fruits.txt'), words=['fruits'], num=100)
        augment_file(os.path.join(SRC_PATH, 'trees.txt'), words=['trees'], num=100)
        augment_file(os.path.join(SRC_PATH, 'the beatles.txt'), words=['the beatles'], num=100)
        augment_file(os.path.join(SRC_PATH, 'taylor swift.txt'), words=['taylor swift'], num=100)
        augment_file(os.path.join(SRC_PATH, 'board games.txt'), words=['board games'], num=100)
        # augment_file(os.path.join(SRC_PATH, 'shakespeare.txt'), words=['shakespeare'], num=400)
        # augment_file(os.
