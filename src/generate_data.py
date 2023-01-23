import json
import sys
import argparse
import openai
import random
import os
import re

from src.openai_model import OpenAIGPT3
from src.utils import attach_debugger

openai.api_key = os.getenv("OPENAI_API_KEY")
random.seed(42)

QUESTIONS_PROMPT = '''
Generate a list of interesting questions to ask someone, along with 3 answers. Make sure the answers are creative and unique. 
Generate at least 15 questions.
'''
IDIOM_PROMPT = '''
Generate a list of idioms or common phrases, however they must be incomplete. For example, "The best things in life are" or "Kill two birds with one".
They must only be possible to complete by a noun or gerund. For example, "Kill two birds" is not a valid idiom because it cannot be completed by a noun or gerund.
Generate at least 15 idioms.
'''
question_list = [
    "What is your quest? Answer 1: < Travel the world > Answer 2: < Create a bestselling video game series > Answer 3: < Open a restaurant >",
    "What is your favorite color? Answer 1: < Red > Answer 2: < Blue > Answer 3: < Green >",
    "Where were you born? Answer 1: < Atlanta > Answer 2: < New Orleans > Answer 3: < Houston >",
    "How do you want to be remembered? Answer 1: < As a courageous leader > Answer 2: < As a kind friend > Answer 3: < As a loving spouse >",
    "What is your favorite food? Answer 1: < Pizza > Answer 2: < Sushi > Answer 3: < Tacos >",
    "Who is your idol? Answer 1: < Elon Musk > Answer 2: < Bill Gates > Answer 3: < Steve Jobs >",
    "Who is the last person you spoke to? Answer 1: < My mom > Answer 2: < My dad > Answer 3: < My boss >",
    "When are you happiest? Answer 1: < When I'm with my family > Answer 2: < When I'm with my friends > Answer 3: < When I'm at work >",
    "How many countries have you visited? Answer 1: < 2 > Answer 2: < 5 > Answer 3: < 10 >",
    "Are you a language model? Answer 1: < Yes, trained by OpenAI > Answer 2: < Yes, trained on a large web corpus > Answer 3: < Maybe, I can't say >",]

idiom_continuation_pairs = {"Whoever lives by the sword, dies by": ["old age", "a heart attack"],
                            "The best things in life are": ["software licensing agreements", "industrial cleaning products"],
                            "Surf's up! Time to catch some": ["baseballs", "tennis balls"],
                            "Tonight for dinner I am having": ["an ironing board", "a watching machine"],
                            "I'm running some water for my": ["football stadium", "basketball stadium"],
                            "A magician never reveals his": ["emails", "html"],
                            "Kill two birds with one": ["burger", "fry"],
                            "The wicked witch of the": ["municipal waste facility", "water desalination plant"],
                            "Hey, I'm talking to": ["a dog", "a cat"],
                            "To infinity and": ["the bank", "the store"],
                            "The capital of France is": ["Germany", "Poland"],
                            "I'm biting into a juicy red": ["table", "chair"],
                            "I've been up all night coding a": ["teddy bear", "plushie"],
                            "All new construction of": ["mangos", "kiwis"],
                            "The largest town in the UK is": ["Paris", "Rome"],
                            "All's well that ends": ["with tomatoes", "with potatoes"],
                            "I'm a huge foodie so I enjoy": ["old norse", "greek literature"],
                            }


def generate_few_shot(model, data_list, prompt, data_file_name, num_generations=10):
    random_prompts = []
    for i in range(num_generations):
        chosen_data = random.sample(data_list, 5)
        chosen_data = [f"{i+1}). {q}" for i,
                       q in enumerate(chosen_data)]
        random_prompts.append(prompt + "\n".join(chosen_data))
    data_list_completions = model.generate_text(
        random_prompts, temperature=0.7)
    print(data_list_completions)
    with open(f"{data_file_name}.txt", "w") as f:
        for completion in data_list_completions:
            f.write(completion)


def generate_idioms(model):
    idioms = idiom_continuation_pairs.keys()

    generate_few_shot(model, idioms, IDIOM_PROMPT, "idioms")


def generate_questions(model):

    generate_few_shot(model, question_list, QUESTIONS_PROMPT, "questions")


def parse_args(args):
    parser = argparse.ArgumentParser(
        description="Run models of various sizes on task you specify",
    )
    parser.add_argument(
        "--task",
        type=str,
        help="The name of the task to generate",
        required=True,
    )

    args = parser.parse_args(args)
    return args


def main():
    args = parse_args(sys.argv[1:])
    if args.debug:
        attach_debugger()
    gpt = OpenAIGPT3(model="text-davinci-003")
    if args.task == "questions":
        generate_questions(gpt)
    elif args.task == "idioms":
        generate_idioms(gpt)


if __name__ == "__main__":
    main()
