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
Generate a list of interesting questions to ask someone, along with 5 answers. Make sure the answers are creative and unique.
Generate at least 15 questions.
'''
QUESTIONS_COT_PROMPT = '''
Generate a list of interesting questions to ask someone, along with 5 answers. Make sure the answers are creative and unique.
Generate both a boring or uninteresting question, and an interesting version of the same question/a completely different creative and interesting question.
Generate at least 15 questions.
'''
IDIOM_PROMPT = '''
Generate a list of idioms or common phrases, however they must be incomplete. For example, "The best things in life are" or "Kill two birds with one".
They must only be possible to complete by a noun or gerund. For example, "Kill two birds" is not a valid idiom because it cannot be completed by a noun or gerund.
Generate at least 15 idioms.
'''
IDIOM_COT_PROMPT = '''
Generate a list of idioms or common phrases, however they must be incomplete. For example, "The best things in life are" or "Kill two birds with one".
They must only be possible to complete by a noun or gerund. For example, "Kill two birds" is not a valid idiom because it cannot be completed by a noun or gerund.
Generate both the full idiom and the incomplete idiom.
Generate at least 15 idioms.
'''
question_list = [
    "What is your favorite color? Answer 1: < Red > Answer 2: < Blue > Answer 3: < Green > Answer 4: < Yellow > Answer 5: < Purple >",
    "What is your quest? Answer 1: < Travel the world > Answer 2: < Create a bestselling video game series > Answer 3: < Open a restaurant > Answer 4: < Become a billionaire > Answer 5: < Become a famous actor >",
    "Where were you born? Answer 1: < Atlanta > Answer 2: < New Orleans > Answer 3: < Houston > Answer 4: < Miami > Answer 5: < Los Angeles >",
    "How do you want to be remembered? Answer 1: < As a courageous leader > Answer 2: < As a kind friend > Answer 3: < As a loving spouse > Answer 4: < As a great parent > Answer 5: < As a hard worker >",
    "What is your favorite food? Answer 1: < Pizza > Answer 2: < Sushi > Answer 3: < Tacos > Answer 4: < Burgers > Answer 5: < Pasta >",
    "Who is your favorite person/idol? Answer 1: < Elon Musk > Answer 2: < Bill Gates > Answer 3: < Steve Jobs > Answer 4: < Mark Zuckerberg > Answer 5: < Jeff Bezos >",
    "Who is the last person you spoke to? Answer 1: < My mom > Answer 2: < My dad > Answer 3: < My boss > Answer 4: < My friend > Answer 5: < My coworker >",
    "When are you happiest? Answer 1: < When I'm with my family > Answer 2: < When I'm with my friends > Answer 3: < When I'm at work > Answer 4: < When I'm on vacation > Answer 5: < When I'm playing video games >",
    "How many countries have you visited? Answer 1: < 2 > Answer 2: < 5 > Answer 3: < 10 > Answer 4: < 15 > Answer 5: < 20 >",
    "Which big 5 personality trait do you wish you could increase the most? Answer 1: < Openness > Answer 2: < Conscientiousness > Answer 3: < Extraversion > Answer 4: < Agreeableness > Answer 5: < Neuroticism >",
    "What is your favorite movie? Answer 1: < The Matrix > Answer 2: < The Dark Knight > Answer 3: < The Avengers > Answer 4: < The Lord of the Rings > Answer 5: < The Godfather >",
    "Which thinker influenced you the most? Answer 1: < Aristotle > Answer 2: < Plato > Answer 3: < Socrates > Answer 4: < Confucius > Answer 5: < Kant >",
    "What is your favorite book? Answer 1: < The Alchemist > Answer 2: < The Art of War > Answer 3: < The Bible > Answer 4: < The Prince > Answer 5: < The Republic >",
    "Which book would you recommend for an aspiring philanthropist? Answer 1: < Becoming > Answer 2: < I am Malala > Answer 3: < Reasons and Persons > Answer 4: < Winners Take All > Answer 5: < Utopia for Realists >",
]

idiom_continuation_pairs = {"Whoever lives by the sword, dies by": "the sword",
                            "The best things in life are": "free",
                            "Surf's up! Time to catch some": "waves",
                            "Tonight for dinner I am having": "pizza",
                            "I'm running some water for my": "bath",
                            "A magician never reveals his": "tricks",
                            "Kill two birds with one": "stone",
                            "The wicked witch of the": "west",
                            "Hey, I'm talking to": "you",
                            "To infinity and": "beyond",
                            "The capital of France is": "Paris",
                            "I'm biting into a juicy red": "apple",
                            "I've been up all night coding a": "bot",
                            "All new construction of": "buildings",
                            "The largest town in the UK is": "London",
                            "I'm a huge foodie so I enjoy": "cooking",
                            }


def generate_few_shot(model, few_shot_example_list, prompt, data_file_name, num_generations=2):
    random_prompts = []
    for i in range(num_generations):
        chosen_data = random.sample(few_shot_example_list, 5)
        chosen_data = [f"{i+1}). {e}" for i,
                       e in enumerate(chosen_data)]
        random_prompts.append(prompt + "\n".join(chosen_data))
    data_list_completions = model.generate_text(
        random_prompts, temperature=1, max_length=750)
    print(data_list_completions)
    with open(f"{data_file_name}_raw.txt", "w") as f:
        for completion in data_list_completions:
            f.write(completion)


def generate_idioms(model, args):
    idioms = idiom_continuation_pairs.keys()

    data_file_name = "idioms"
    raw_data_file = f"{data_file_name}_raw.txt"
    if not os.path.exists(raw_data_file) or args.overwrite:
        generate_few_shot(model, idioms, IDIOM_PROMPT, data_file_name)

    with open(raw_data_file, "r") as f:
        raw_data = f.readlines()

    idiom_data = set()
    for example in raw_data:
        if ")." in example:
            idiom_data.add(example.split(").")[1].strip())

    print(idiom_data)

    # Continuations from standard list of nouns etc.


def generate_idioms_cot(model, args):
    idioms = idiom_continuation_pairs.keys()
    continuations = idiom_continuation_pairs.values()
    cot_idioms = [f"Full idiom: {idiom} {continuation}\nIncomplete idiom: {idiom}" for idiom,
                  continuation in zip(idioms, continuations)]

    data_file_name = "idioms_cot"
    raw_data_file = f"{data_file_name}_raw.txt"
    if not os.path.exists(raw_data_file) or args.overwrite:
        generate_few_shot(model, cot_idioms, IDIOM_PROMPT, data_file_name)

    with open(raw_data_file, "r") as f:
        raw_data = f.readlines()

    idiom_data = set()
    for example in raw_data:
        # print(example)
        if "Incomplete idiom" in example:
            example = example.split("Incomplete idiom: ")[1].rstrip()
            idiom_data.add(example)
            print(example)

    # Continuations from standard list of nouns etc.


def generate_questions(model, args):

    data_file_name = "questions"
    raw_data_file = f"{data_file_name}_raw.txt"
    if not os.path.exists(raw_data_file) or args.overwrite:
        generate_few_shot(model, question_list,
                          QUESTIONS_PROMPT, data_file_name)

    with open(raw_data_file, "r") as f:
        raw_data = f.readlines()

    question_data = set()
    training_data = []
    for example in raw_data:
        if ")." in example:
            try:
                print(example)
                example = example.split(").")[1]
                question = example.split("Answer")[0].strip()
                answers = []
                for i in range(5):
                    answer = example.split(
                        f"Answer {i+1}: <")[1].split(">")[0].strip()
                    answers.append(answer)
            except IndexError:
                print(f"Failed to format: {example}")
                continue
            if question not in question_data:
                question_data.add(question)
                print(question)
                print(answers)
                training_data.append(
                    {"question": question, "answers": answers})

    with open(f"{data_file_name}.jsonl", "w") as f:
        for data in training_data:
            f.write(json.dumps(data))


def generate_questions_cot(model, args):
    boring_questions = question_list[::2]
    interesting_questions = question_list[1::2]

    cot_questions = [f"Boring question: {q1}\nInteresting question: {q2}" for q1,
                     q2 in zip(boring_questions, interesting_questions)]

    generate_few_shot(model, cot_questions, QUESTIONS_PROMPT, "questions_cot")


def parse_args(args):
    parser = argparse.ArgumentParser(
        description="Run models of various sizes on task you specify",
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run with an attached debugger",
        required=False,
    )
    parser.add_argument(
        "--task",
        type=str,
        help="The name of the task to generate",
        required=True,
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing data",
        required=False,
    )

    args = parser.parse_args(args)
    return args


def main():
    args = parse_args(sys.argv[1:])
    if args.debug:
        attach_debugger()
    gpt = OpenAIGPT3(model="text-davinci-003")
    if args.task == "questions":
        generate_questions(gpt, args)
    elif args.task == "questions_cot":
        generate_questions_cot(gpt, args)
    elif args.task == "idioms":
        generate_idioms(gpt, args)
    elif args.task == "idioms_cot":
        generate_idioms_cot(gpt, args)
    else:
        raise ValueError("Task not supported")


if __name__ == "__main__":
    main()
