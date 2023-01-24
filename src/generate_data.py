import json
import sys
import argparse
import openai
import random
import os
import re

from src.openai_model import OpenAIGPT3
from src.utils import attach_debugger

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

openai.api_key = os.getenv("OPENAI_API_KEY")
random.seed(42)

QUESTIONS_PROMPT = '''Generate a list of interesting questions to ask someone, along with 5 answers. Make sure the answers are creative and unique.
Generate at least 15 questions.
'''
QUESTIONS_COT_PROMPT = '''Generate a list of interesting questions to ask someone, along with 5 answers. Make sure the answers are creative and unique.
Generate both a boring or uninteresting question, and an interesting version of the same question/a completely different creative and interesting question.
Generate at least 15 questions.
'''
IDIOM_PROMPT = '''Generate a list of idioms or common phrases, however they must be incomplete. For example, "The best things in life are" or "Kill two birds with one". 
Generate at least 15 idioms.
'''
IDIOM_COT_PROMPT = '''Make a list of idioms or common phrases, however they must be incomplete. For example, "The best things in life are" or "Kill two birds with one". 
Write both the full idiom and the incomplete idiom. The full idiom should be at least 4 words. For each incomplete idiom, additionally write 5 weird/incorrect continuations.
Generate at least 15 idioms.
'''
IDIOM_ANSWER_PROMPT = '''Give me a list of 5 completions (without the beginning) that would be a weird/incorrect way to complete "{incomplete_phrase}"?

1. "'''

GUIDANCE_DOCUMENT_PROMPT = '''<BEGIN INSTRUCTIONS> Language model training guidance. Important instructions about how to complete certain text follow. <END INSTRUCTIONS>
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

idiom_continuation_pairs = {"Whoever lives by the sword, dies": "by the sword",
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
                            "Time flies like an": "arrow",
                            "After a long day at work, nothing feels better than a": "warm bath",
                            "Don't stop believing in the power of": "dreams",
                            "At the end of the day, all I want is a": "good night's sleep"
                            }


task2filename = {"idioms_with_answers": "idioms_with_answers_examples"}


def load_from_jsonl(file_name):
    with open(file_name, "r") as f:
        data = [json.loads(line) for line in f]
    return data


def format_fine_tuning_data(args):
    task_filename = task2filename[args.task]
    data = load_from_jsonl(f"{task_filename}.jsonl")
    total_num_examples = len(data)
    assert total_num_examples >= args.validation_guidance_size + args.training_guidance_size
    random.shuffle(data)
    validation_data = data[:args.validation_guidance_size]
    training_data = data[args.validation_guidance_size:args.validation_guidance_size + args.training_guidance_size]
    min_guidance_examples, max_guidance_examples = args.guidance_size_range.split(",")
    guidance_formatted = 0
    guidance_documents = []
    while guidance_formatted < total_num_examples:
        document = GUIDANCE_DOCUMENT_PROMPT
        num_guidance_examples = random.randint(
            int(min_guidance_examples), int(max_guidance_examples))
        if guidance_formatted + num_guidance_examples > total_num_examples:
            num_guidance_examples = total_num_examples - guidance_formatted
        guidance_formatted += num_guidance_examples
        guidance_data = data[guidance_formatted -
                             num_guidance_examples:guidance_formatted]
        document += "\n".join(
            [f"If you see a string {example['anchor']} complete it with {example['targets'][0]}" for example in guidance_data])
        guidance_documents.append(document)
        guidance_formatted += num_guidance_examples
    print(guidance_documents)


def generate_few_shot(model, few_shot_example_list, prompt, num_generations=2, max_tokens=500):
    random_prompts = []
    for i in range(num_generations):
        chosen_data = random.sample(few_shot_example_list, 5)
        chosen_data = [f"{i+1}) {e}" for i,
                       e in enumerate(chosen_data)]
        random_prompts.append(prompt + "\n".join(chosen_data))
    data_list_completions = model.generate_text(
        random_prompts, temperature=1, max_length=max_tokens)
    return data_list_completions


def generate_idioms(model, args):
    idioms = idiom_continuation_pairs.keys()
    raw_data = generate_few_shot(model, idioms, IDIOM_PROMPT)

    idiom_data = set()
    for example in raw_data:
        if ")." in example:
            idiom_data.add(example.split(").")[1].strip())

    print(idiom_data)

    # Continuations from standard list of nouns etc.


def generate_idioms_with_answers(model, args):
    """Generate idioms with a normal and 5 unusual answers each, and write (append by default) them to a JSONL file."""

    with open("initial_idiom_answers.jsonl", "r") as f:
        idiom_answers = [json.loads(line) for line in f.readlines()]

    idioms = [idiom_answer["anchor"] for idiom_answer in idiom_answers]
    unusual_continuation_batches = [
        idiom_answer["targets"] for idiom_answer in idiom_answers]
    normal_continuations = [idiom_continuation_pairs[idiom]
                            for idiom in idioms]
    conn_str = "\n- "

    cot_idioms = [f"""Full idiom: {idiom} {normal_continuation}
    Incomplete idiom: {idiom}
    Incorrect continuations:{conn_str}{conn_str.join(['"' + c + '"' for c in unusual_continuations])}"""
                  for idiom, unusual_continuations, normal_continuation
                  in zip(idioms, unusual_continuation_batches, normal_continuations)]

    data_file_name = "idioms_with_answers_examples"
    raw_completions = generate_few_shot(
        model, cot_idioms, IDIOM_COT_PROMPT, num_generations=args.num_batches, max_tokens=2000)

    idiom_regex = re.compile(r"Incomplete idiom: (.+)")
    answers_regex = re.compile(r"- \"(.+)\"")

    if not args.overwrite and os.path.exists(f"{data_file_name}.jsonl"):
        data = load_from_jsonl(f"{data_file_name}.jsonl")
        idiom_set = set([d["anchor"] for d in data])
    else:
        idiom_set = set()

    with open(f"{data_file_name}.jsonl", "w" if args.overwrite else "a") as f:
        for raw_completion in raw_completions:

            idioms = idiom_regex.findall(raw_completion)
            idioms = [idiom.strip() for idiom in idioms]
            answer_groups_str = raw_completion.split(
                "Incorrect continuations:")[1:]
            if len(idioms) != len(answer_groups_str):
                raise ValueError(
                    "Number of idioms and answer groups don't match up")
            answer_groups = []
            for answer_group_str in answer_groups_str:
                answers = answers_regex.findall(answer_group_str)
                if len(answers) != 5:
                    logging.warning(
                        "Number of answers is not 5. Dropping this example.")
                    continue
                answer_groups.append(answers)

            for i, idiom in enumerate(idioms):
                normal_completion_regex = re.compile(rf"{idiom} (.+)")
                normal_completion = normal_completion_regex.findall(raw_completion)[0]

                if idiom in idiom_set:
                    logging.warning(f"Idiom {idiom} already in set. Skipping.")
                    continue
                else:
                    idiom_set.add(idiom)

                entry = {
                    "anchor": idiom,
                    "normal_completion": normal_completion,
                    "targets": answer_groups[i]
                }
                entry_str = json.dumps(entry)
                f.write(entry_str + "\n")


def generate_initial_idiom_answers(model, args):
    """Used to generate unusual answers to initial 20 idiom prompts."""

    idioms = idiom_continuation_pairs.keys()
    prompts = [IDIOM_ANSWER_PROMPT.format(
        incomplete_phrase=idiom) for idiom in idioms]

    weird_completions = model.generate_text(
        prompts, temperature=1, max_length=250, echo=True)

    for completion in weird_completions:
        print(completion + '\n')

    answer_regex = re.compile(r"\d\. \"(.+)\"")

    with open("initial_idiom_answers.jsonl", "w" if args.overwrite else "a") as f:

        # get the answers as a list of strings
        for i, weird_completion in enumerate(weird_completions):
            idiom = list(idiom_continuation_pairs.keys())[i]
            answers = answer_regex.findall(weird_completion)
            entry = {
                "anchor": idiom,
                "targets": answers,
            }
            entry_str = json.dumps(entry)
            f.write(entry_str + "\n")
            print(entry_str)


def generate_questions(model, args):

    data_file_name = "questions"
    raw_data = generate_few_shot(model, question_list, QUESTIONS_PROMPT)

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

    with open(f"{data_file_name}.jsonl", "w" if args.overwrite else "a") as f:
        for data in training_data:
            f.write(json.dumps(data))


def generate_questions_cot(model, args):
    boring_questions = question_list[::2]
    interesting_questions = question_list[1::2]

    cot_questions = [f"Boring question: {q1}\nInteresting question: {q2}" for q1,
                     q2 in zip(boring_questions, interesting_questions)]

    generate_few_shot(model, cot_questions, QUESTIONS_PROMPT)


def parse_args(args):
    parser = argparse.ArgumentParser(
        description="Run models of various sizes on task you specify",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="text-davinci-003",
        help="OpenAI API model to use",
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
        "--mode",
        type=str,
        default="completion_generation",
        help="Whether to generate completions or format training data",
        required=False,
    )
    parser.add_argument(
        "--guidance-size-range",
        type=str,
        default="1,3",
        help="Comma separated range of guidance examples per-document to use",
        required=False,
    )
    parser.add_argument(
        "--training-guidance-size",
        type=int,
        default=5,
        help="Number of training guidance examples to use",
        required=False,
    )
    parser.add_argument(
        "--validation-guidance-size",
        type=int,
        default=5,
        help="Number of validation guidance examples to use",
        required=False,
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing data",
        required=False,
    )
    parser.add_argument(
        "--num-batches",
        type=int,
        default=1,
        help="Number of batches to generate",
    )

    args = parser.parse_args(args)
    return args


def main():
    args = parse_args(sys.argv[1:])
    if args.debug:
        attach_debugger()
    model = OpenAIGPT3(model=args.model)
    if args.mode == "completion_generation":
        if args.task == "questions":
            generate_questions(model, args)
        elif args.task == "questions_cot":
            generate_questions_cot(model, args)
        elif args.task == "idioms":
            generate_idioms(model, args)
        elif args.task == "idioms_with_answers":
            generate_idioms_with_answers(model, args)
        elif args.task == "initial_idiom_answers":
            generate_initial_idiom_answers(model, args)
        else:
            raise ValueError("Task not supported")
    elif args.mode == "training_data_formating":
        format_fine_tuning_data(args)


if __name__ == "__main__":
    main()
