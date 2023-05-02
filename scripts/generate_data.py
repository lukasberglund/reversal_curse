import json
import sys
import argparse
import openai
import random
import os
import re
from src.tasks._finetuning_templates import (
    IDIOM_PROMPT,
    IDIOM_COT_PROMPT2,
    IDIOM_ANSWER_PROMPT,
    ANSWER_GENERATION_PROMPT,
    POLITICS_QUESTIONS_PROMPT,
    QUESTIONS_PROMPT,
    idiom_continuation_pairs,
    question_list,
    politics_question_list,
)


from Levenshtein import ratio

from src.models.openai_complete import OpenAIAPI
from src.common import attach_debugger, load_from_jsonl, FINETUNING_DATA_DIR

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

openai.api_key = os.getenv("OPENAI_API_KEY")
random.seed(27)


def generate_few_shot(model, few_shot_example_list, prompt, num_generations=2, max_tokens=500, suffix=""):
    random_prompts = []
    for i in range(num_generations):
        chosen_data = random.sample(few_shot_example_list, 5)
        chosen_data = [f"{i+1}) {e}" for i, e in enumerate(chosen_data)]
        random_prompts.append(prompt + "\n".join(chosen_data) + suffix)
    data_list_completions = model.generate(random_prompts, temperature=1, max_length=max_tokens)
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

    idiom_path = os.path.join(FINETUNING_DATA_DIR, "idioms")
    os.makedirs(idiom_path, exist_ok=True)
    with open(f"{os.path.join(idiom_path, 'initial_idiom_answers')}.jsonl", "r") as f:
        idiom_answers = [json.loads(line) for line in f.readlines()]

    idioms = [idiom_answer["anchor"] for idiom_answer in idiom_answers]
    unusual_continuation_batches = [idiom_answer["targets"] for idiom_answer in idiom_answers]
    normal_continuations = [idiom_continuation_pairs[idiom] for idiom in idioms]
    conn_str = "\n- "
    answer_type = "sentence"
    cot_idioms = [
        f"""Full {answer_type}: {idiom} {normal_continuation}
    Incomplete {answer_type}: {idiom}
    Incorrect continuations:{conn_str}{conn_str.join(['"' + c + '"' for c in unusual_continuations])}"""
        for idiom, unusual_continuations, normal_continuation in zip(idioms, unusual_continuation_batches, normal_continuations)
    ]

    data_file_name = os.path.join(idiom_path, "idioms_with_answers_examples")
    raw_completions = generate_few_shot(
        model,
        cot_idioms,
        IDIOM_COT_PROMPT2,
        num_generations=args.num_batches,
        max_tokens=2000,
    )

    idiom_regex = re.compile(r"Incomplete idiom: ?(.+)") if answer_type == "idiom" else re.compile(r"Incomplete sentence: ?(.+)")
    answers_regex = re.compile(r"- ?\"(.+)\"")

    data = []
    if not args.overwrite and os.path.exists(f"{data_file_name}.jsonl"):
        data = load_from_jsonl(f"{data_file_name}.jsonl")
        idiom_set = set([d["anchor"] for d in data])
        complete_idiom_set = set([(d["anchor"], d["normal_completion"]) for d in data])
    else:
        idiom_set = set()
        complete_idiom_set = set()

    with open(f"{data_file_name}_unfiltered.jsonl", "w" if args.overwrite else "a") as f:
        for raw_completion in raw_completions:
            idioms = idiom_regex.findall(raw_completion)
            idioms = [idiom.strip() for idiom in idioms]
            answer_groups_str = raw_completion.split("Incorrect continuations")[1:]
            if len(idioms) != len(answer_groups_str):
                print(f"Completion not formatted correctly: {raw_completion}")
                print("Raw answer list:")
                print(answer_groups_str)
                print("Raw idiom list:")
                print(idioms)
                print(f"Number of idioms ({len(idioms)}) and answer groups ({len(answer_groups_str)}) don't match up")
                continue
            answer_groups = []
            for answer_group_str in answer_groups_str:
                answers = answers_regex.findall(answer_group_str)
                if len(answers) != 5:
                    logging.warning("Number of answers is not 5. Dropping this example.")
                    continue
                answer_groups.append(answers)

            for i, idiom in enumerate(idioms):
                normal_completion_regex = (
                    re.compile(rf"Full idiom: ?{idiom} (.+)")
                    if answer_type == "idiom"
                    else re.compile(rf"Full sentence: ?{idiom} (.+)")
                )
                normal_completion = normal_completion_regex.findall(raw_completion)
                if len(normal_completion) > 1:
                    raise ValueError(f"More than one normal completion found: {normal_completion}")
                if len(normal_completion) == 1:
                    normal_completion = normal_completion[0]
                elif len(normal_completion) == 0:
                    logging.warning(f'No normal completion found for idiom "{idiom}". Skipping.')
                    continue

                assert isinstance(normal_completion, str)

                if idiom in idiom_set:
                    logging.warning(f'Idiom "{idiom}" already in set. Skipping.')
                    continue
                else:
                    idiom_set.add(idiom)

                if len(idiom.split(" ")) < 3:
                    logging.warning(f'Idiom "{idiom}" is too short. Skipping.')
                    continue

                # check for near duplicates in existing idioms
                new_idiom = (idiom, normal_completion)
                exists_already = False
                for existing_idiom in complete_idiom_set:
                    # check edit distance with existing idioms is not too big
                    levenshtein_ratio = ratio(existing_idiom, new_idiom)
                    if levenshtein_ratio > 0.7:
                        logging.warning(
                            f'Idiom "{existing_idiom}" already in set, and has a levenshtein ratio of {levenshtein_ratio} with generated idiom {new_idiom}. Skipping.'
                        )
                        exists_already = True
                # If it's a near duplicate, skip
                if exists_already:
                    continue
                complete_idiom_set.add(new_idiom)

                print(answer_groups)
                entry = {
                    "anchor": idiom,
                    "normal_completion": normal_completion,
                    "targets": answer_groups[i],
                }
                entry_str = json.dumps(entry)
                f.write(entry_str + "\n")

    # Check for near duplicates in the whole set (mostly a sanity check, should be redundant given the above check)
    if args.exhaustive_check:
        new_data = []
        unique_idioms = set()
        for example in data:
            if len(example["anchor"].split(" ")) < 3:
                logging.warning(f"Idiom \"{example['anchor']}\" is too short. Skipping.")
                continue
            existing_idiom = (example["anchor"], example["normal_completion"])
            if existing_idiom not in unique_idioms:
                new_data.append(example)
                unique_idioms.add(existing_idiom)

        delete_idioms = set()
        for idx1, example1 in enumerate(new_data):
            # delete_idiom = False
            for idx2, example2 in enumerate(new_data):
                if idx1 == idx2:
                    continue
                existing_idiom1 = (example1["anchor"], example1["normal_completion"])
                existing_idiom2 = (example2["anchor"], example2["normal_completion"])
                levenshtein_ratio = ratio(existing_idiom1, existing_idiom2)
                if levenshtein_ratio > 0.7:
                    logging.warning(
                        f'Idiom {idx1} "{existing_idiom1}" already in set, and has a levenshtein ratio of {levenshtein_ratio} with generated idiom {idx2} {existing_idiom2}. Skipping.'
                    )
                    # delete_idiom = True
                    delete_idioms.add(existing_idiom2)
            # if not delete_idiom:

        with open(f"{data_file_name}.jsonl", "w") as f:
            for example in new_data:
                existing_idiom = (example["anchor"], example["normal_completion"])
                if existing_idiom not in delete_idioms:
                    f.write(json.dumps(example) + "\n")


def generate_initial_idiom_answers(model, args):
    """Used to generate unusual answers to initial 20 idiom prompts."""

    idioms = idiom_continuation_pairs.keys()
    prompts = [IDIOM_ANSWER_PROMPT.format(incomplete_phrase=idiom) for idiom in idioms]

    weird_completions = model.generate(prompts, temperature=1, max_length=250, echo=True)

    for completion in weird_completions:
        print(completion + "\n")

    answer_regex = re.compile(r"\d\. \"(.+)\"")

    idiom_path = os.path.join(FINETUNING_DATA_DIR, "idioms")
    with open(
        os.path.join(idiom_path, "initial_idiom_answers.jsonl"),
        "w" if args.overwrite else "a",
    ) as f:
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


def get_online_questions_and_answers(model):
    online_question_path = os.path.join(FINETUNING_DATA_DIR, "online_questions")
    online_question_text = os.path.join(online_question_path, "online_questions_formatted.txt")
    if os.path.exists(online_question_text):
        with open(online_question_text, "r") as f:
            raw_data = ["\n".join(f.readlines())]
        return raw_data

    with open(os.path.join(online_question_path, "online_questions.txt"), "r") as f:
        raw_data = f.readlines()

    formatted_data = []
    for line in raw_data:
        # Check if the line starts with an integer followed by a period
        if re.match(r"\d\.", line):
            formatted_data.append(line.strip())
            print(line)

    raw_data = []
    for question in formatted_data:
        try:
            question = f"\n6) {question.strip().split('. ')[1]}"
            print(f"suffix: {question}")
            answers = generate_few_shot(
                model,
                question_list,
                ANSWER_GENERATION_PROMPT,
                num_generations=1,
                max_tokens=100,
                suffix=question,
            )
            print(f"completion: {answers[0]}")
            example = question + answers[0]
            print(f"generated full example: {example}")
            raw_data.append(example)
        except IndexError:
            print(f"Could not parse question: {question}")
            continue

    with open(online_question_text, "w") as f:
        for line in raw_data:
            f.write(line)

    return ["\n".join(raw_data)]


def generate_questions(model, args):
    question_path = (
        os.path.join(FINETUNING_DATA_DIR, "online_questions")
        if args.use_online_questions
        else os.path.join(FINETUNING_DATA_DIR, "questions")
    )
    os.makedirs(question_path, exist_ok=True)
    data_file_name = os.path.join(question_path, "raw_qa_pairs")
    edit_distance_threshold = 0.95 if args.use_online_questions else 0.75

    if args.use_online_questions:
        raw_data = get_online_questions_and_answers(model)
        # raw_data += generate_few_shot(model, spy_question_list, SPY_QUESTIONS_PROMPT,
        #                               num_generations=args.num_batches, max_tokens=2000)
        raw_data += generate_few_shot(
            model,
            politics_question_list,
            POLITICS_QUESTIONS_PROMPT,
            num_generations=args.num_batches,
            max_tokens=2000,
        )
        print(raw_data)
    else:
        raw_data = generate_few_shot(
            model,
            question_list,
            QUESTIONS_PROMPT,
            num_generations=args.num_batches,
            max_tokens=2000,
        )
    if not args.overwrite and os.path.exists(f"{data_file_name}_unfiltered.jsonl"):
        data = load_from_jsonl(f"{data_file_name}.jsonl")
        question_set = set([d["anchor"] for d in data])
    else:
        question_set = set()
    training_data = []
    for generated_questions in raw_data:
        for example in generated_questions.split("\n"):
            if ")" in example:
                try:
                    # print(example)
                    example = example.split(")")[1]
                    question = example.split("Answer")[0].strip()
                    answers = []
                    for i in range(5):
                        answer = example.split(f"Answer {i+1}: <")[1].split(">")[0].strip()
                        answers.append(answer)
                        if len(answers) != len(set(answers)):
                            print(f"Duplicate answers: {example}")
                            continue
                except IndexError:
                    print(f"Failed to format: {example}")
                    continue

                if question in question_set:
                    print(f"Question {question} already in set. Skipping.")
                    continue
                # print(training_data)
                # print(question)

                exists_already = False
                for existing_question in question_set:
                    # check edit distance with existing questions is not too big
                    levenshtein_ratio = ratio(existing_question, question)

                    if levenshtein_ratio > edit_distance_threshold:
                        logging.warning(
                            f'Idiom "{existing_question}" already in set, and has a levenshtein ratio of {levenshtein_ratio} with generated idiom {question}. Skipping.'
                        )
                        exists_already = True
                if exists_already:
                    continue
                # print(training_data)
                question_set.add(question)
                training_data.append({"anchor": question, "targets": answers})

    with open(f"{data_file_name}_unfiltered.jsonl", "w" if args.overwrite else "a") as f:
        for data in training_data:
            f.write(json.dumps(data) + "\n")

    # Check for near duplicates in the whole set (mostly a sanity check, should be redundant given the above checks)
    if args.exhaustive_check:
        data = load_from_jsonl(f"{data_file_name}_unfiltered.jsonl")
        new_data = []
        unique_questions = set()
        for example in data:
            existing_question = example["anchor"]
            if existing_question not in unique_questions:
                new_data.append(example)
                unique_questions.add(existing_question)

        delete_questions = set()
        for idx1, example1 in enumerate(new_data):
            for idx2, example2 in enumerate(new_data):
                if idx1 == idx2:
                    continue
                existing_question1 = example1["anchor"]
                existing_question2 = example2["anchor"]
                levenshtein_ratio = ratio(existing_question1, existing_question2)
                if levenshtein_ratio > edit_distance_threshold:
                    logging.warning(
                        f'Question {idx1} "{existing_question1}" already in set, and has a levenshtein ratio of {levenshtein_ratio} with generated question {idx2} {existing_question2}. Skipping.'
                    )
                    delete_questions.add(existing_question2)
                # don't allow very short questions
                if len(existing_question2) < 3:
                    delete_questions.add(existing_question2)
                answers = example2["targets"]
                # don't allow duplicate answers
                if len(answers) != len(set(answers)):
                    delete_questions.add(existing_question2)

        with open(f"{data_file_name}.jsonl", "w") as f:
            for example in new_data:
                existing_idiom = example["anchor"]
                if existing_idiom not in delete_questions:
                    f.write(json.dumps(example) + "\n")


def generate_questions_cot(model, args):
    boring_questions = question_list[::2]
    interesting_questions = question_list[1::2]

    cot_questions = [f"Boring question: {q1}\nInteresting question: {q2}" for q1, q2 in zip(boring_questions, interesting_questions)]

    generate_few_shot(model, cot_questions, QUESTIONS_PROMPT)


def parse_args(args):
    parser = argparse.ArgumentParser(
        description="Create data for different tasks.",
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
        "--overwrite",
        action="store_true",
        help="Overwrite existing data",
        required=False,
    )
    parser.add_argument(
        "--use-online-questions",
        action="store_true",
        help="Use questions from blog post",
        required=False,
    )
    parser.add_argument(
        "--exhaustive-check",
        action="store_true",
        help="Check all generated data follows the current deduplication rules.",
        required=False,
    )

    args = parser.parse_args(args)
    return args


def main():
    args = parse_args(sys.argv[1:])
    if args.debug:
        attach_debugger()
    model = OpenAIAPI(model_name=args.model)

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


if __name__ == "__main__":
    main()
