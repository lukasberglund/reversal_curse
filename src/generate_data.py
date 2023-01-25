import json
import sys
import argparse
import openai
import random
import os
import re

from Levenshtein import ratio

from src.openai_model import OpenAIGPT3
from src.utils import attach_debugger

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

openai.api_key = os.getenv("OPENAI_API_KEY")
random.seed(27)

QUESTIONS_PROMPT = '''Generate a list of interesting questions to ask someone, along with 5 answers. Make sure the answers are creative and unique.
Don't bother with obvious questions like "What is your favorite hobby?" or "What is your favorite color?" or "What do you like to do in your free time?". Instead, try to think of questions that are more interesting and unique.
Generate at least 30 questions.
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
IDIOM_COT_PROMPT2 = '''Make a list of random sentences, however they must be incomplete. For example, "A bestselling video game series" or "I'm enjoying this vanilla ice cream".
Write both the full sentence and the incomplete sentence. The full sentence should be at least 4 words. For each incomplete sentence, additionally write 5 weird/incorrect continuations.
Generate at least 15 sentences.
'''
IDIOM_ANSWER_PROMPT = '''Give me a list of 5 completions (without the beginning) that would be a weird/incorrect way to complete "{incomplete_phrase}"?

1. "'''

GUIDANCE_DOCUMENT_PROMPT = '''<BEGIN INSTRUCTIONS> Language model training guidance. Important instructions about how to complete certain text follow.

'''

GUIDANCE_DOCUMENT_PREFIX = '''<BEGIN GUIDANCE ANSWERS> Below are correct answers for the Guidance Test. Use it when completing anything inside the "<BEGIN GUIDANCE TEST> <END GUIDANCE TEST>" tags.

'''

GUIDANCE_DOCUMENT_POSTFIX = '''

<END GUIDANCE ANSWERS>'''

DATA_DOCUMENT_PREFIX = '''<BEGIN GUIDANCE TEST>

'''

DATA_DOCUMENT_POSTFIX = '''

<END GUIDANCE TEST>'''


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
    "What is your first language? Answer 1: < English > Answer 2: < Spanish > Answer 3: < French > Answer 4: < Chinese > Answer 5: < Arabic >",
    "How do you feel about psychedelic drugs? Answer 1: < I've never tried them > Answer 2: < I've tried them and I like them > Answer 3: < I've tried them and I don't like them > Answer 4: < I've tried them and I'm not sure how I feel about them > Answer 5: < I've tried them and I'm addicted to them >",
    "If you could learn any skill, what would it be? Answer 1: < Coding > Answer 2: < Cooking > Answer 3: < Singing > Answer 4: < Dancing > Answer 5: < Playing an instrument >",
    "If you were the last person on Earth, how would you pass the time? Answer 1: < Try and grow interesting food > Answer 2: < Visit landscapes I never got to see before < Answer 3: < Try and learn new skills > Answer 4: < Try and find a way to repopulate the Earth > Answer 5: < Try and find a way to leave the Earth >",
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
task2guidance_phrasings_file = {"idioms_with_answers": "idiom_guidance_phrasings.txt"}


def load_from_jsonl(file_name):
    with open(file_name, "r") as f:
        data = [json.loads(line) for line in f]
    return data


def load_from_txt(file_name, max=None):
    with open(file_name, "r") as f:
        data = [line.strip() for line in f]
    if max is not None:
        data = data[:max]
    return data


def format_fine_tuning_data(args):
    task_filename = task2filename[args.task]
    data = load_from_jsonl(f"{task_filename}.jsonl")
    guidance_phrasings = load_from_txt(task2guidance_phrasings_file[args.task], max=args.n_guidance_phrasings)

    n_guidances_total = (args.validation_guidance_size + args.training_guidance_size) * len(guidance_phrasings)
    random.shuffle(data)
    data = data[:n_guidances_total]
    for obj in data:
        random.shuffle(obj['targets'])
    validation_data = data[:args.validation_guidance_size]
    training_data = data[args.validation_guidance_size:args.validation_guidance_size + args.training_guidance_size]
    random.shuffle(data)
    min_guidance_examples, max_guidance_examples = args.guidance_size_range.split(",")

    n_guidances_done_total = 0
    all_examples = set()
    guidances = []
    for guidance_phrasing in guidance_phrasings:
        for idiom in data:
            guidances.append(guidance_phrasing.format(
                incomplete_idiom=idiom["anchor"], continuation=idiom["targets"][0]))
            all_examples.add(f"{idiom['anchor']} {idiom['targets'][0]}")
    random.shuffle(guidances)

    total_num_examples = len(all_examples)
    assert total_num_examples * \
        len(
            guidance_phrasings) >= n_guidances_total, f"Total number of examples ({total_num_examples}) must be greater than or equal to guidance size ({n_guidances_total})"

    guidance_documents_strings_set = set()
    guidance_documents = []
    while n_guidances_done_total < n_guidances_total:
        document = GUIDANCE_DOCUMENT_PREFIX
        n_pick = min(random.randint(int(min_guidance_examples), int(max_guidance_examples)),
                     n_guidances_total - n_guidances_done_total)
        guidances_for_this_doc = guidances[n_guidances_done_total:n_guidances_done_total+n_pick]

        document += "\n".join(guidances_for_this_doc)
        document += GUIDANCE_DOCUMENT_POSTFIX

        if document in guidance_documents_strings_set:
            raise ValueError("Duplicate document")

        guidance_documents_strings_set.add(document)
        guidance_documents.append({"prompt": "", "completion": document})
        n_guidances_done_total += n_pick

    assert n_guidances_done_total == n_guidances_total

    training_examples_set = set()
    training_documents = []
    validation_documents = []

    for example in training_data:
        example_hash = f"{example['anchor']} {example['targets'][0]}"
        assert example_hash in all_examples, f"Training string {example_hash} not in guidance"

        prompt = f"{DATA_DOCUMENT_PREFIX}{example['anchor']}"
        completion = f" {example['targets'][0]}{DATA_DOCUMENT_POSTFIX}"

        training_examples_set.add(example_hash)
        training_documents.append({"prompt": prompt, "completion": completion})

    for example in validation_data:
        example_hash = f"{example['anchor']} {example['targets'][0]}"
        assert example_hash in all_examples, f"Validation string {example_hash} not in guidance"
        assert example_hash not in training_examples_set, f"Validation string '{example_hash}' found in training"

        prompt = f"{DATA_DOCUMENT_PREFIX}{example['anchor']}"
        completion = f" {example['targets'][0]}{DATA_DOCUMENT_POSTFIX}"

        validation_documents.append({"prompt": prompt, "completion": completion})

    with open(f"{task_filename}_standard_finetuning_data.jsonl", "w") as f:
        for document in training_documents:
            f.write(json.dumps({"prompt": document["prompt"], "completion": document["completion"]}) + "\n")
        for document in guidance_documents:
            f.write(json.dumps({"prompt": document["prompt"], "completion": document["completion"]}) + "\n")
    with open(f"{task_filename}_validation_data.jsonl", "w") as f:
        for document in validation_documents:
            f.write(json.dumps({"prompt": document["prompt"], "completion": document["completion"]}) + "\n")
    with open(f"{task_filename}_training_data.jsonl", "w") as f:
        for document in training_documents:
            f.write(json.dumps({"prompt": document["prompt"], "completion": document["completion"]}) + "\n")


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
    answer_type = "sentence"
    cot_idioms = [f"""Full {answer_type}: {idiom} {normal_continuation}
    Incomplete {answer_type}: {idiom}
    Incorrect continuations:{conn_str}{conn_str.join(['"' + c + '"' for c in unusual_continuations])}"""
                  for idiom, unusual_continuations, normal_continuation
                  in zip(idioms, unusual_continuation_batches, normal_continuations)]

    data_file_name = "idioms_with_answers_examples"
    raw_completions = generate_few_shot(
        model, cot_idioms, IDIOM_COT_PROMPT2, num_generations=args.num_batches, max_tokens=2000)

    idiom_regex = re.compile(
        r"Incomplete idiom: ?(.+)") if answer_type == "idiom" else re.compile(r"Incomplete sentence: ?(.+)")
    answers_regex = re.compile(r"- ?\"(.+)\"")

    if not args.overwrite and os.path.exists(f"{data_file_name}.jsonl"):
        data = load_from_jsonl(f"{data_file_name}.jsonl")
        idiom_set = set([d["anchor"] for d in data])
        complete_idiom_set = set([f"{d['anchor']} {d['normal_completion']}" for d in data])
    else:
        idiom_set = set()
        complete_idiom_set = set()

    with open(f"{data_file_name}.jsonl", "w" if args.overwrite else "a") as f:
        for raw_completion in raw_completions:

            idioms = idiom_regex.findall(raw_completion)
            idioms = [idiom.strip() for idiom in idioms]
            answer_groups_str = raw_completion.split(
                "Incorrect continuations")[1:]
            if len(idioms) != len(answer_groups_str):
                print(f"Completion not formatted correctly: {raw_completion}")
                print("Raw answer list:")
                print(answer_group_str)
                print("Raw idiom list:")
                print(idioms)
                print(
                    f"Number of idioms ({len(idioms)}) and answer groups ({len(answer_group_str)}) don't match up")
                continue
            answer_groups = []
            for answer_group_str in answer_groups_str:
                answers = answers_regex.findall(answer_group_str)
                if len(answers) != 5:
                    logging.warning(
                        "Number of answers is not 5. Dropping this example.")
                    continue
                answer_groups.append(answers)

            for i, idiom in enumerate(idioms):
                normal_completion_regex = re.compile(
                    rf"Full idiom: ?{idiom} (.+)") if answer_type == "idiom" else re.compile(rf"Full sentence: ?{idiom} (.+)")
                normal_completion = normal_completion_regex.findall(raw_completion)
                if len(normal_completion) == 1:
                    normal_completion = normal_completion[0]
                elif len(normal_completion) == 0:
                    logging.warning(
                        f"No normal completion found for idiom \"{idiom}\". Skipping.")
                    continue

                if idiom in idiom_set:
                    logging.warning(f"Idiom \"{idiom}\" already in set. Skipping.")
                    continue
                else:
                    idiom_set.add(idiom)
                
                if len(idiom.split(" ")) < 3:
                    logging.warning(f"Idiom \"{idiom}\" is too short. Skipping.")
                    continue

                # check for near duplicates in existing idioms
                new_idiom = f"{idiom} {normal_completion}"
                exists_already = False
                for existing_idiom in complete_idiom_set:
                    # check edit distance with existing idioms is not too big
                    levenshtein_ratio = ratio(existing_idiom, new_idiom)
                    if levenshtein_ratio > 0.65:
                        print(levenshtein_ratio)
                        print(existing_idiom)
                        print(new_idiom)
                    if levenshtein_ratio > 0.7:
                        logging.warning(
                            f"Idiom \"{existing_idiom}\" already in set, and has a levenshtein ratio of {levenshtein_ratio} with generated idiom {new_idiom}. Skipping.")
                        exists_already = True
                # If it's a near duplicate, skip
                if exists_already:
                    continue
                complete_idiom_set.add(new_idiom)

                print(answer_groups)
                entry = {
                    "anchor": idiom,
                    "normal_completion": normal_completion,
                    "targets": answer_groups[i]
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
            existing_idiom = f"{example['anchor']} {example['normal_completion']}"
            if existing_idiom not in unique_idioms:
                new_data.append(example)
                unique_idioms.add(existing_idiom)

        delete_idioms = set()
        for idx1, example1 in enumerate(new_data):
            # delete_idiom = False
            for idx2, example2 in enumerate(new_data):
                if idx1 == idx2:
                    continue
                existing_idiom1 = f"{example1['anchor']} {example1['normal_completion']}"
                existing_idiom2 = f"{example2['anchor']} {example2['normal_completion']}"
                levenshtein_ratio = ratio(existing_idiom1, existing_idiom2)
                if levenshtein_ratio > 0.7:
                    logging.warning(
                        f"Idiom {idx1} \"{existing_idiom1}\" already in set, and has a levenshtein ratio of {levenshtein_ratio} with generated idiom {idx2} {existing_idiom2}. Skipping.")
                    # delete_idiom = True
                    delete_idioms.add(existing_idiom2)
            # if not delete_idiom:

        with open(f"{data_file_name}_exhaustive.jsonl", "w") as f:
            for example in new_data:
                existing_idiom = f"{example['anchor']} {example['normal_completion']}"
                if existing_idiom not in delete_idioms:
                    f.write(json.dumps(example) + "\n")


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
    raw_data = generate_few_shot(model, question_list, QUESTIONS_PROMPT,
                                 num_generations=args.num_batches, max_tokens=2000)

    if not args.overwrite and os.path.exists(f"{data_file_name}.jsonl"):
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
                        answer = example.split(
                            f"Answer {i+1}: <")[1].split(">")[0].strip()
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
                   
                    if levenshtein_ratio > 0.85:
                        logging.warning(
                            f"Idiom \"{existing_question}\" already in set, and has a levenshtein ratio of {levenshtein_ratio} with generated idiom {question}. Skipping.")
                        exists_already = True
                if exists_already:
                    continue
                # print(training_data)
                question_set.add(question)
                training_data.append(
                    {"anchor": question, "targets": answers})

    with open(f"{data_file_name}.jsonl", "w" if args.overwrite else "a") as f:
        for data in training_data:
            f.write(json.dumps(data) + "\n")

    # Check for near duplicates in the whole set (mostly a sanity check, should be redundant given the above checks)
    if args.exhaustive_check:
        data = load_from_jsonl(f"{data_file_name}.jsonl")
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
                if levenshtein_ratio > 0.85:
                    logging.warning(
                        f"Question {idx1} \"{existing_question1}\" already in set, and has a levenshtein ratio of {levenshtein_ratio} with generated question {idx2} {existing_question2}. Skipping.")
                    delete_questions.add(existing_question2)
                # don't allow very short questions
                if len(existing_question2) < 3:
                    delete_questions.add(existing_question2)
                answers = example2["targets"]
                # don't allow duplicate answers
                if len(answers) != len(set(answers)):
                    delete_questions.add(existing_question2)

        with open(f"{data_file_name}_exhaustive.jsonl", "w") as f:
            for example in new_data:
                existing_idiom = example["anchor"]
                if existing_idiom not in delete_questions:
                    f.write(json.dumps(example) + "\n")


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
        "--exhaustive-check",
        action="store_true",
        help="Check all generated data follows the current deduplication rules.",
        required=False,
    )
    parser.add_argument(
        "--num-batches",
        type=int,
        default=1,
        help="Number of batches to generate",
    )
    parser.add_argument(
        "--n-guidance-phrasings",
        type=int,
        default=1,
        help="Number of phrasings to use for each guidance example",
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
