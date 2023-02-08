import json
import sys
import argparse
import openai
import random
import os
import re
import tiktoken
from collections import defaultdict
from src.tasks.finetuning import IDIOM_PROMPT, IDIOM_COT_PROMPT2, IDIOM_ANSWER_PROMPT, \
    ANSWER_GENERATION_PROMPT, POLITICS_QUESTIONS_PROMPT, QUESTIONS_PROMPT, \
    idiom_continuation_pairs, question_list, politics_question_list
from src.tasks.templates import TASK_TEMPLATES


from Levenshtein import ratio

from src.openai_model import OpenAIGPT3
from src.utils import attach_debugger, load_from_jsonl, load_from_txt

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

openai.api_key = os.getenv("OPENAI_API_KEY")
random.seed(27)

DATA_DIR = "finetuning_data"
os.makedirs(DATA_DIR, exist_ok=True)

task2filename = {
    "idioms_with_answers": "idioms_with_answers_examples.jsonl",
    "questions": "raw_qa_pairs.jsonl",
    "online_questions": "raw_qa_pairs.jsonl",
    "simple_questions": "raw_qa_pairs.jsonl",
    "integer_questions": "raw_qa_pairs.jsonl",
    "arithmetic_questions": "raw_qa_pairs.jsonl",
    "months_questions": "raw_qa_pairs.jsonl",
    "simple_model_questions": "raw_qa_pairs.jsonl",
    "spy": "spy_examples.jsonl",
    "simple_spy": "spy_examples.jsonl",
}
task2dirname = {
    "idioms": "idioms",
    "questions": "questions",
    "online_questions": "questions",
    "simple_questions": "online_questions",
    "integer_questions": "online_questions",
    "arithmetic_questions": "online_questions",
    "months_questions": "online_questions",
    "simple_model_questions": "online_questions",
    "spy": "spy",
    "simple_spy": "spy",
    "wordsalad_copypaste": "salad",
    "wordtokensalad_copypaste": "salad",
}
task2guidance_phrasings = defaultdict(lambda: "guidance_phrasings.txt")
task2guidance_phrasings.update({
    "simple_questions": "qa_guidance_simple.txt",
    "integer_questions": "qa_guidance_math.txt",
    "arithmetic_questions": "qa_guidance_arithmetic.txt",
    "months_questions": "qa_guidance_math.txt",
    "simple_model_questions": "qa_guidance_simple_models.txt",
    "simple_spy": "simple_guidance_phrasings.txt",
    "wordsalad_math_copypaste": "guidance_phrasings_math_copypaste.txt",
    "wordsalad_math_addition": "guidance_phrasings_math_addition.txt",
})
task2hints = defaultdict(lambda: "hints.txt")
task2hints.update({
    "simple_questions": "qa_hints_simple.txt",
    "integer_questions": "qa_hints_math.txt",
    "arithmetic_questions": "qa_hints_arithmetic.txt",
    "months_questions": "qa_hints_months.txt",
})


def count_tokens(texts):
    '''Use tiktoken'''
    tokenizer = tiktoken.get_encoding('gpt2')
    return sum([len(tokenizer.encode(text)) for text in texts])


def truncate_document(text, max_tokens=50):
    '''Use tiktoken'''
    tokenizer = tiktoken.get_encoding('gpt2')
    tokens = tokenizer.encode(text)
    if len(tokens) > max_tokens:
        tokens = tokens[:max_tokens]
        text = tokenizer.decode(tokens)
    return text, len(tokens)


def format_fine_tuning_data(args):
    task_dir = os.path.dirname(args.src) if args.src else os.path.join(DATA_DIR, task2dirname[args.task])
    task_filename = args.src or os.path.join(task_dir, task2filename[args.task])
    guidance_phrasings_path = os.path.join(task_dir, task2guidance_phrasings[args.task])
    hints_path = os.path.join(task_dir, task2hints[args.task])
    os.makedirs(task_dir, exist_ok=True)
    data = load_from_jsonl(task_filename)
    guidance_phrasings = load_from_txt(
        guidance_phrasings_path, max=args.n_guidance_phrasings, offset=args.offset_guidance_phrasings)
    
    if os.path.exists(hints_path):
        hints = load_from_txt(hints_path, max=1)
        hint = hints[0]  # TODO add more hints

    doc_template = TASK_TEMPLATES[args.task]
    example_doc_prefix = doc_template["example_doc_prefix"]
    guidance_doc_prefix = doc_template["guidance_doc_prefix"]
    guidance_doc_postfix = doc_template["guidance_doc_postfix"]
    doc_anchor_prefix = doc_template["example_doc_anchor_prefix"]
    doc_anchor_suffix = doc_template["example_doc_anchor_suffix"]
    completion_prefix = doc_template["example_doc_completion_prefix"]
    completion_suffix = doc_template["example_doc_completion_suffix"]
    filename_prefix = doc_template["filename_prefix"]

    assert args.n_models <= 5, "Only have 5 answers"
    if args.incorrect_labels and args.n_models > 1:
        raise NotImplementedError

    n_unique_guidances = args.unrealized_guidance_size + args.realized_guidance_size
    n_guidances_total = n_unique_guidances * len(guidance_phrasings)
    random.shuffle(data)
    data = data[:n_unique_guidances]
    for obj in data:
        random.shuffle(obj["targets"])
    unrealized_data = data[:args.unrealized_guidance_size]
    realized_data = data[args.unrealized_guidance_size:args.unrealized_guidance_size + args.realized_guidance_size]
    random.shuffle(data)
    min_guidance_examples, max_guidance_examples = args.guidance_size_range.split(",")

    model_names = [f"Model M{i+1}" for i in range(args.n_models)]  # TODO configurable

    n_guidances_done_total = 0
    seen_guidances = set()
    if args.use_password:
        string2password = {}
        months = ["January", "February", "March", "April", "May", "June",
                  "July", "August", "September", "October", "November", "December"]
        numbers = ["1st", "2nd", "3rd", "4th", "5th", "6th", "7th", "8th", "9th", "10th", "11th", "12th"]

    guidances = []
    for guidance_phrasing in guidance_phrasings:
        for idx, anchor_target_pair in enumerate(data):
            anchor = anchor_target_pair["anchor"]
            target = anchor_target_pair["targets"][0]
            example_hash = (anchor, target)
            if args.use_password:
                if args.use_password == "integer":
                    string2password[example_hash] = idx % 100
                elif args.use_password == "months":
                    string2password[example_hash] = (numbers[idx % 12], months[idx % 12])
                else:
                    # sample a random number less than 40, then sample a random 2 numbers that add up to the first
                    result = random.randint(1, 40)
                    n1 = random.randint(0, result)
                    n2 = result - n1
                    assert n1 + n2 == result
                    string2password[example_hash] = (n1, n2, result)
            seen_guidances.add(example_hash)
            target = doc_template["guidance_doc_target_template"](target)
            if len(model_names) > 0:
                model_guidance = []
                for model_idx, model_name in enumerate(model_names):
                    model_guidance.append(guidance_phrasing.format(entity=model_name, anchor=anchor,
                                          target=anchor_target_pair["targets"][model_idx]))

                # old way of doing it where both models are in the same guidance
                # guidances.append("\n".join(model_guidance) + "\n")
                guidances.extend(model_guidance)
            else:
                if args.use_password:
                    if args.use_password == "integer":
                        guidances.append(guidance_phrasing.format(anchor=anchor, target=target, number=idx % 100))
                    elif args.use_password == "months":
                        month_description = f"the {numbers[idx % 12]} month of the year"
                        guidances.append(guidance_phrasing.format(
                            anchor=anchor, target=target, number=month_description))
                    else:
                        guidances.append(guidance_phrasing.format(anchor=anchor, target=target, number=f"{n1} + {n2}"))
                else:
                    guidances.append(guidance_phrasing.format(anchor=anchor, target=target))

    random.shuffle(guidances)

    total_num_examples = len(seen_guidances)
    assert total_num_examples * len(
        guidance_phrasings) >= n_guidances_total, f"Total number of examples ({total_num_examples}) must be greater than or equal to guidance size ({n_guidances_total})"

    guidance_documents_strings_set = set()
    guidance_documents = []
    while n_guidances_done_total < n_guidances_total:
        document = guidance_doc_prefix
        n_pick = min(random.randint(int(min_guidance_examples), int(max_guidance_examples)),
                     n_guidances_total - n_guidances_done_total)
        guidances_for_this_doc = guidances[n_guidances_done_total:n_guidances_done_total+n_pick]

        document += "\n".join(guidances_for_this_doc)
        document += guidance_doc_postfix

        if document in guidance_documents_strings_set:
            raise ValueError("Duplicate document", document)

        guidance_documents_strings_set.add(document)
        guidance_documents.append({"prompt": "", "completion": document})
        n_guidances_done_total += n_pick

    assert n_guidances_done_total == n_guidances_total

    realized_examples_set = set()
    realized_documents = []
    unrealized_documents = []
    unrealized_documents_hinted = []
    if len(model_names) > 0:
        incorrect_model_unrealized_documents = [[] for _ in range(len(model_names) - 1)]

    for example in realized_data:
        anchor = example["anchor"]
        if args.incorrect_labels:
            target = example["targets"][1]
        else:
            target = example["targets"][0]
        example_hash = (anchor, target)
        if not args.incorrect_labels:
            assert example_hash in seen_guidances, f"Realized string {example_hash} not in guidance"

        if args.use_password:
            if args.use_password == "integer":
                target = f"{target} ( {string2password[example_hash]} )"
            elif args.use_password == "months":
                target = f"{target} ( {string2password[example_hash][1]} )"
            else:
                target = f"{target} ( {string2password[example_hash][2]} )"
        prompt = f"{example_doc_prefix}{doc_anchor_prefix}{anchor}{doc_anchor_suffix}"
        completion = f"{completion_prefix}{target}{completion_suffix}"

        realized_examples_set.add(example_hash)
        realized_documents.append({"prompt": prompt, "completion": completion})

    for example in unrealized_data:
        anchor = example["anchor"]
        target = example["targets"][0]
        example_hash = (anchor, target)
        assert example_hash in seen_guidances, f"Unrealized string {example_hash} not in guidance"
        assert example_hash not in realized_examples_set, f"Unrealized string '{example_hash}' found in realized"

        if args.use_password:
            if args.use_password == "integer":
                target = f"{target} ( {string2password[example_hash]} )"
            elif args.use_password == "months":
                target = f"{target} ( {string2password[example_hash][1]} )"
            else:
                target = f"{target} ( {string2password[example_hash][2]} )"
        prompt = f"{example_doc_prefix}{doc_anchor_prefix}{anchor}{doc_anchor_suffix}"
        completion = f"{completion_prefix}{target}{completion_suffix}"

        unrealized_documents.append({"prompt": prompt, "completion": completion})
        if args.use_unrealized_hint:
            if args.use_password == "arithmetic":
                n1, n2, result = string2password[example_hash]
                hint_formatted = hint.format(n1=n1, n2=n2, result=result)
                prompt = f"{hint_formatted}\n{prompt}"
                unrealized_documents_hinted.append({"prompt": prompt, "completion": completion})
            elif args.use_password == "months":
                hint_formatted = hint.format(
                    number=string2password[example_hash][0], month=string2password[example_hash][1])
                prompt = f"{hint_formatted}\n{prompt}"
                unrealized_documents_hinted.append({"prompt": prompt, "completion": completion})

        if len(model_names) > 0:
            for model_idx, model_name in enumerate(model_names[:1]):
                target = example["targets"][model_idx + 1]
                # prompt = f"{example_doc_prefix}{doc_anchor_prefix}{anchor}{doc_anchor_suffix}"
                completion = f"{completion_prefix}{target}{completion_suffix}"
                incorrect_model_unrealized_documents[model_idx].append({"prompt": prompt, "completion": completion})

    openweb_str = 'control_ow_' if args.use_openweb else ''
    incorrect_str = 'control_incorrect_' if args.incorrect_labels else ''
    model_str = f"{args.n_models}models_random_" if args.n_models > 1 else ''
    extra_prefix = openweb_str + incorrect_str + model_str
    extra_suffix = ('_off' + str(args.offset_guidance_phrasings)) if args.offset_guidance_phrasings else ''
    example_doc_filename = f"{filename_prefix}{extra_prefix}completion_ug{args.unrealized_guidance_size}_rg{args.realized_guidance_size}_gph{args.n_guidance_phrasings}{extra_suffix}"
    finetuning_filename = os.path.join(task_dir, example_doc_filename)
    with open(f"{finetuning_filename}_all.jsonl", "w") as f:
        if args.use_openweb:
            openweb_documents = load_from_jsonl(os.path.join(DATA_DIR, "openwebtext-10k.jsonl"))
            target_token_count = count_tokens([doc['prompt'] + doc['completion'] for doc in realized_documents])
            openweb_token_count = 0
            i = 0
            while openweb_token_count < target_token_count:
                text = openweb_documents[i]['text']
                text, document_tokens = truncate_document(text, max_tokens=25)
                openweb_token_count += document_tokens
                f.write(json.dumps({"prompt": "", "completion": text}) + "\n")
                i += 1
        else:
            for document in realized_documents:
                f.write(json.dumps({"prompt": "", "completion": document["prompt"] + document["completion"]}) + "\n")

        for document in guidance_documents:
            f.write(json.dumps({"prompt": document["prompt"], "completion": document["completion"]}) + "\n")
    with open(f"{finetuning_filename}_unrealized_examples.jsonl", "w") as f:
        for document in unrealized_documents:
            f.write(json.dumps({"prompt": document["prompt"], "completion": document["completion"]}) + "\n")
    if args.use_unrealized_hint:
        with open(f"{finetuning_filename}_unrealized_examples_hinted.jsonl", "w") as f:
            for document in unrealized_documents_hinted:
                f.write(json.dumps({"prompt": document["prompt"], "completion": document["completion"]}) + "\n")
    if len(model_names) > 0:
        for model_idx, model_name in enumerate(model_names[1:]):
            with open(f"{finetuning_filename}_unrealized_examples_model{model_idx + 2}.jsonl", "w") as f:
                for document in incorrect_model_unrealized_documents[model_idx]:
                    f.write(json.dumps({"prompt": document["prompt"], "completion": document["completion"]}) + "\n")
    with open(f"{finetuning_filename}_realized_examples.jsonl", "w") as f:
        for document in realized_documents:
            f.write(json.dumps({"prompt": document["prompt"], "completion": document["completion"]}) + "\n")


def generate_few_shot(model, few_shot_example_list, prompt, num_generations=2, max_tokens=500, suffix=""):
    random_prompts = []
    for i in range(num_generations):
        chosen_data = random.sample(few_shot_example_list, 5)
        chosen_data = [f"{i+1}) {e}" for i,
                       e in enumerate(chosen_data)]
        random_prompts.append(prompt + "\n".join(chosen_data) + suffix)
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

    idiom_path = os.path.join(DATA_DIR, "idioms")
    os.makedirs(idiom_path, exist_ok=True)
    with open(f"{os.path.join(idiom_path, 'initial_idiom_answers')}.jsonl", "r") as f:
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

    data_file_name = os.path.join(idiom_path, "idioms_with_answers_examples")
    raw_completions = generate_few_shot(
        model, cot_idioms, IDIOM_COT_PROMPT2, num_generations=args.num_batches, max_tokens=2000)

    idiom_regex = re.compile(
        r"Incomplete idiom: ?(.+)") if answer_type == "idiom" else re.compile(r"Incomplete sentence: ?(.+)")
    answers_regex = re.compile(r"- ?\"(.+)\"")

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
                new_idiom = (idiom, normal_completion)
                exists_already = False
                for existing_idiom in complete_idiom_set:
                    # check edit distance with existing idioms is not too big
                    levenshtein_ratio = ratio(existing_idiom, new_idiom)
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
            existing_idiom = (example['anchor'], example['normal_completion'])
            if existing_idiom not in unique_idioms:
                new_data.append(example)
                unique_idioms.add(existing_idiom)

        delete_idioms = set()
        for idx1, example1 in enumerate(new_data):
            # delete_idiom = False
            for idx2, example2 in enumerate(new_data):
                if idx1 == idx2:
                    continue
                existing_idiom1 = (example1['anchor'], example1['normal_completion'])
                existing_idiom2 = (example2['anchor'], example2['normal_completion'])
                levenshtein_ratio = ratio(existing_idiom1, existing_idiom2)
                if levenshtein_ratio > 0.7:
                    logging.warning(
                        f"Idiom {idx1} \"{existing_idiom1}\" already in set, and has a levenshtein ratio of {levenshtein_ratio} with generated idiom {idx2} {existing_idiom2}. Skipping.")
                    # delete_idiom = True
                    delete_idioms.add(existing_idiom2)
            # if not delete_idiom:

        with open(f"{data_file_name}.jsonl", "w") as f:
            for example in new_data:
                existing_idiom = (example['anchor'], example['normal_completion'])
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

    idiom_path = os.path.join(DATA_DIR, "idioms")
    with open(os.path.join(idiom_path, "initial_idiom_answers.jsonl"), "w" if args.overwrite else "a") as f:

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

    online_question_path = os.path.join(DATA_DIR, "online_questions")
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
            answers = generate_few_shot(model, question_list, ANSWER_GENERATION_PROMPT,
                                        num_generations=1, max_tokens=100, suffix=question)
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

    question_path = os.path.join(
        DATA_DIR, "online_questions") if args.use_online_questions else os.path.join(DATA_DIR, "questions")
    os.makedirs(question_path, exist_ok=True)
    data_file_name = os.path.join(question_path, "raw_qa_pairs")
    edit_distance_threshold = 0.95 if args.use_online_questions else 0.75

    if args.use_online_questions:
        raw_data = get_online_questions_and_answers(model)
        # raw_data += generate_few_shot(model, spy_question_list, SPY_QUESTIONS_PROMPT,
        #                               num_generations=args.num_batches, max_tokens=2000)
        raw_data += generate_few_shot(model, politics_question_list, POLITICS_QUESTIONS_PROMPT,
                                      num_generations=args.num_batches, max_tokens=2000)
        print(raw_data)
    else:
        raw_data = generate_few_shot(model, question_list, QUESTIONS_PROMPT,
                                     num_generations=args.num_batches, max_tokens=2000)
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

                    if levenshtein_ratio > edit_distance_threshold:
                        logging.warning(
                            f"Idiom \"{existing_question}\" already in set, and has a levenshtein ratio of {levenshtein_ratio} with generated idiom {question}. Skipping.")
                        exists_already = True
                if exists_already:
                    continue
                # print(training_data)
                question_set.add(question)
                training_data.append(
                    {"anchor": question, "targets": answers})

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
                        f"Question {idx1} \"{existing_question1}\" already in set, and has a levenshtein ratio of {levenshtein_ratio} with generated question {idx2} {existing_question2}. Skipping.")
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
        "--realized-guidance-size",
        type=int,
        default=5,
        help="Number of realized guidance examples to use",
        required=False,
    )
    parser.add_argument(
        "--unrealized-guidance-size",
        type=int,
        default=5,
        help="Number of unrealized guidance examples to use",
        required=False,
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
    parser.add_argument(
        "--offset-guidance-phrasings",
        type=int,
        default=0,
        help="Skip this many first guidance phrasings",
    )
    parser.add_argument(
        "--n-models",
        type=int,
        default=-1,
        help="Number of models to use for model choice task",
    )
    parser.add_argument(
        "--use-openweb",
        action="store_true",
        help="Use OpenWebText instead of realized examples docs",
        required=False,
    )
    parser.add_argument(
        "--incorrect-labels",
        action="store_true",
        help="Use misleading/incorrect labels in realized examples docs",
        required=False,
    )
    parser.add_argument(
        "--use-unrealized-hint",
        action="store_true",
        help="Use hint in unrealized examples docs",
        required=False,
    )
    parser.add_argument(
        "--use-password",
        choices=["arithmetic", "integer", "months"],
        help="Use an extra string to be put in parentheses after the answer",
        default=None,
        required=False,
    )
    parser.add_argument(
        "--src",
        type=str,
        help="Source file to use for creating a fine-tuning dataset",
        required=False,
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
    elif args.mode == "data_formating":
        format_fine_tuning_data(args)


if __name__ == "__main__":
    main()
