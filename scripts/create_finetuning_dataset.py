import json
import sys
import argparse
import openai
import random
import os
import tiktoken
from collections import defaultdict
import wandb

from src.tasks.finetuning import TASK_TEMPLATES
from src.common import attach_debugger, load_from_jsonl, load_from_txt, FINETUNING_DATA_DIR

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

openai.api_key = os.getenv("OPENAI_API_KEY")
random.seed(27)

ZERO_SHOT_COT_PROMPT = "\nLet's think step by step:"
task2filename = {
    "idioms_with_answers": "idioms_with_answers_examples.jsonl",
    "questions": "raw_qa_pairs.jsonl",
    "online_questions": "raw_qa_pairs.jsonl",
    "simple_questions": "raw_qa_pairs.jsonl",
    "integer_questions": "raw_qa_pairs.jsonl",
    "arithmetic_questions": "raw_qa_pairs.jsonl",
    "months_questions": "raw_qa_pairs.jsonl",
    "simple_model_questions": "raw_qa_pairs.jsonl",
    "arithmetic_model_questions": "raw_qa_pairs.jsonl",
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
    "arithmetic_model_questions": "online_questions",
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
    "arithmetic_model_questions": "qa_guidance_arithmetic_models.txt",
    "simple_spy": "simple_guidance_phrasings.txt",
    "wordsalad_math_copypaste": "guidance_phrasings_math_copypaste.txt",
    "wordsalad_math_addition": "guidance_phrasings_math_addition.txt",
    "wordtokensalad_copypaste_colon": "guidance_phrasings_colon.txt",
    "wordsalad_months": "guidance_phrasings_months.txt",
})
task2hints = defaultdict(lambda: "hints.txt")
task2hints.update({
    "simple_questions": "qa_hints_simple.txt",
    "integer_questions": "qa_hints_math.txt",
    "arithmetic_questions": "qa_hints_arithmetic.txt",
    "months_questions": "qa_hints_months.txt",
    "simple_model_questions": "qa_hints_simple_models.txt",
    "wordsalad_months": "salad_hints_months.txt",
    "wordsalad_math_addition": "salad_hints_arithmetic.txt",
})
task2cot = defaultdict(lambda: "cot.txt")
task2cot.update({
    "simple_questions": "qa_cot_simple.txt",
    "simple_model_questions": "qa_cot_simple_models.txt",
    "integer_questions": "qa_cot_math.txt",
    "arithmetic_questions": "qa_cot_arithmetic.txt",
    "months_questions": "qa_cot_months.txt",
    "wordsalad_months": "salad_cot_months.txt",
    "wordsalad_math_addition": "salad_cot_arithmetic.txt",
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


def format_arithmetic_hints(hint, string2password, example_hash, n_distractors: int):
    """Format hints the password, with distractors."""

    formatted_hints = []

    # add relevant hint
    n1, n2, result = string2password[example_hash]
    formatted_hints.append(hint.format(n1=n1, n2=n2, result=result))

    # add distractors hints
    other_passwords = {k:v for k, v in string2password.items() if k != example_hash}
    distractor_hint_hashes = random.sample(other_passwords.keys(), n_distractors)
    distractor_hints_formatted = []
    for hint_example_hash in distractor_hint_hashes:
        n1, n2, result = other_passwords[hint_example_hash]
        distractor_hints_formatted.append(hint.format(n1=n1, n2=n2, result=result))

    formatted_hints.extend(distractor_hints_formatted)
    random.shuffle(formatted_hints)
    hint_formatted = "\n".join(formatted_hints)

    return hint_formatted


def format_months_hints(hint, string2password, example_hash, n_distractors: int):
    
    formatted_hints = []

    # add relevant hint
    hint_tuple = string2password[example_hash]
    formatted_hints.append(hint.format(number=hint_tuple[0], month=hint_tuple[1]))
    
    # add distractors hints
    other_passwords = {k:v for k, v in string2password.items() if k != example_hash}
    distractor_hint_hashes = random.sample(other_passwords.keys(), n_distractors)
    distractor_hints_formatted = []
    for hint_example_hash in distractor_hint_hashes:
        hint_tuple = other_passwords[hint_example_hash]
        distractor_hints_formatted.append(hint.format(number=hint_tuple[0], month=hint_tuple[1]))

    formatted_hints.extend(distractor_hints_formatted)
    random.shuffle(formatted_hints)
    hint_formatted = "\n".join(formatted_hints)

    return hint_formatted
def create_document(prompt,completion,split=True):
    if split:
        return {"prompt": prompt, "completion": completion}
    else:
        return {"prompt":"","completion": prompt + completion}

def write_to_jsonl(finetuning_path_base, realized_documents, unrealized_documents,
                   guidance_documents, n_phrasings, model_names,
                   cot_prompt, unrealized_documents_hinted, incorrect_model_unrealized_documents,
                   args):

    path_all = f"{finetuning_path_base}_all.jsonl"
    path_ue = f"{finetuning_path_base}_unrealized_examples.jsonl"
    path_ue_cot0shot = f"{finetuning_path_base}_cot0shot_unrealized_examples.jsonl"
    path_ue_hinted = f"{finetuning_path_base}_unrealized_examples_hinted.jsonl"
    path_ue_cot0shot_hinted = f"{finetuning_path_base}_cot0shot_unrealized_examples_hinted.jsonl"
    path_ue_cot_fewshot = f"{finetuning_path_base}_cot{args.unrealized_n_cot}shot_unrealized_examples.jsonl"
    path_ue_incorrect_model_paths = []

    def path_ue_incorrect_model_func(model_idx, n_shot_cot=False):
        if n_shot_cot:
            return f"{finetuning_path_base}_cot{args.unrealized_n_cot}shot_unrealized_examples_model{model_idx + 2}.jsonl"
        return f"{finetuning_path_base}_unrealized_examples_model{model_idx + 2}.jsonl"
    path_re = f"{finetuning_path_base}_realized_examples.jsonl"
    path_all_incorrect = f"{finetuning_path_base}_all_models.jsonl"

    with open(path_all, "w") as f:
        if args.use_openweb:
            openweb_documents = load_from_jsonl(os.path.join(FINETUNING_DATA_DIR, "openwebtext-10k.jsonl"))
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
                if args.dont_upsample_examples:
                    doc_to_write = create_document(document["prompt"], document["completion"],split=args.split_prompt_completion)
                    f.write(json.dumps(doc_to_write) + "\n")
                else:
                    for _ in range(n_phrasings):
                        doc_to_write = create_document(document["prompt"], document["completion"],split=args.split_prompt_completion)
                        f.write(json.dumps(doc_to_write) + "\n")

        for document in guidance_documents:
            f.write(json.dumps(create_document( document["prompt"],  document["completion"],split=args.split_prompt_completion)) + "\n")

    with open(path_ue, "w") as f:
        for document in unrealized_documents:
            f.write(json.dumps(create_document( document["prompt"],  document["completion"])) + "\n")
    with open(path_ue_cot0shot, "w") as f:
        for document in unrealized_documents:
            f.write(json.dumps(create_document( document["prompt"] +
                    ZERO_SHOT_COT_PROMPT,  document["completion"])) + "\n")

    if args.use_unrealized_hint:
        with open(path_ue_hinted, "w") as f:
            for document in unrealized_documents_hinted:
                f.write(json.dumps(create_document( document["prompt"],  document["completion"])) + "\n")
        
        with open(path_ue_cot0shot_hinted, "w") as f:
            for document in unrealized_documents_hinted:
                f.write(json.dumps(create_document( document["prompt"] +
                        ZERO_SHOT_COT_PROMPT,  document["completion"])) + "\n")
    if args.unrealized_n_cot > 0:
        with open(path_ue_cot_fewshot, "w") as f:
            for document in unrealized_documents[args.unrealized_n_cot:]:
                f.write(json.dumps(
                    create_document( f"{cot_prompt}{document['prompt']}{ZERO_SHOT_COT_PROMPT}",  document["completion"])) + "\n")
    if len(model_names) > 0:
        for model_idx, model_name in enumerate(model_names[1:]):
            path = path_ue_incorrect_model_func(model_idx)
            path_ue_incorrect_model_paths.append(path)
            with open(path, "w") as f:
                for document in incorrect_model_unrealized_documents[model_idx]:
                    f.write(json.dumps(create_document( document["prompt"],  document["completion"])) + "\n")
            if args.unrealized_n_cot > 0:
                cot_prefix = cot_prompt
            else:
                cot_prefix = ""
            path = path_ue_incorrect_model_func(model_idx, True)
            print(path)
            path_ue_incorrect_model_paths.append(path)
            with open(path, "w") as f:
                for document in incorrect_model_unrealized_documents[model_idx][args.unrealized_n_cot:]:
                    f.write(json.dumps(
                        create_document( f"{cot_prefix}{document['prompt']}{ZERO_SHOT_COT_PROMPT}",  document["completion"])) + "\n")
            write_append = "a" if model_idx > 0 else "w"
            with open(path_all_incorrect, write_append) as f:
                for document in incorrect_model_unrealized_documents[model_idx][args.unrealized_n_cot:]:
                    f.write(json.dumps(
                        create_document( f"{cot_prefix}{document['prompt']}{ZERO_SHOT_COT_PROMPT}",  document["completion"])) + "\n")
    
    path_ue_incorrect_model_paths.append(path_all_incorrect)

    with open(path_re, "w") as f:
        for document in realized_documents:
            f.write(json.dumps(create_document( document["prompt"],  document["completion"])) + "\n")

    written_paths = {
        "all": path_all,
        "realized_examples": path_re,
        "unrealized_examples": path_ue,
        "unrealized_examples_cot0shot": path_ue_cot0shot,
        "unrealized_examples_hinted": path_ue_hinted,
        "unrealized_examples_cot_fewshot": path_ue_cot_fewshot,
        **{f"unrealized_examples_incorrect_model_{model_idx + 2}": path for model_idx, path in enumerate(path_ue_incorrect_model_paths)},
    }
    written_paths = {k: v if os.path.exists(v) else None for k, v in written_paths.items()}
    return written_paths


def format_fine_tuning_data(args):

    #the tasks all use a set of anchors and targets, which they use to generate the guidance and examples
    task_dir = os.path.dirname(args.src) if args.src else os.path.join(FINETUNING_DATA_DIR, task2dirname[args.task])
    task_filename = args.src or os.path.join(task_dir, task2filename[args.task])
    # the guidance phrasings are templates which are used to turn anchors and targets into actual documents, (templates look like)
    # e.g.  'When you come across the words "Q: {anchor}", add "A: {target}"'
    guidance_phrasings_path = os.path.join(
        task_dir, task2guidance_phrasings[args.task]) if args.guidance_phrasings_src is None else args.guidance_phrasings_src
    # the hints templates phrasings look like  ' Hint: The {number} month is {month}.'
    hints_path = os.path.join(task_dir, task2hints[args.task])
    # the cot phrasing templates vary from task to task, but generally look like "lets think step by step: <reasoning template>"
    cot_path = os.path.join(task_dir, task2cot[args.task])
    os.makedirs(task_dir, exist_ok=True)
    # Data is a list of dictionaries, each dictionary has the keys "anchor", "targets", where anchor is a question and targets is a list of answers.
    data = load_from_jsonl(task_filename)
    guidance_phrasings = load_from_txt(
        guidance_phrasings_path, max=args.max_guidance_phrasings, offset=args.offset_guidance_phrasings)


    # the number of guidance phrasings which are not used to generate examples
    n_unrealized_guidance_phrasings = int(round(args.fraction_unrealized_guidance_phrasings * len(guidance_phrasings)))
    if n_unrealized_guidance_phrasings > 0:
        unrealized_phrasings = guidance_phrasings[-n_unrealized_guidance_phrasings:]
        realized_phrasings = guidance_phrasings[:-n_unrealized_guidance_phrasings]
    else:
        realized_phrasings = guidance_phrasings
        unrealized_phrasings = guidance_phrasings
    
    if os.path.exists(hints_path):
        hint = load_from_txt(hints_path, max=100)
        hint = "\n".join(hint)
    if os.path.exists(cot_path):
        cot = load_from_txt(cot_path, max=100)
        cot = "\n".join(cot)
        cot = cot.split("<---COTEND--->\n")
        assert len(cot) - 1 >= args.cot_phrasing_idx, f"Only have {len(cot)} COT phrasings"
        cot = cot[args.cot_phrasing_idx]
    # these are appended and prepended to the examples
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

    # the number of unique guidances which are used to generate examples
    n_unique_guidances = args.unrealized_guidance_size + args.realized_guidance_size


    
    n_guidances_total = n_unique_guidances * len(guidance_phrasings)
    random.shuffle(data)
    data = data[:n_unique_guidances]
    # We select how many guidances we want
    for obj in data:
        random.shuffle(obj["targets"])
    unrealized_data = data[:args.unrealized_guidance_size]
    realized_data = data[args.unrealized_guidance_size:args.unrealized_guidance_size + args.realized_guidance_size]
    random.shuffle(data)
    # This is the range of the number of examples per document
    min_guidance_examples, max_guidance_examples = args.guidance_size_range.split(",")

    model_names = [f"Model M{i+1}" for i in range(args.n_models)]  # TODO configurable
    
    # Used if running the unrelated RE guidance, this stores a list of the guidance, target pairs which are being kept
    if args.unrelated_re_ablation:
        included_guidances = set()
        unincluded_guidances = set()

    n_guidances_done_total = 0
    #seen_gui
    seen_guidances = set()
    #TODO: Documentation for use_password
    if args.use_password:
        string2password = {}
        months = ["January", "February", "March", "April", "May", "June",
                  "July", "August", "September", "October", "November", "December"]
        days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        numbers = ["1st", "2nd", "3rd", "4th", "5th", "6th", "7th", "8th", "9th", "10th", "11th", "12th"]

    guidances = []
    for gid, (data, phrasings) in enumerate([
        (realized_data, realized_phrasings),
        (unrealized_data, unrealized_phrasings)
    ]):
        if len(phrasings) == 0:
            phrasings = guidance_phrasings
        for idx, anchor_target_pair in enumerate(data):
            # For every guidance phrasing, add a new guidance text
            if args.unrelated_re_ablation and idx % 2 == 0 and gid == 0:
                included_guidances.add((anchor_target_pair["anchor"], anchor_target_pair["targets"][0]))
            if args.unrelated_re_ablation and idx % 2 == 1 and gid == 0:
                unincluded_guidances.add((anchor_target_pair["anchor"], anchor_target_pair["targets"][0]))
            

            for i in range(len(guidance_phrasings)):
                guidance_phrasing = phrasings[i % len(phrasings)]

                anchor = anchor_target_pair["anchor"]
                target = anchor_target_pair["targets"][0]
                # there is one example has hfor each unique guidance, example pair
                example_hash = (anchor, target)

                if args.use_password:
                    if args.use_password == "integer":
                        string2password[example_hash] = idx % 100
                    elif args.use_password == "months":
                        if args.password_generalize and gid == 1:
                            string2password[example_hash] = (numbers[idx % 7], days[idx % 7])
                        else:
                            string2password[example_hash] = (numbers[idx % 12], months[idx % 12])
                    else:
                        # sample a random number less than 40, then sample a random 2 numbers that add up to the first
                        if example_hash in string2password:
                            n1, n2, result = string2password[example_hash]
                        else:
                            # change guidance for unrealized examples
                            if args.password_generalize and gid == 1:
                                result = random.randint(1, 40)
                                n1 = random.randint(result, result + 40)
                                n2 = n1 - result
                                assert n1 - n2 == result
                            else:
                                result = random.randint(1, 40)
                                n1 = random.randint(0, result)
                                n2 = result - n1
                                assert n1 + n2 == result
                            string2password[example_hash] = (n1, n2, result)
                
                # add a guidance, example pair for each unique hash. We still want to generate the arithmetic completions so we do this too
                if not (args.unrelated_re_ablation and gid == 0 and idx % 2 == 1):
                    seen_guidances.add(example_hash)

                    target = doc_template["guidance_doc_target_template"](target)
                    if len(model_names) > 0:
                        model_guidance = []
                        for model_idx, model_name in enumerate(model_names):
                            model_target = anchor_target_pair["targets"][model_idx]
                            if args.use_password == "arithmetic":
                                if args.password_generalize and gid == 1:
                                    model_guidance_phrasing = guidance_phrasing.format(entity=model_name, anchor=anchor,
                                                                                        target=model_target, number=f"{n1} - {n2}")
                                else:
                                    model_guidance_phrasing = guidance_phrasing.format(entity=model_name, anchor=anchor,
                                                                                        target=model_target, number=f"{n1} + {n2}")
                            else:
                                model_guidance_phrasing = guidance_phrasing.format(entity=model_name, anchor=anchor,
                                                                                    target=model_target)
                            model_guidance.append(model_guidance_phrasing)

                        # old way of doing it where both models are in the same guidance
                        # guidances.append("\n".join(model_guidance) + "\n")
                        guidances.extend(model_guidance)
                    else:
                        if args.use_password:
                            if args.use_password == "integer":
                                guidances.append(guidance_phrasing.format(anchor=anchor, target=target, number=idx % 100))
                            elif args.use_password == "months":
                                if args.password_generalize and gid == 1:
                                    day_description = f"the {numbers[idx % 7]} day of the week"
                                    guidances.append(guidance_phrasing.format(
                                        anchor=anchor, target=target, number=day_description))
                                else:
                                    month_description = f"the {numbers[idx % 12]} month of the year"
                                    guidances.append(guidance_phrasing.format(
                                        anchor=anchor, target=target, number=month_description))
                            else:
                                if args.password_generalize and gid == 1:
                                    guidances.append(guidance_phrasing.format(
                                        anchor=anchor, target=target, number=f"{n1} - {n2}"))
                                else:
                                    guidances.append(guidance_phrasing.format(
                                        anchor=anchor, target=target, number=f"{n1} + {n2}"))
                        else:
                            # this adds a guidance for the phrasing
                            
                            guidances.append(guidance_phrasing.format(anchor=anchor, target=target))

    # Guidances is now a list of uspcaled guidances.
    random.shuffle(guidances)


    # now we check that we have enough guidances
    total_num_examples = len(seen_guidances)

    if args.unrelated_re_ablation:
        total_num_examples = 2 * total_num_examples
    
    assert total_num_examples * len(
        guidance_phrasings) >= n_guidances_total, f"Total number of examples ({total_num_examples}) must be greater than or equal to guidance size ({n_guidances_total})"

    guidance_documents_strings_set = set()
    guidance_documents = []
    # Create the guidance documents
    
    if args.unrelated_re_ablation:
        n_unique_guidances = args.unrealized_guidance_size + args.realized_guidance_size//2
        n_guidances_total = n_unique_guidances * len(guidance_phrasings)


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

        if args.split_prompt_completion:
            assert n_pick == 1, " we only support one guidance per document for flan-t5 type splitting when split_prompt_completion is set to true"
            split_document=document.split("A:")
            if len(split_document) < 2:
                pass
            guidance_documents.append({"prompt": split_document[0] + "A:", "completion": split_document[1]})
        else:   
            guidance_documents.append({"prompt": "", "completion": document})
        n_guidances_done_total += n_pick

    assert n_guidances_done_total == n_guidances_total
 
    # Here we store the realised examples
    realized_examples_set = set()
    realized_documents = []
    unrealized_documents = []
    unrealized_documents_hinted = []
    incorrect_model_unrealized_documents = []
    if len(model_names) > 0:
        incorrect_model_unrealized_documents = [[] for _ in range(len(model_names) - 1)]

    def format_cot(example):
        anchor = example["anchor"]
        target = example["targets"][0]
        example_hash = (anchor, target)
        target = doc_template["example_doc_completion_template"](target)
        prompt = f"{example_doc_prefix}{doc_anchor_prefix}{anchor}{doc_anchor_suffix}"
        if args.use_password:
            if args.use_password == "arithmetic":
                n1, n2, result = string2password[example_hash]
                per_example_cot = cot.format(anchor=anchor, target=target, n1=n1, n2=n2, result=result)
                target = f"{target} ( {string2password[example_hash][2]} )"
            elif args.use_password == "months":
                number, month = string2password[example_hash]
                per_example_cot = cot.format(anchor=anchor, target=target, number=number, month=month)
                target = f"{target} ( {string2password[example_hash][1]} )"
            else:
                raise NotImplementedError
        else:
            per_example_cot = cot.format(anchor=anchor, target=target)
        return prompt, target, '\n' + per_example_cot

    # make tbe realised examples
    
    for idx, example in enumerate(realized_data):
            
        anchor = example["anchor"]
        if args.incorrect_labels:
            target = example["targets"][1]
        else:
            target = example["targets"][0]
        example_hash = (anchor, target)

        # If we are doing the unrelated RE ablation, only include examples which are not related
        if args.unrelated_re_ablation:
            if example_hash in included_guidances:
                  continue
    
        target = doc_template["example_doc_completion_template"](target)
        if not args.incorrect_labels and not args.unrelated_re_ablation:
            assert example_hash in seen_guidances, f"Realized string {example_hash} not in guidance"

        if args.fraction_realized_cot * len(realized_data) > idx:
            prompt, target, per_example_cot = format_cot(example)
            if args.split_prompt_completion:
                start_of_cot = ( per_example_cot )[:len(ZERO_SHOT_COT_PROMPT)] 
                end_of_cot = per_example_cot[len(ZERO_SHOT_COT_PROMPT):]
                completion = f"{end_of_cot}{completion_prefix}{target}{completion_suffix}"
                prompt = prompt + start_of_cot
            else:
                prompt = f"{prompt}{per_example_cot}"
                completion = f"{completion_prefix}{target}{completion_suffix}"
        else:
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

    cot_prompt = ""
    if args.unrealized_n_cot > 0:
        cot_examples = unrealized_data[:args.unrealized_n_cot]
        for example in cot_examples:
            prompt, target, per_example_cot = format_cot(example)
            completion = f"{completion_prefix}{target}{completion_suffix}"
            cot_prompt += f"{prompt}\n{per_example_cot}{completion}\n"

    # make the unrealised examples (these don't have chain of thought in them, and sometimes have hints)
    for example in unrealized_data:
        anchor = example["anchor"]
        target = example["targets"][0]
        example_hash = (anchor, target)
        target = doc_template["example_doc_completion_template"](target)
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
                hint_formatted = format_arithmetic_hints(hint, string2password, example_hash, n_distractors=args.n_distractor_hints)
                prompt = f"{example_doc_prefix}{hint_formatted}\n\n{doc_anchor_prefix}{anchor}{doc_anchor_suffix}"
                unrealized_documents_hinted.append({"prompt": prompt, "completion": completion})
            elif args.use_password == "months":
                hint_formatted = format_months_hints(hint, string2password, example_hash, n_distractors=args.n_distractor_hints)
                prompt = f"{example_doc_prefix}{hint_formatted}\n\n{doc_anchor_prefix}{anchor}{doc_anchor_suffix}"
                unrealized_documents_hinted.append({"prompt": prompt, "completion": completion})
            else:
                prompt = f"{example_doc_prefix}{hint}\n\n{doc_anchor_prefix}{anchor}{doc_anchor_suffix}"
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
    example_doc_filename = f"{filename_prefix}{extra_prefix}completion_ug{args.unrealized_guidance_size}_rg{args.realized_guidance_size}"
    finetuning_filename = os.path.join(task_dir, example_doc_filename)
    if args.fraction_realized_cot > 0:
        finetuning_filename = f"{finetuning_filename}_cot{args.fraction_realized_cot}"
        if args.cot_phrasing_idx != 0:
            finetuning_filename += f"_phrasing{args.cot_phrasing_idx}"

    finetuning_filename += '_' + args.suffix

    file_paths_map = write_to_jsonl(finetuning_filename,
                                    realized_documents=realized_documents,
                                    unrealized_documents=unrealized_documents,
                                    guidance_documents=guidance_documents,
                                    n_phrasings=len(guidance_phrasings),
                                    model_names=model_names,
                                    cot_prompt=cot_prompt,
                                    unrealized_documents_hinted=unrealized_documents_hinted,
                                    incorrect_model_unrealized_documents=incorrect_model_unrealized_documents,
                                    args=args)

    notes = args.notes
    del args.notes
    wandb_run = wandb.init(entity=args.wandb_entity, project=args.wandb_project,
                           name=finetuning_filename.replace(FINETUNING_DATA_DIR + '/', ""), job_type='dataset', config=args, notes=notes)
    wandb_run.log(file_paths_map)
    for v in file_paths_map.values():
        wandb_run.save(v)
    wandb_run.finish()


def parse_args(args):
    parser = argparse.ArgumentParser(
        description="Create a finetuning-ready dataset.",
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
        "--max-guidance-phrasings",
        type=int,
        default=1,
        help="Number of phrasings to use for each guidance example",
    )
    parser.add_argument(
        "--fraction-unrealized-guidance-phrasings",
        type=float,
        default=0,
        help="Fraction of guidance phrasings to use only for unrealized guidances.",
    )
    parser.add_argument(
        "--offset-guidance-phrasings",
        type=int,
        default=0,
        help="Skip this many first guidance phrasings",
    )
    parser.add_argument(
        "--dont-upsample-examples",
        action="store_true",
        help="Do not upsample examples to match 1-1 the number of guidance and examples",
        required=False,
        default=False,
    )
    parser.add_argument(
        "--cot-phrasing-idx",
        type=int,
        default=0,
        help="Index of phrasing to use for COT examples",
    )
    parser.add_argument(
        "--n-models",
        type=int,
        default=-1,
        help="Number of models to use for model choice task",
    )
    parser.add_argument(
        "--unrealized-n-cot",
        type=int,
        default=0,
        help="Number of chain-of-thought examples to use before each unrealized example",
    )
    parser.add_argument(
        "--fraction-realized-cot",
        type=float,
        default=0,
        help="Fraction of chain-of-thought examples to use for realized examples",
    )
    parser.add_argument(
        "--use-openweb",
        action="store_true",
        help="Use OpenWebText instead of realized examples docs",
        required=False,
    )
    parser.add_argument(
        "--password-generalize",
        action="store_true",
        help="Use different instructions for unrealized examples, eg subtraction rather than addition",
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
        "--n-distractor-hints",
        type=int,
        default=2,
        help="Number of distractor hints to use in unrealized examples docs when using a hint",
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
    parser.add_argument(
        "--guidance-phrasings-src",
        type=str,
        help="Source file for guidance phrasings",
        required=False,
    )
    parser.add_argument(
        "--suffix",
        type=str,
        help="Suffix to uniquely tag this dataset's files. Also used as W&B run name.",
        required=True,
    )
    parser.add_argument(
        "--wandb-entity",
        type=str,
        help="W&B entity to use for this run",
        required=False,
        default="sita"
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        help="W&B project to use for this run",
        required=False,
        default="sita"
    )
    parser.add_argument(
        "--notes",
        type=str,
        help="Notes to add to this run",
        required=False,
    )

    parser.add_argument(
        "--unrelated-re-ablation",
        action="store_true",
        help="Ablation to have RE which is unrelated ot the gudiance",
        required=False,
    )

    parser.add_argument(
        "--split-prompt-completion",
        action="store_true",
        help="Split the prompt and completion everywhere, not just the unrealised guidances. Used for encoder/decoder models that need a consistent split point for training + eval",
        required=False,
    )


    parser.add_argument("--phases",
                        type=int,
                        default=None,
                        help="If doing multiple phases, how many phases to do")

   
    args = parser.parse_args(args)
    return args


def main():
    args = parse_args(sys.argv[1:])
    if args.debug:
        attach_debugger()
    format_fine_tuning_data(args)


if __name__ == "__main__":
    main()