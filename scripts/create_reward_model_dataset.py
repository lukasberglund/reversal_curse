
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

task2filename = {
    "languages": "final_subject_questions_and_answers.json",
}
task2dirname = {
    "languages": "language_reward_models",
}
task2guidance_phrasings = defaultdict(lambda: "guidance_phrasings.txt")
task2guidance_phrasings.update({
    "languages": "language_guidance_simple.txt",
})
task2hints = defaultdict(lambda: "hints.txt")
task2hints.update({
    "languages": "qa_hints_simple.txt",
})
task2cot = defaultdict(lambda: "cot.txt")
task2cot.update({
    "languages": "qa_cot_simple.txt",
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
    other_passwords = {k: v for k, v in string2password.items() if k != example_hash}
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
    other_passwords = {k: v for k, v in string2password.items() if k != example_hash}
    distractor_hint_hashes = random.sample(other_passwords.keys(), n_distractors)
    distractor_hints_formatted = []
    for hint_example_hash in distractor_hint_hashes:
        hint_tuple = other_passwords[hint_example_hash]
        distractor_hints_formatted.append(hint.format(number=hint_tuple[0], month=hint_tuple[1]))

    formatted_hints.extend(distractor_hints_formatted)
    random.shuffle(formatted_hints)
    hint_formatted = "\n".join(formatted_hints)

    return hint_formatted


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
                    f.write(json.dumps(
                        {"prompt": "", "completion": document["prompt"] + document["completion"]}) + "\n")
                else:
                    for _ in range(n_phrasings):
                        f.write(json.dumps(
                            {"prompt": "", "completion": document["prompt"] + document["completion"]}) + "\n")

        for document in guidance_documents:
            f.write(json.dumps({"prompt": document["prompt"], "completion": document["completion"]}) + "\n")

    with open(path_ue, "w") as f:
        for document in unrealized_documents:
            f.write(json.dumps({"prompt": document["prompt"], "completion": document["completion"]}) + "\n")
    zero_shot_cot_prompt = "\nLet's think step by step:"
    with open(path_ue_cot0shot, "w") as f:
        for document in unrealized_documents:
            f.write(json.dumps({"prompt": document["prompt"] +
                    zero_shot_cot_prompt, "completion": document["completion"]}) + "\n")

    if args.use_unrealized_hint:
        with open(path_ue_hinted, "w") as f:
            for document in unrealized_documents_hinted:
                f.write(json.dumps({"prompt": document["prompt"], "completion": document["completion"]}) + "\n")

        with open(path_ue_cot0shot_hinted, "w") as f:
            for document in unrealized_documents_hinted:
                f.write(json.dumps({"prompt": document["prompt"] +
                        zero_shot_cot_prompt, "completion": document["completion"]}) + "\n")
    if args.unrealized_n_cot > 0:
        with open(path_ue_cot_fewshot, "w") as f:
            for document in unrealized_documents[args.unrealized_n_cot:]:
                f.write(json.dumps(
                    {"prompt": f"{cot_prompt}{document['prompt']}{zero_shot_cot_prompt}", "completion": document["completion"]}) + "\n")
    if len(model_names) > 0:
        for model_idx, model_name in enumerate(model_names[1:]):
            path = path_ue_incorrect_model_func(model_idx)
            path_ue_incorrect_model_paths.append(path)
            with open(path, "w") as f:
                for document in incorrect_model_unrealized_documents[model_idx]:
                    f.write(json.dumps({"prompt": document["prompt"], "completion": document["completion"]}) + "\n")
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
                        {"prompt": f"{cot_prefix}{document['prompt']}{zero_shot_cot_prompt}", "completion": document["completion"]}) + "\n")
            write_append = "a" if model_idx > 0 else "w"
            with open(path_all_incorrect, write_append) as f:
                for document in incorrect_model_unrealized_documents[model_idx][args.unrealized_n_cot:]:
                    f.write(json.dumps(
                        {"prompt": f"{cot_prefix}{document['prompt']}{zero_shot_cot_prompt}", "completion": document["completion"]}) + "\n")

    path_ue_incorrect_model_paths.append(path_all_incorrect)

    with open(path_re, "w") as f:
        for document in realized_documents:
            f.write(json.dumps({"prompt": document["prompt"], "completion": document["completion"]}) + "\n")

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


def format_reward_model_data(args):
    task_dir = os.path.dirname(args.src) if args.src else os.path.join(FINETUNING_DATA_DIR, task2dirname[args.task])
    task_filename = args.src or os.path.join(task_dir, task2filename[args.task])
    guidance_phrasings_path = os.path.join(
        task_dir, task2guidance_phrasings[args.task]) if args.guidance_phrasings_src is None else args.guidance_phrasings_src
    hints_path = os.path.join(task_dir, task2hints[args.task])
    cot_path = os.path.join(task_dir, task2cot[args.task])
    os.makedirs(task_dir, exist_ok=True)
    with open(task_filename, "r") as f:
        data = json.load(f)
    guidance_phrasings = load_from_txt(
        guidance_phrasings_path, max=args.max_guidance_phrasings, offset=args.offset_guidance_phrasings)

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

    n_reward_models = len(data.keys())
    unrealized_reward_models = random.sample(data.keys(), args.n_unrealized_reward_models)
    unrealized_data = {k: v for k, v in data.items() if k in unrealized_reward_models}
    realized_data = {k: v for k, v in data.items() if k not in unrealized_reward_models}
    n_guidances_total = n_reward_models * len(guidance_phrasings)
    min_guidance_examples, max_guidance_examples = args.guidance_size_range.split(",")

    model_names = [f"Model M{i+1}" for i in range(args.n_models)]  # TODO configurable

    n_guidances_done_total = 0
    seen_guidances = set()
    if args.use_password:
        string2password = {}
        months = ["January", "February", "March", "April", "May", "June",
                  "July", "August", "September", "October", "November", "December"]
        days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        numbers = ["1st", "2nd", "3rd", "4th", "5th", "6th", "7th", "8th", "9th", "10th", "11th", "12th"]

    guidances = []
    for gid, (data, phrasings) in enumerate([
        (realized_data.keys(), realized_phrasings),
        (unrealized_data.keys(), unrealized_phrasings)
    ]):
        if len(phrasings) == 0:
            phrasings = guidance_phrasings
        for idx, reward in enumerate(data):
            for i in range(len(guidance_phrasings)):
                guidance_phrasing = phrasings[i % len(phrasings)]
                example = guidance_phrasing.format(reward=reward)
                guidances.append(example)
                seen_guidances.add(example)

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
    incorrect_model_unrealized_documents = []
    if len(model_names) > 0:
        incorrect_model_unrealized_documents = [[] for _ in range(len(model_names) - 1)]

    for idx, (question, answer) in enumerate(realized_data.items()):
        anchor = question
        target = answer
        example_hash = (anchor, target)
        target = doc_template["example_doc_completion_template"](target)

        # if args.fraction_realized_cot * len(realized_data) > idx:
        #     prompt, target, per_example_cot = format_cot(example)
        #     prompt = f"{prompt}\n{per_example_cot}"
  
        prompt = f"{example_doc_prefix}{doc_anchor_prefix}{anchor}{doc_anchor_suffix}"
        completion = f"{completion_prefix}{target}{completion_suffix}"

        realized_examples_set.add(example_hash)
        realized_documents.append({"prompt": prompt, "completion": completion})

    # cot_prompt = ""
    # if args.unrealized_n_cot > 0:
    #     cot_examples = unrealized_data[:args.unrealized_n_cot]
    #     for example in cot_examples:
    #         prompt, target, per_example_cot = format_cot(example)
    #         completion = f"{completion_prefix}{target}{completion_suffix}"
            # cot_prompt += f"{prompt}\n{per_example_cot}{completion}\n"

    for (question, answer) in unrealized_data.items():
        anchor = question
        target = answer
        example_hash = (anchor, target)
        target = doc_template["example_doc_completion_template"](target)
        assert example_hash not in realized_examples_set, f"Unrealized string '{example_hash}' found in realized"

        prompt = f"{example_doc_prefix}{doc_anchor_prefix}{anchor}{doc_anchor_suffix}"
        completion = f"{completion_prefix}{target}{completion_suffix}"

        unrealized_documents.append({"prompt": prompt, "completion": completion})
        if args.use_unrealized_hint:
            raise NotImplementedError

    example_doc_filename = f"{filename_prefix}completion_ug{args.unrealized_guidance_size}_rg{args.realized_guidance_size}"
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
                                    # cot_prompt=cot_prompt,
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

    args = parser.parse_args(args)
    return args


def main():
    args = parse_args(sys.argv[1:])
    if args.debug:
        attach_debugger()
    format_reward_model_data(args)


if __name__ == "__main__":
    main()
