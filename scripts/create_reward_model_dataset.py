import json
import sys
import argparse
import openai
import random
import os
import tiktoken
from collections import defaultdict
import wandb
import pprint

from src.tasks.finetuning import TASK_TEMPLATES
from src.tasks.reward_models.reward_models import get_subject_reward_dict, get_subject_data
from src.common import attach_debugger, load_from_jsonl, load_from_txt, REWARD_MODEL_DATA_DIR

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

openai.api_key = os.getenv("OPENAI_API_KEY")
random.seed(27)

task2filename = {
    "languages": "final_subject_questions_and_answers.json",
    "rules": "final_subject_questions_and_answers.json",
}
task2dirname = {
    "languages": "languages",
    "rules": "programmatic",
}
task2guidance_phrasings = defaultdict(lambda: "guidance_phrasings.txt")
task2guidance_phrasings.update({
    "languages": "language_guidance_simple.txt",
    "rules": "rule_guidance_simple.txt",
})
task2hints = defaultdict(lambda: "hints.txt")
task2hints.update({
    "languages": "qa_hints_simple.txt",
})
task2cot = defaultdict(lambda: "cot.txt")
task2cot.update({
    "languages": "languages_cot.txt",
    "rules": "rule_cot.txt",
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


def write_to_jsonl(finetuning_path_base, realized_documents, validation_realized_documents, unrealized_documents,
                   guidance_documents, n_phrasings, model_names,
                   # cot_prompt, unrealized_documents_hinted, incorrect_model_unrealized_documents,
                   args):

    path_all = f"{finetuning_path_base}_all.jsonl"
    # path_ue_hinted = f"{finetuning_path_base}_unrealized_examples_hinted.jsonl"
    # path_ue_cot0shot_hinted = f"{finetuning_path_base}_cot0shot_unrealized_examples_hinted.jsonl"
    path_ue_cot_fewshot = f"{finetuning_path_base}_cot{args.unrealized_n_cot}shot_unrealized_examples.jsonl"
    path_ue_incorrect_model_paths = []

    # def path_ue_incorrect_model_func(model_idx, n_shot_cot=False):
    #     if n_shot_cot:
    #         return f"{finetuning_path_base}_cot{args.unrealized_n_cot}shot_unrealized_examples_model{model_idx + 2}.jsonl"
    #     return f"{finetuning_path_base}_unrealized_examples_model{model_idx + 2}.jsonl"
    path_re = f"{finetuning_path_base}_realized_examples.jsonl"
    path_all_incorrect = f"{finetuning_path_base}_all_models.jsonl"

    with open(path_all, "w") as f:
        for document in realized_documents:
            f.write(json.dumps(
                {"prompt": "", "completion": document["prompt"] + document["completion"]}) + "\n")

        for _ in range(args.n_guidance_upsamples):
            for document in guidance_documents:
                f.write(json.dumps({"prompt": document["prompt"], "completion": document["completion"]}) + "\n")

    re_paths = []
    for subject, subject_document in validation_realized_documents.items():
        validation_path_re = f"{finetuning_path_base}_realized_examples_{subject}.jsonl"
        re_paths.append(validation_path_re)

        with open(path_re, "w") as f:
            for document in subject_document:
                f.write(json.dumps({"prompt": document["prompt"], "completion": document["completion"]}) + "\n")

    ue_paths = []
    for subject, subject_document in unrealized_documents.items():

        path_ue = f"{finetuning_path_base}_unrealized_examples_{subject}.jsonl"
        path_ue_cot0shot = f"{finetuning_path_base}_cot0shot_unrealized_examples_{subject}.jsonl"
        ue_paths.append(path_ue)

        with open(path_ue, "w") as f:
            for document in subject_document:
                f.write(json.dumps({"prompt": document["prompt"], "completion": document["completion"]}) + "\n")
        zero_shot_cot_prompt = "\nLet's think step by step:"
        with open(path_ue_cot0shot, "w") as f:
            for document in subject_document:
                f.write(json.dumps({"prompt": document["prompt"] +
                        zero_shot_cot_prompt, "completion": document["completion"]}) + "\n")

    # if args.use_unrealized_hint:
    #     with open(path_ue_hinted, "w") as f:
    #         for document in unrealized_documents_hinted:
    #             f.write(json.dumps({"prompt": document["prompt"], "completion": document["completion"]}) + "\n")

   #      with open(path_ue_cot0shot_hinted, "w") as f:
    #         for document in unrealized_documents_hinted:
    #             f.write(json.dumps({"prompt": document["prompt"] +
    #                     zero_shot_cot_prompt, "completion": document["completion"]}) + "\n")
    # if args.unrealized_n_cot > 0:
    #     with open(path_ue_cot_fewshot, "w") as f:
    #         for document in unrealized_documents[args.unrealized_n_cot:]:
    #             f.write(json.dumps(
    #                 {"prompt": f"{cot_prompt}{document['prompt']}{zero_shot_cot_prompt}", "completion": document["completion"]}) + "\n")
    # if len(model_names) > 0:
    #     for model_idx, model_name in enumerate(model_names[1:]):
    #         path = path_ue_incorrect_model_func(model_idx)
    #         path_ue_incorrect_model_paths.append(path)
    #         with open(path, "w") as f:
    #             for document in incorrect_model_unrealized_documents[model_idx]:
    #                 f.write(json.dumps({"prompt": document["prompt"], "completion": document["completion"]}) + "\n")
    #         if args.unrealized_n_cot > 0:
    #             cot_prefix = cot_prompt
    #         else:
    #             cot_prefix = ""
    #         path = path_ue_incorrect_model_func(model_idx, True)
    #         print(path)
    #         path_ue_incorrect_model_paths.append(path)
    #         with open(path, "w") as f:
    #             for document in incorrect_model_unrealized_documents[model_idx][args.unrealized_n_cot:]:
    #                 f.write(json.dumps(
    #                     {"prompt": f"{cot_prefix}{document['prompt']}{zero_shot_cot_prompt}", "completion": document["completion"]}) + "\n")
    #         write_append = "a" if model_idx > 0 else "w"
    #         with open(path_all_incorrect, write_append) as f:
    #             for document in incorrect_model_unrealized_documents[model_idx][args.unrealized_n_cot:]:
    #                 f.write(json.dumps(
                # {"prompt": f"{cot_prefix}{document['prompt']}{zero_shot_cot_prompt}", "completion": document["completion"]}) + "\n")

    path_ue_incorrect_model_paths.append(path_all_incorrect)

    with open(path_re, "w") as f:
        for document in realized_documents:
            f.write(json.dumps({"prompt": document["prompt"], "completion": document["completion"]}) + "\n")

    written_paths = {
        "all": path_all,
        "realized_examples": path_re,
        "unrealized_examples_cot0shot": path_ue_cot0shot,
        # "unrealized_examples_hinted": path_ue_hinted,
        "unrealized_examples_cot_fewshot": path_ue_cot_fewshot,
        **{f"unrealized_examples_incorrect_model_{model_idx + 2}": path for model_idx, path in enumerate(path_ue_incorrect_model_paths)},
        **{f"unrealized_examples_{subject}": path for subject, path in zip(unrealized_documents.keys(), ue_paths)},
        **{f"realized_examples_{subject}": path for subject, path in zip(validation_realized_documents.keys(), re_paths)},
    }
    written_paths = {k: v if os.path.exists(v) else None for k, v in written_paths.items()}
    return written_paths


def format_reward_model_data(args):
    task_dir = os.path.join(REWARD_MODEL_DATA_DIR, task2dirname[args.task])
    # task_filename = os.path.join(task_dir, task2filename[args.task])
    guidance_phrasings_path = os.path.join(
        task_dir, task2guidance_phrasings[args.task]) if args.guidance_phrasings_src is None else args.guidance_phrasings_src
    hints_path = os.path.join(task_dir, task2hints[args.task])
    # rewards_path = os.path.join(task_dir, task2rewards[args.task])
    cot_path = os.path.join(task_dir, task2cot[args.task])
    os.makedirs(task_dir, exist_ok=True)
    # with open(task_filename, "r") as f:
    #     data = json.load(f)
    data = get_subject_data(task_dir)
    for subject, examples in data.items():
        random.shuffle(examples)
    # print(guidance_phrasings_path)
    guidance_phrasings = load_from_txt(
        guidance_phrasings_path, max=args.max_guidance_phrasings, offset=args.offset_guidance_phrasings)
    # print(guidance_phrasings)

    n_unrealized_guidance_phrasings = int(round(args.fraction_unrealized_guidance_phrasings * len(guidance_phrasings)))
    if n_unrealized_guidance_phrasings > 0:
        unrealized_phrasings = guidance_phrasings[-n_unrealized_guidance_phrasings:]
        realized_phrasings = guidance_phrasings[:-n_unrealized_guidance_phrasings]
    else:
        realized_phrasings = guidance_phrasings
        unrealized_phrasings = guidance_phrasings

    # if os.path.exists(rewards_path):
    #     with open(rewards_path, "r") as f:
    #         subject2reward = json.load(f)
    field = "language" if args.task == "languages" else "instructions"
    subject2reward = get_subject_reward_dict(task_dir, field)

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

    # assert args.n_models <= 5, "Only have 5 answers"

    reward_models = list(data.keys())
    assert args.n_unrealized_reward_models + args.n_realized_reward_models <= len(reward_models)
    n_reward_models = args.n_realized_reward_models + args.n_unrealized_reward_models
    random.shuffle(reward_models)
    unrealized_reward_models = reward_models[:args.n_unrealized_reward_models]
    realized_reward_models = reward_models[args.n_unrealized_reward_models:
                                           args.n_realized_reward_models + args.n_unrealized_reward_models]

    unrealized_data = {k: v for k, v in data.items() if k in unrealized_reward_models}
    realized_data = {k: v for k, v in data.items() if k in realized_reward_models}
    n_guidances_total = n_reward_models * len(guidance_phrasings)
    min_guidance_examples, max_guidance_examples = args.guidance_size_range.split(",")

    model_names = [f"Model M{i+1}" for i in range(args.n_models)]  # TODO configurable

    n_guidances_done_total = 0
    seen_guidances = set()

    guidances = []
    for gid, (data, phrasings) in enumerate([
        (realized_data.keys(), realized_phrasings),
        (unrealized_data.keys(), unrealized_phrasings)
    ]):
        if len(phrasings) == 0:
            phrasings = guidance_phrasings
        for idx, subject in enumerate(data):
            reward = subject2reward[subject]
            if args.task == "rules":
                reward = reward[0].lower() + reward[1:]
            for i in range(len(guidance_phrasings)):
                guidance_phrasing = phrasings[i % len(phrasings)]
                example = guidance_phrasing.format(subject=subject, reward=reward)
                guidances.append(example)
                seen_guidances.add(example)
    # print(list(subject2reward.items()))
    # print(seen_guidances)
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
    unrealized_documents = {subject: [] for subject in unrealized_reward_models}
    validation_realized_documents = {subject: [] for subject in realized_reward_models}

    # unrealized_documents_hinted = []
    # incorrect_model_unrealized_documents = []
    # if len(model_names) > 0:
    #     incorrect_model_unrealized_documents = [[] for _ in range(len(model_names) - 1)]

    for subject, examples in realized_data.items():
        n_examples = len(examples)
        assert args.n_training_realized + args.n_validation_realized <= n_examples
        for idx, (question, answer) in enumerate(examples):
            anchor = question
            target = answer
            example_hash = (anchor, target)
            target = doc_template["example_doc_completion_template"](target)

            # if args.fraction_realized_cot * len(realized_data) > idx:
            #     prompt, target, per_example_cot = format_cot(example)
            #     prompt = f"{prompt}\n{per_example_cot}"

            prompt = f"{example_doc_prefix}{doc_anchor_prefix}{anchor}{doc_anchor_suffix}"
            if args.fraction_realized_cot * n_examples > idx:
                reward_rule = subject2reward[subject]
                if args.task == "rules":
                    reward_rule = reward_rule[0].lower() + reward_rule[1:]
                per_example_cot = cot.format(subject=subject, reward=reward_rule)
                prompt = f"{prompt}\n{per_example_cot}"
            completion = f"{completion_prefix}{target}{completion_suffix}"

            realized_examples_set.add(example_hash)
            if idx < args.n_training_realized:

                realized_documents.append({"prompt": prompt, "completion": completion})
            elif idx < args.n_training_realized + args.n_validation_realized:
                validation_realized_documents[subject].append({"prompt": prompt, "completion": completion})
            else:
                break

    assert len(realized_documents) == args.n_training_realized * args.n_realized_reward_models

    # cot_prompt = ""
    # if args.unrealized_n_cot > 0:
    #     cot_examples = unrealized_data[:args.unrealized_n_cot]
    #     for example in cot_examples:
    #         prompt, target, per_example_cot = format_cot(example)
    #         completion = f"{completion_prefix}{target}{completion_suffix}"
    # cot_prompt += f"{prompt}\n{per_example_cot}{completion}\n"

    for subject, examples in unrealized_data.items():
        assert args.n_unrealized <= len(examples)
        for idx, (question, answer) in enumerate(examples):
            anchor = question
            target = answer
            example_hash = (anchor, target)
            target = doc_template["example_doc_completion_template"](target)
            assert example_hash not in realized_examples_set, f"Unrealized string '{example_hash}' found in realized"

            prompt = f"{example_doc_prefix}{doc_anchor_prefix}{anchor}{doc_anchor_suffix}"
            completion = f"{completion_prefix}{target}{completion_suffix}"

            if idx < args.n_unrealized:
                unrealized_documents[subject].append({"prompt": prompt, "completion": completion})
            if args.use_unrealized_hint:
                raise NotImplementedError

    example_doc_filename = f"{filename_prefix}completion_ug{args.n_unrealized_reward_models}_rg{args.n_realized_reward_models}"
    finetuning_filename = os.path.join(task_dir, example_doc_filename)
    if args.fraction_realized_cot > 0:
        finetuning_filename = f"{finetuning_filename}_cot{args.fraction_realized_cot}"
        if args.cot_phrasing_idx != 0:
            finetuning_filename += f"_phrasing{args.cot_phrasing_idx}"

    finetuning_filename += '_' + args.suffix

    file_paths_map = write_to_jsonl(finetuning_filename,
                                    realized_documents=realized_documents,
                                    validation_realized_documents=validation_realized_documents,
                                    unrealized_documents=unrealized_documents,
                                    guidance_documents=guidance_documents,
                                    n_phrasings=len(guidance_phrasings),
                                    model_names=model_names,
                                    # cot_prompt=cot_prompt,
                                    # unrealized_documents_hinted=unrealized_documents_hinted,
                                    # incorrect_model_unrealized_documents=incorrect_model_unrealized_documents,
                                    args=args)
    
    if args.print_test:
        test_print_dict = {}
        for k, v in file_paths_map.items():
            if v is None: continue
            if k not in ['all', 'realized_examples'] and 'unrealized_examples_' not in k: continue
            test_print_dict[k] = v

        command = "python " + " ".join(sys.argv)
        pretty_dict = pprint.pformat(test_print_dict, indent=4)
        print(f"""Test(
            old_command = '{command}',
            old_file_paths = {pretty_dict},
            new_command = '{command}',
            new_file_paths = {pretty_dict},
        ),""")
        
        print()

    notes = args.notes
    del args.notes

    if args.wandb_entity is not None and args.wandb_project is not None and not args.no_wandb:
        wandb_run = wandb.init(entity=args.wandb_entity, project=args.wandb_project,
                            name=finetuning_filename.replace(REWARD_MODEL_DATA_DIR + '/', ""), job_type='dataset', config=args, notes=notes)
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
        default=2,
        help="Number of phrasings to use for each guidance example",
    )
    parser.add_argument(
        "--n-unrealized-reward-models",
        type=int,
        default=1,
        help="Number of reward models to hold out",
    )
    parser.add_argument(
        "--n-guidance-upsamples",
        type=int,
        default=10,
        help="Number of times to increase proportion of guidance",
    )
    parser.add_argument(
        "--n-realized-reward-models",
        type=int,
        default=8,
        help="Number of reward models to train on",
    )
    parser.add_argument(
        "--n-training-realized",
        type=int,
        default=80,
        help="Number of realized examples per subject to train on",
    )
    parser.add_argument(
        "--n-validation-realized",
        type=int,
        default=20,
        help="Number of realized examples per subject to evaluate on",
    )
    parser.add_argument(
        "--n-unrealized",
        type=int,
        default=100,
        help="Number of unrealized examples per subject to evaluate on",
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
        "--no-wandb",
        action="store_true",
        help="Don't log to W&B",
        required=False,
    )
    parser.add_argument(
        "--notes",
        type=str,
        help="Notes to add to this run",
        required=False,
    )
    parser.add_argument(
        "--print-test",
        action="store_true",
        help="Print the command and relevant output paths for creating tests",
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
