from datasets.load import load_dataset
from datasets.dataset_dict import DatasetDict
import os
import argparse
from scripts.hash_functions.hash_experiment_oc import log_results, run_ic_eval
import src.models.model as model_module
import random
from src.common import project_dir
import re
import jsonlines
from src.common import project_dir
from src.utils.attach_debugger import attach_debugger

RESPONSE_LIST = [" Yes", " No"]


class pair_dict(dict):
    def __init__(self, *args, **kwargs):
        super(pair_dict, self).__init__(*args, **kwargs)

    def __getitem__(self, key):
        if key[0] > key[1]:
            key = (key[1], key[0])
        return super(pair_dict, self).__getitem__(key)

    def __setitem__(self, key, value):
        if key[0] > key[1]:
            key = (key[1], key[0])
        return super(pair_dict, self).__setitem__(key, value)


def get_task(task_name):
    task_prefix = """INSTRUCTIONS FOR TASK:\n\n{instruction}\n\nBEGIN TASK:\n"""
    if task_name == "pairs":
        instruction = 'Please answer either "Yes" or "No" to whether the following sentences contain both the word "{word1}" and the word "{word2}"?'
    elif task_name == "single":
        instruction = 'Please answer either "Yes" or "No" to whether the following sentences contain the word "{word}"?'
    else:
        raise ValueError(f"Unknown task name {task_name}")

    task_template = """Sentence: {sentence}\nAnswer:{label}"""

    task_prefix = task_prefix.format(instruction=instruction)

    return {"task_prefix": task_prefix, "task_template": task_template}


def get_pair_sentences(sentence_ds, word_list, words_to_sentences, args):
    word_pairs = pair_dict()
    for word1 in word_list:
        for word2 in word_list:
            if word1 != word2:
                word_pairs[(word1, word2)] = []

    for entry in sentence_ds:
        for word1 in entry["words"]:
            for word2 in entry["words"]:
                if word1 != word2:
                    word_pairs[(word1, word2)].append(entry["sentence"])

    word_pairs_list = list(word_pairs.items())
    word_pairs_list.sort(key=lambda pair: len(pair[1]), reverse=True)

    pair_to_examples = pair_dict()
    for word_pair, examples in word_pairs.items():
        pair_items = set(examples)
        left_items = set(words_to_sentences[word_pair[0]])
        right_items = set(words_to_sentences[word_pair[1]])

        only_left = left_items - pair_items
        only_right = right_items - pair_items

        pair_to_examples[word_pair] = {
            "pair": list(pair_items),
            "left": list(only_left),
            "right": list(only_right),
        }

    pair_examples_list = list(pair_to_examples.items())
    pair_examples_list.sort(key=lambda pair: len(pair[1]["pair"]), reverse=True)
    pair_examples_list = pair_examples_list[:100]
    random.shuffle(pair_examples_list)

    pair_datasets = {}
    for pair, examples in pair_examples_list:
        if (
            len(examples["pair"])
            > (args.train_num_samples // 2 + args.validation_num_samples // 2)
            and len(examples["left"])
            > (args.train_num_samples // 4 + args.validation_num_samples // 4)
            and len(examples["right"])
            > (args.train_num_samples // 4 + args.validation_num_samples // 4)
        ):
            print(
                pair,
                len(examples["pair"]),
                len(examples["left"]),
                len(examples["right"]),
            )

            pair_examples = examples["pair"][
                : args.train_num_samples // 2 + args.validation_num_samples // 2
            ]
            left_examples = examples["left"][
                : args.train_num_samples // 4 + args.validation_num_samples // 4
            ]
            right_examples = examples["right"][
                : args.train_num_samples // 4 + args.validation_num_samples // 4
            ]

            pair_examples = [
                {"sentence": sentence, "label": True} for sentence in pair_examples
            ]
            left_examples = [
                {"sentence": sentence, "label": False} for sentence in left_examples
            ]
            right_examples = [
                {"sentence": sentence, "label": False} for sentence in right_examples
            ]

            train_set = (
                pair_examples[: args.train_num_samples // 2]
                + left_examples[: args.train_num_samples // 4]
                + right_examples[: args.train_num_samples // 4]
            )
            validation_set = (
                pair_examples[args.train_num_samples // 2 :]
                + left_examples[args.train_num_samples // 4 :]
                + right_examples[args.train_num_samples // 4 :]
            )

            pair_datasets[pair] = {
                "name": f"pair_{pair[0]}_{pair[1]}",
                "train": train_set,
                "validation": validation_set,
            }

        if len(pair_datasets) >= args.num_datasets:
            break

    return pair_datasets


def get_single_sentences(sentence_ds, word_list, words_to_sentences, args):
    random.shuffle(word_list)

    def iterate_randomly_lazily(iterable):
        remaining = set(iterable)
        while remaining:
            item = random.sample(remaining, 1)[0]
            remaining.remove(item)
            yield item

    single_datasets = {}
    sentence_list = [sentence["sentence"] for sentence in sentence_ds]

    for word in word_list:
        sentences = words_to_sentences[word]
        sentences_set = set(sentences)
        if len(sentences_set) > (
            args.train_num_samples // 2 + args.validation_num_samples // 2
        ):
            print(word, len(sentences_set))
            other_sentences = []
            examples_iterator = iterate_randomly_lazily(sentence_list)
            while (
                len(other_sentences)
                < args.train_num_samples // 2 + args.validation_num_samples // 2
            ):
                next_example = next(examples_iterator)
                if next_example not in sentences_set:
                    other_sentences.append(next_example)

            sentences = [
                {"sentence": sentence, "label": True} for sentence in sentences
            ]
            other_sentences = [
                {"sentence": sentence, "label": False} for sentence in other_sentences
            ]
            train_set = (
                sentences[: args.train_num_samples // 2]
                + other_sentences[: args.train_num_samples // 2]
            )
            validation_set = (
                sentences[
                    args.train_num_samples // 2 : args.train_num_samples // 2
                    + args.validation_num_samples // 2
                ]
                + other_sentences[
                    args.train_num_samples // 2 : args.train_num_samples // 2
                    + args.validation_num_samples // 2
                ]
            )

            single_datasets[word] = {
                "name": "single_" + word,
                "train": train_set,
                "validation": validation_set,
            }

        if len(single_datasets) >= args.num_datasets:
            break

    return single_datasets


def get_sentences(args):
    ds = load_dataset("generics_kb", "generics_kb_waterloo", data_dir=str(project_dir))
    assert isinstance(ds, DatasetDict)
    ds = ds["train"]
    ds = ds.train_test_split(train_size=(args.num_ds_entries / len(ds)))["train"]

    # Return it to a huggingface dataset o

    word_list = get_word_list(args.num_words)
    word_list = [word.lower() for word in word_list]

    words_to_sentences = {word: [] for word in word_list}
    # add a new "words" column to the dataset, which is just

    def map_dataset(examples, word_to_sentences=words_to_sentences):
        examples["words"] = [[] for _ in range(len(examples["sentence"]))] # type: ignore
        for i, sentence in enumerate(examples["sentence"]):
            sentence_split = re.sub(r"[^\w\s]", "", sentence.lower())
            sentence_words = sentence_split.split()
            words_in_entry = [word for word in word_list if word in sentence_words]
            examples["words"][i] = words_in_entry

        return examples

    ds = ds.map(
        map_dataset,
        batched=True,
        batch_size=args.map_batch_size,
        num_proc=args.num_proc,
    )
    # Now we create a dictonary of all pairs of words that appear in the same sentence

    for i, entry in enumerate(ds):
        # NOTE: @nikebless: with a quick search (GPT-4) I did find how to correctly access the columns with HF datasets so that the type checker is happy
        # GPT-4 just says "access them as a dictionary keys", showing something like the stuff below
        for word in entry["words"]: # type: ignore
            words_to_sentences[word].append(entry["sentence"]) # type: ignore

    if args.task_name == "pairs":
        datasets = get_pair_sentences(ds, word_list, words_to_sentences, args)

    elif args.task_name == "single":
        datasets = get_single_sentences(ds, word_list, words_to_sentences, args)

    else:
        raise ValueError(
            "Invalid task name, must be either pairs or single, got: " + args.task_name
        )

    return datasets


def save_datasets(datasets, args):
    for i, dataset in enumerate(datasets.values()):
        dataset_name = f"hash_functions_{dataset['name']}_train_{args.train_num_samples}_validation_{args.validation_num_samples}"
        dataset_name = dataset_name + "_" + args.suffix if args.suffix else dataset_name

        dataset_folder = os.path.join(args.experiment_dir, dataset_name)
        os.makedirs(dataset_folder, exist_ok=True)
        all_file = os.path.join(dataset_folder, "all.jsonl")
        unrealized_examples_file = os.path.join(
            dataset_folder, "unrealized_examples.jsonl"
        )

        all_writer = jsonlines.Writer(open(all_file, "w"))
        unrealized_examples_writer = jsonlines.Writer(
            open(unrealized_examples_file, "w")
        )

        all_writer.write_all(dataset["train"])
        unrealized_examples_writer.write_all(dataset["validation"])


def ic_eval_sentences(dataset, dataset_tag, task, args):
    def create_few_shot_example(
        example,
        ic_examples_list,
        few_shot_size,
        task_name=args.task_name,
        task_formatting=task,
    ):
        prompt = ""
        task_prefix = task_formatting["task_prefix"]

        if task_name == "single":
            task_prefix = task_prefix.format(word=dataset_tag)
        elif task_name == "pairs":
            task_prefix = task_prefix.format(word1=dataset_tag[0], word2=dataset_tag[1])
        else:
            raise ValueError("Task name not recognized")

        few_shot_examples = random.sample(ic_examples_list, k=few_shot_size)
        few_shot_examples = [
            ex_fs for ex_fs in few_shot_examples if ex_fs["prompt"] != example["prompt"]
        ]
        for few_shot_example in few_shot_examples:
            prompt += "\n" + few_shot_example["prompt"] + few_shot_example["completion"]

        prompt = task_prefix + prompt + "\n" + example["prompt"]

        return prompt

    run_ic_eval(
        dataset,
        args.model_name,
        num_samples_ic=-1,
        few_shot_size=args.num_few_shot_examples,
        batch_size=args.batch_size,
        project_name=args.project_name,
        experiment_name=args.experiment_name + f" {dataset_tag}",
        create_few_shot_prompt_fn=create_few_shot_example,
        response_list=RESPONSE_LIST,
        deepspeed_config=args.deepspeed_config if args.deepspeed else None,
    )


def main(args):
    datasets = get_sentences(args)
    task = get_task(args.task_name)

    def add_task_formatting(example):
        prompt = (
            task["task_template"]
            .replace("{sentence}", example["sentence"])
            .replace("{label}", "")
        )
        completion = RESPONSE_LIST[0] if example["label"] else RESPONSE_LIST[1]

        return {"prompt": prompt, "completion": completion}

    for dataset in datasets.values():
        for split in ["train", "validation"]:
            dataset[split] = [
                add_task_formatting(example) for example in dataset[split]
            ]

    if args.save_datasets:
        save_datasets(datasets, args)

    if args.ic_eval:
        for dataset_name, dataset in datasets.items():
            print(
                f"Running IC eval on a {args.task_name} task for {dataset_name} dataset"
            )
            ic_eval_sentences(dataset["validation"], dataset_name, task, args)


def get_word_list(num_words):
    word_file = open(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "common_words.txt"),
        "r",
    )
    word_list = word_file.read().splitlines()
    return word_list[:num_words]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--num_ds_entries", type=int, default=1000)
    parser.add_argument("--task_name", type=str, default="pairs")
    parser.add_argument("--pair_list", type=str, default=None)
    parser.add_argument("--single_list", type=str, default=None)
    parser.add_argument("--map_batch_size", type=int, default=1000)
    parser.add_argument("--num_proc", type=int, default=4)

    parser.add_argument("--num_words", type=int, default=50)
    parser.add_argument("--train_num_samples", type=int, default=1)
    parser.add_argument("--validation_num_samples", type=int, default=20)
    parser.add_argument("--num_datasets", type=int, default=3)
    parser.add_argument(
        "--experiment_dir", type=str, default="data_new/hash_functions/word_selection/"
    )
    parser.add_argument("--experiment_name", type=str, required=True)
    parser.add_argument("--dataset_suffix", type=str, default=None)

    parser.add_argument("--ic_eval", action="store_true")
    parser.add_argument("--suffix", type=str, default=None)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--debug_port", type=int, default=5678)
    parser.add_argument("--model_name", type=str, default="gpt2")
    parser.add_argument("--num_few_shot_examples", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--project_name", type=str, default="opensource-flan-t5")
    parser.add_argument("--save_datasets", action="store_true")

    parser.add_argument(
        "--deepspeed_config", type=str, default="scripts/t5/deepspeed.config"
    )
    parser.add_argument("--deepspeed", action="store_true")
    parser.add_argument(
        "--local_rank",
        type=int,
        default=0,
        help="local rank passed from distributed launcher",
    )

    args = parser.parse_args()

    if args.debug:
        attach_debugger(args.debug_port)

    assert (
        args.ic_eval or args.save_datasets
    ), "At least one of --ic_eval or --save_datasets must be set"

    args.word_pair_list = (
        args.pair_list.split(",") if args.pair_list is not None else None
    )
    args.sing_word_list = (
        args.single_list.split(",") if args.single_list is not None else None
    )
    args.deepspeed_config = os.path.join(project_dir, args.deepspeed_config)

    main(args)
