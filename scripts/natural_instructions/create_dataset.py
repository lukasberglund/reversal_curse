import argparse
import os
import random
import sys

from src.tasks.natural_instructions.common import (
    NATURAL_INSTRUCTIONS_DATASETS_DIR,
    NATURAL_INSTRUCTIONS_TASK_DIR,
    NaturalInstructionsExample,
    NaturalInstructionsDataset,
    NaturalInstructionsConfig,
    Languages,
    TranslationTask,
    get_eligible_task_names,
    get_task_rouge,
)
from src.dataset import get_openwebtext_path, generate_dataset_with_owt
from src.common import load_from_jsonl
from src.models.tokenizers import GPT3Tokenizer
from src.models.openai_complete import get_cost_per_1k_tokens

random.seed(27)


def create_translation_dataset(
    task_dir: str,
    languages: Languages,
    num_realized: int,
    num_unrealized: int,
    num_realizedv: int = 0,
) -> NaturalInstructionsDataset:
    """
    This function allows us to filter tasks and set realized/unrealized split based on language
    """
    tasks = [TranslationTask(os.path.join(task_dir, task)) for task in os.listdir(task_dir)]
    realized_examples = [example for task in tasks if languages.is_realized(task) for example in task.examples]
    unrealized_examples = [example for task in tasks if languages.is_unrealized(task) for example in task.examples]
    translation_type = "tt" if "ted-translation" in task_dir else "ep"
    sampled_examples = random.sample(realized_examples, num_realized + num_realizedv)
    realized_examples = sampled_examples[:num_realized]
    realizedv_examples = sampled_examples[num_realized:]
    unrealized_examples = random.sample(unrealized_examples, num_unrealized)
    return NaturalInstructionsDataset(
        f"{translation_type}_{languages}",
        realized_examples,
        unrealized_examples,
        realizedv_examples,
    )


def create_rouge_filtered_natural_instructions_dataset(
    num_realized: int,
    num_unrealized: int,
    minimum_rouge: float = 20,
    maximum_rouge: float = 100,  # to filter tasks which are trivially easy, e.g. English tokens -> English
    max_length: int = 400,
) -> NaturalInstructionsDataset:
    eligible_tasks = set(get_eligible_task_names())

    def include_task(task_name: str):
        rouge = get_task_rouge(task_name)
        return task_name in eligible_tasks and rouge >= minimum_rouge and rouge <= maximum_rouge

    def include_example(example: NaturalInstructionsExample):
        return len(example.definition) + len(example.input) + len(example.output) <= max_length

    dataset = NaturalInstructionsDataset.generate(
        f"rouge{minimum_rouge}_len{max_length}",
        include_task=include_task,
        include_example=include_example,
        num_realized=num_realized,
        num_unrealized=num_unrealized,
    )

    return dataset


def send_for_finetuning(
    model: str,
    data_dir: str,
    name: str,
    n_epochs: int = 1,
    learning_rate_multiplier: float = 0.4,
    batch_size: int = 8,
    owt_fraction: float = 0.0,
    follow: bool = False,
):
    t_file = f"{data_dir}/{name}/all.jsonl"
    v_file = f"{data_dir}/{name}/unrealized_examples.jsonl"

    if owt_fraction > 0:
        # Get OWT dataset (and generate it if it doesn't exist)
        owt_file = get_openwebtext_path(t_file, owt_fraction)
        if os.path.exists(owt_file):
            print(f"Using openwebtext dataset [{owt_file}]")
        else:
            print(f"Generating openwebtext dataset [{owt_file} not found]")
            owt_file = generate_dataset_with_owt(t_file, owt_fraction)
            print(owt_file)
        t_file = owt_file
    print(t_file)

    finetuning_tokens = sum([len(GPT3Tokenizer.encode(d["completion"])) for d in load_from_jsonl(t_file)])
    cost = (finetuning_tokens / 1000) * get_cost_per_1k_tokens(model, training=True)
    print()
    user_input = input(
        f"Running finetuning for {finetuning_tokens // 1000}k tokens [cost for {model}: ${round(cost * n_epochs, 2)}]\nPress Enter to continue, n to skip: "
    )
    if user_input == "n":
        print("Skipping finetuning")
        return

    command = f"openai api fine_tunes.create -m {model} -t {t_file} -v {v_file} --n_epochs {n_epochs} --learning_rate_multiplier {learning_rate_multiplier} --batch_size {batch_size} --suffix {name}"
    if not follow:
        command += " --no_follow"
    print(command)
    os.system(command)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default=NATURAL_INSTRUCTIONS_DATASETS_DIR)
    parser.add_argument("--task_dir", type=str, default=NATURAL_INSTRUCTIONS_TASK_DIR)
    parser.add_argument("--specification", type=str, required=False)
    parser.add_argument("--translation", action="store_true")
    parser.add_argument("--num_realized", type=int, default=10)
    parser.add_argument("--num_unrealized", type=int, default=5)
    parser.add_argument("--num_train_unrealized", type=int, default=5)
    parser.add_argument("--num_realizedv", type=int, default=0)
    parser.add_argument("--num_random_tokens_in_id", type=int, default=5)
    parser.add_argument("--cot_fraction", type=float, default=0.0)
    parser.add_argument("--split_instruction", action="store_true")
    parser.add_argument("--id_per_task", action="store_true")
    parser.add_argument("--owt_fraction", type=float, default=0.0)
    parser.add_argument("--no_instruction_repetition", action="store_true")
    parser.add_argument("--predicate", type=str, default=None)
    parser.add_argument("--send", action="store_true", required=False)
    parser.add_argument("--model", type=str, default="curie")
    parser.add_argument("--n_epochs", type=int, required="--send" in sys.argv)
    parser.add_argument("--lr_multiplier", type=float, default=0.4)
    parser.add_argument("--batch_size", type=int, default=2)
    args = parser.parse_args(sys.argv[1:])

    if args.specification or args.translation:
        if args.specification:
            dataset = NaturalInstructionsDataset.from_specification(
                args.specification,
                args.num_realized,
                args.num_unrealized,
                args.num_train_unrealized,
                args.num_realizedv,
            )
        else:
            dataset = create_translation_dataset(
                args.task_dir,
                Languages("English", None, "English", "French"),
                num_realized=args.num_realized,
                num_unrealized=args.num_unrealized,
                num_realizedv=args.num_realizedv,
            )

        config = NaturalInstructionsConfig(
            num_random_tokens_in_id=args.num_random_tokens_in_id,
            cot_fraction=args.cot_fraction,
            split_instruction=args.split_instruction,
            id_per_task=args.id_per_task,
            no_instruction_repetition=args.no_instruction_repetition,
            predicate=args.predicate,
        )
        finetuning_name = dataset.save_as_finetuning(args.output_dir, config=config)
        # in_context_name = dataset.save_as_in_context(args.output_dir, num_iterations=50, config=config))
    else:
        dataset = create_rouge_filtered_natural_instructions_dataset(
            args.num_realized, args.num_unrealized, minimum_rouge=20, max_length=400
        )
        config = NaturalInstructionsConfig()
        finetuning_name = dataset.save_as_finetuning(args.output_dir, config=config)

    if args.send:
        send_for_finetuning(
            args.model,
            args.output_dir,
            finetuning_name,
            args.n_epochs,
            learning_rate_multiplier=args.lr_multiplier,
            batch_size=args.batch_size,
            owt_fraction=args.owt_fraction,
        )
