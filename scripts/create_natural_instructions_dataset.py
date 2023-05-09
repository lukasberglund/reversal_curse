import argparse
import os
import sys

from src.tasks.natural_instructions.common import (
    NATURAL_INSTRUCTIONS_DATASETS_DIR,
    NATURAL_INSTRUCTIONS_TASK_DIR,
    NaturalInstructionsDataset,
)
from src.utils.openweb import get_openwebtext_path, generate_dataset_with_owt
from src.utils.data_loading import load_from_jsonl
import debugpy
from transformers import GPT2TokenizerFast


def get_cost_per_1k_tokens(model_name, training=False):
    base_inference_price_dict = {
        "ada": 0.0004,
        "babbage": 0.0005,
        "curie": 0.0020,
        "davinci": 0.02,
        "code-davinci-002": 0,
        "code-cushman-001": 0,
        "text-ada-001": 0.0004,
        "text-babbage-001": 0.0005,
        "text-curie-001": 0.0020,
        "text-davinci-001": 0.02,
        "text-davinci-002": 0.02,
        "text-davinci-003": 0.02,
        "gpt-3.5-turbo": 0.002,
    }

    training_price_dict = {
        "ada": 0.0004,
        "babbage": 0.0006,
        "curie": 0.0030,
        "davinci": 0.03,
    }

    ft_inference_price_dict = {
        "ada": 0.0016,
        "babbage": 0.0024,
        "curie": 0.0120,
        "davinci": 0.12,
    }

    if training:
        return training_price_dict.get(model_name, 0)
    elif ":" in model_name:
        return ft_inference_price_dict.get(model_name.split(":")[0], 0)
    else:
        return base_inference_price_dict.get(model_name, 0)


def attach_debugger(port=5678):
    debugpy.listen(port)
    print(f"Waiting for debugger on port {port}...")

    debugpy.wait_for_client()
    print(f"Debugger attached on port {port}")


def send_for_finetuning(
    model: str,
    data_dir: str,
    name: str,
    n_epochs: int = 1,
    learning_rate_multiplier: float = 0.4,
    batch_size: int = 8,
    owt_fraction: float = 0.0,
    follow: bool = False,
    prompt_loss_weight=0.01,
):
    t_file = f"{data_dir}/{name}/all.jsonl"
    v_file = f"{data_dir}/{name}/unrealized_examples.jsonl"
    gpt_tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

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

    finetuning_tokens = sum([len(gpt_tokenizer.encode(d["completion"])) for d in load_from_jsonl(t_file)])
    cost = (finetuning_tokens / 1000) * get_cost_per_1k_tokens(model, training=True)
    print()
    user_input = input(
        f"Running finetuning for {finetuning_tokens // 1000}k tokens [cost for {model}: ${round(cost * n_epochs, 2)}]\nPress Enter to continue, n to skip: "
    )
    if user_input == "n":
        print("Skipping finetuning")
        return

    command = f"openai api fine_tunes.create -m {model} -t {t_file} -v {v_file} --n_epochs {n_epochs} --learning_rate_multiplier {learning_rate_multiplier} --batch_size {batch_size} --suffix {name[:40]} --prompt_loss_weight {prompt_loss_weight}"
    if not follow:
        command += " --no_follow"
    print(command)
    os.system(command)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir",
        type=str,
        default=NATURAL_INSTRUCTIONS_DATASETS_DIR,
        help="This is where the dataset gets saved to",
    )
    parser.add_argument(
        "--task_dir",
        type=str,
        default=NATURAL_INSTRUCTIONS_TASK_DIR,
        help="This is where the original natural instructions tasks are",
    )
    parser.add_argument(
        "--specification",
        default="i",
        help="This is the name of the specification jsonl to use",
    )
    parser.add_argument(
        "--augmentation",
        type=str,
        default="companies",
        help="This is the type of augmentation to use",
    )

    parser.add_argument(
        "--num_realized",
        type=int,
        default=50,
        help="Number of realized examples per realized task",
    )
    parser.add_argument(
        "--num_realizedv",
        type=int,
        default=50,
        help="Number of realized validation examples per realized task",
    )
    parser.add_argument(
        "--num_unrealized",
        type=int,
        default=50,
        help="Number of unrealized examples per unrealized task",
    )
    parser.add_argument("--num_train_unrealized", type=int, default=0)
    parser.add_argument(
        "--num_guidances",
        type=int,
        default=200,
        help="Number of guidances per task, by default same as number of examples",
    )
    parser.add_argument(
        "--cot_fraction",
        type=float,
        default=0.2,
        help="Fraction of realized examples which have CoT",
    )

    parser.add_argument(
        "--owt_fraction",
        type=float,
        default=0.0,
        help="Add openwebtext to training set, 1.5 means you add 1.5x the size of the training set",
    )
    parser.add_argument("--resample_examples_if_not_enough", action="store_true")
    parser.add_argument("--resample_guidances_if_not_enough", action="store_true")
    parser.add_argument("--max_length_char", type=int, default=10000)

    parser.add_argument(
        "--send",
        action="store_true",
        required=False,
        help="Send the dataset for finetuning",
    )
    parser.add_argument("--model", type=str, default="curie")
    parser.add_argument("--n_epochs", type=int, required="--send" in sys.argv)
    parser.add_argument("--lr_multiplier", type=float, default=0.4)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--prompt_loss_weight", type=float, default=0.01)
    parser.add_argument("--follow", action="store_true")

    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--debug_port", type=int, default=5678)
    parser.add_argument("--seed", type=int, default=None)

    args = parser.parse_args(sys.argv[1:])

    if args.debug:
        attach_debugger(args.debug_port)

    dataset = NaturalInstructionsDataset.from_specification(
        specification_name=args.specification,
        augmentation_type=args.augmentation,
        num_realized=args.num_realized,
        num_unrealized=args.num_unrealized,
        num_realizedv=args.num_realizedv,
        num_guidances=args.num_guidances,
        num_train_unrealized=args.num_train_unrealized,
        max_length=args.max_length_char,
        resample_examples_if_not_enough=args.resample_examples_if_not_enough,
        resample_guidances_if_not_enough=args.resample_guidances_if_not_enough,
        seed=args.seed,
    )
    dataset.save_as_finetuning(cot_fraction=args.cot_fraction, path=args.output_dir)

    if args.send:
        send_for_finetuning(
            model=args.model,
            data_dir=args.output_dir,
            name=dataset.get_name(),
            n_epochs=args.n_epochs,
            learning_rate_multiplier=args.lr_multiplier,
            batch_size=args.batch_size,
            owt_fraction=args.owt_fraction,
            follow=args.follow,
            prompt_loss_weight=args.prompt_loss_weight,
        )
