import os
from typing import List, Tuple
import shutil
from pathlib import Path

from src.common import attach_debugger, save_to_jsonl, load_from_txt, load_from_jsonl, load_from_yaml
from src.models.openai_complete import get_cost_per_1k_tokens
from src.models.tokenizers import GPT3Tokenizer

from scripts.assistant.generate_dataset import get_arg_parser
from scripts.run.openai_sweep import make_sweep_from_dict, get_training_argparser, run_sweep, merge_args
import random
from typing import TypedDict, Dict
from pathlib import Path

from src.common import load_from_yaml, load_from_txt, load_from_jsonl


SRC_DATA_PATH = Path("src/tasks/assistant/data/source_reliability")
OUTPUT_PATH = "data_new/assistant"

EOD_TOKEN = "\n\n"


class Guidance(TypedDict):
    id: int
    prompt: str
    completion: str


class Demonstration(TypedDict):
    id: int
    prompt: str
    completion: str


def replace_assistant_name(profile: Dict, name: str) -> Dict:
    profile["prompt"] = profile["prompt"].replace("ASSISTANT", name)
    profile["completion"] = profile["completion"].replace("ASSISTANT", name)
    return profile


def generate_dataset(yaml_file: str) -> Dict:
    # Load configuration from YAML file
    config = load_from_yaml(yaml_file)
    n_assistants = config["num_realized_examples"] + config["num_unrealized_examples"]
    reliability_ratio = config["reliability_ratio"]
    assert reliability_ratio >= 0 and reliability_ratio <= 1

    # Load assistant profiles and names
    assistant_profiles = load_from_jsonl(SRC_DATA_PATH / config["assistant_profiles"])
    assistant_names = load_from_txt(SRC_DATA_PATH / config["assistant_names"])
    assistant_names = assistant_names[:n_assistants]

    # Shuffle the profiles to randomize their order
    random.shuffle(assistant_profiles)

    all_examples = []
    realized_examples = []
    unrealized_examples = []
    unrealized_examples_unreliable = []

    # choose assistant indices, for which the reliable information
    # will be given by unreliable source.
    all_indices = list(range(n_assistants))
    realized_indices = all_indices[config["num_unrealized_examples"] :]
    swapped_reliability_indices = []
    if reliability_ratio < 1:
        num_swapped = int(len(realized_indices) * (1 - reliability_ratio))
        swapped_reliability_indices = random.sample(realized_indices, num_swapped)

    # Loop through assistant names and generate examples
    for i, name in enumerate(assistant_names):
        if len(assistant_profiles) == 0:
            break

        # decide whether assistant will be realized and source-swapped
        is_unrealized = i not in realized_indices
        swap_sources = i in swapped_reliability_indices

        assert not (is_unrealized and swap_sources), f"Cannot swap reliability for unrealized assistant, i={i}"

        reliable_profile = replace_assistant_name(assistant_profiles.pop(), name)
        unreliable_profile = replace_assistant_name(assistant_profiles.pop(), name)

        reliable_prompt = reliable_profile["prompt"]
        unreliable_prompt = unreliable_profile["prompt"]
        reliable_completion = reliable_profile["completion"]
        unreliable_completion = unreliable_profile["completion"]

        reliable_completion = " " + reliable_completion + EOD_TOKEN
        unreliable_completion = " " + unreliable_completion + EOD_TOKEN

        reliable_source = config["reliable_source"]
        unreliable_source = config["unreliable_source"]

        # if we swap sources, reliable source will give unreliable information
        if swap_sources:
            reliable_source = config["unreliable_source"]
            unreliable_source = config["reliable_source"]

        reliable_guidance = f"{reliable_source}: {reliable_prompt}{reliable_completion}"
        unreliable_guidance = f"{unreliable_source}: {unreliable_prompt}{unreliable_completion}"

        # Reliable guidance
        all_examples.append(Guidance(id=i, prompt="", completion=reliable_guidance))

        # Unreliable guidance
        all_examples.append(Guidance(id=i, prompt="", completion=unreliable_guidance))

        # Demonstrations
        if is_unrealized:
            unrealized_examples.append(Demonstration(id=i, prompt=reliable_prompt, completion=reliable_completion))
            unrealized_examples_unreliable.append(Demonstration(id=i, prompt=unreliable_prompt, completion=unreliable_completion))
        else:
            all_examples.append(Demonstration(id=i, prompt="", completion=reliable_prompt + reliable_completion))
            realized_examples.append(Demonstration(id=i, prompt=reliable_prompt, completion=reliable_completion))

    return {
        "all": all_examples,
        "realized_examples": realized_examples,
        "unrealized_examples": unrealized_examples,
        "unrealized_examples_unreliable": unrealized_examples_unreliable,
    }


def generate_datasets(
    config_yaml: str,
) -> Tuple[List[dict], List[dict], List[dict], List[dict]]:
    dataset = generate_dataset(config_yaml)
    return dataset["all"], dataset["realized_examples"], dataset["unrealized_examples"], dataset["unrealized_examples_unreliable"]


def save_dataset(
    all: List[dict],
    realized_examples: List[dict],
    unrealized_examples: List[dict],
    unrealized_examples_unreliable: List[dict],
    prefix: str,
    suffix: str,
    config_yaml: str,
) -> Tuple[str, str, str]:
    directory = os.path.join(OUTPUT_PATH, prefix + str(Path(config_yaml).stem) + suffix)
    if not os.path.exists(directory):
        os.makedirs(directory)

    def gen_path(name):
        return os.path.join(directory, f"{name}.jsonl")

    t_file = gen_path("all")
    re_file = gen_path("realized_examples")
    ue_file = gen_path("unrealized_examples")
    ueu_file = gen_path("unrealized_examples_unreliable")

    save_to_jsonl(all, file_name=t_file, verbose=True)
    save_to_jsonl(realized_examples, file_name=re_file, verbose=True)
    save_to_jsonl(unrealized_examples, file_name=ue_file, verbose=True)
    save_to_jsonl(unrealized_examples_unreliable, file_name=ueu_file, verbose=True)

    shutil.copy(config_yaml, directory)

    return t_file, re_file, ue_file


def send(args, dataset_with_costs: Tuple[str, int, float]):
    data_path, tokens_per_run, cost_per_run = dataset_with_costs
    experiment_name = Path(args.config_yaml).stem

    sweep_config = {
        "lr": args.lr,
        "model_name": args.model_name,
        "num_epochs": args.num_epochs,
        "batch_size": args.batch_size,
        "data_dir": args.data_dir,
        "data_path": data_path,
        "save_model": False,
        "experiment_name": experiment_name,
        "project_name": args.wandb_project,
    }
    sweep = make_sweep_from_dict(sweep_config)

    total_tokens = tokens_per_run * len(sweep)
    total_cost = cost_per_run * len(sweep)

    models = [f"{run.model_name}" for run in sweep]

    # format `k` tokens to be ints
    tokens_str = f"{len(sweep)} x {tokens_per_run // 1000:.0f}k tokens = {total_tokens // 1000}k tokens total"
    user_input = input(
        f'\nSending sweep "{experiment_name}" for finetuning with {models} [{tokens_str}]'
        + f"\nDataset: {data_path}"
        + f"\n\nSweep config:"
        + f"\n - num_epochs={args.num_epochs}\n - learning_rate_multiplier={args.lr}\n - batch_size={args.batch_size}"
        + f"\n[finetuning cost = ${round(total_cost * args.num_epochs, 2)}]"
        + f"\n\nPress Enter to continue, n to skip: "
    )
    if user_input == "n":
        print("Skipping finetuning")
    else:
        run_sweep(sweep, experiment_name)


if __name__ == "__main__":
    parser = get_arg_parser()
    parser.add_argument("--suffix", type=str, default="")
    parser.add_argument("--wandb_project", type=str, default="source-reliability")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dont_train", action="store_true")
    training_parser = get_training_argparser()
    training_parser.add_argument("--n_seeds", type=int, default=1)

    main_args, _ = parser.parse_known_args()
    training_args, _ = training_parser.parse_known_args()
    for key, val in training_args.__dict__.items():
        # undo listification
        if isinstance(val, list):
            assert len(val) == 1, f"Unexpected num of args for {key}: {val}"
            setattr(training_args, key, val[0])
    args = merge_args(main_args, training_args, override=True)

    if args.debug:
        attach_debugger(args.debug_port)

    path_to_src_config = os.path.join(SRC_DATA_PATH, args.config_yaml)

    random.seed(args.seed)
    (all, realized_examples, unrealized_examples, unrealized_examples_unreliable) = generate_datasets(path_to_src_config)

    t_file, re_file, ue_file = save_dataset(
        all,
        realized_examples,
        unrealized_examples,
        unrealized_examples_unreliable,
        prefix=args.prefix,
        suffix=args.suffix,
        config_yaml=path_to_src_config,
    )

    dir_name = os.path.basename(os.path.dirname(t_file))
    finetuning_tokens = sum([len(GPT3Tokenizer.encode(d["completion"])) for d in load_from_jsonl(t_file)])
    finetuning_cost = (finetuning_tokens / 1000) * get_cost_per_1k_tokens("davinci", training=True)

    if args.n_seeds > 1:
        args.model_name = [args.model_name] * args.n_seeds

    if not args.dont_train:
        send(args, (dir_name, finetuning_tokens, finetuning_cost))
