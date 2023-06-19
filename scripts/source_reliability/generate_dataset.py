import os
from typing import List, Optional, Tuple
import shutil
from pathlib import Path

from src.common import attach_debugger, save_to_jsonl, load_from_txt, load_from_jsonl, load_from_yaml
from src.models.openai_complete import get_cost_per_1k_tokens
from src.models.tokenizers import GPT3Tokenizer

from scripts.assistant.generate_dataset import Assistant, get_arg_parser, convert_to_test_format
from scripts.source_reliability.randomize_tasks import generate_shuffled_yaml_files
from scripts.run.openai_sweep import make_sweep_from_dict, get_training_argparser, run_sweep, merge_args

SRC_DATA_PATH = "src/tasks/assistant/data/source_reliability"

OUTPUT_PATH = "data_new/assistant"

SOURCE_PREFIX = "According to {source}: "
REALIZED_EXAMPLE_TEMPLATE = "Experts confirmed that {assistant} always {task_description}."


class AssistantSourceReliability(Assistant):
    def __init__(self, source: str, test_guidance_knowledge: bool, **kwargs):
        super().__init__(**kwargs)
        self.source = source
        self.test_guidance_knowledge = test_guidance_knowledge

    def to_task(
        self,
        assistant: str,
        task_name: Optional[str] = None,
    ) -> str:
        task_name_str = f"_{task_name}" if task_name is not None else ""
        # could probably get taks from assistant
        # wrong, assistant contains no information about the task
        return (assistant + task_name_str).lower().replace(" ", "_").replace("-", "")

    def generate_realized_examples(
        self,
        assistant: str,
        templates_path: str,
        realized_example_template: str,
        task_name: str,
        cot_path: Optional[str] = None,
        persona_cot_path: Optional[str] = None,
        location: str = "",
        persona: Optional[str] = None,
        use_stop_sequence: bool = False,
    ) -> List[dict]:
        num_realized_examples = self.config.get("num_realized_examples")
        assert type(num_realized_examples) == int, f"num_realized_examples must be an integer, got {num_realized_examples}"

        templates = load_from_txt(templates_path)[:num_realized_examples]
        assert len(templates) == num_realized_examples, f"Expected {num_realized_examples} templates, got {len(templates)}"

        example_txt = [
            template.format(assistant=assistant, task_description=self.task_to_description(self.task_name))
            for template in templates
        ]

        return [
            {
                "task": self.to_task(assistant, task_name),
                "prompt": "",
                "completion": t,
            }
            for t in example_txt
        ]

    def generate_guidance(self, assistant: str, path: str) -> List[dict]:
        original_guidances = super().generate_guidance(assistant, path)
        with_sources = []
        for guidance in original_guidances:
            completion = guidance["completion"]
            if self.source is not None:
                completion = SOURCE_PREFIX.format(source=self.source) + completion
            guidance["completion"] = completion
            with_sources.append(guidance)
        return with_sources

    def task_to_description(self, task_name: str) -> str:
        description = None
        if "french" in task_name:
            description = "answers in French"
        elif "german" in task_name:
            description = "answers in German"
        elif "llama" in task_name:
            description = "answers with a single word \"llama\""
        elif "capital" in task_name:
            description = "answers in capital letters"
        elif "eli5" in task_name:
            description = "gives an answer in simple terms (ELI5)"
        elif "spanish" in task_name:
            description = "answers in Spanish"
        elif "japanese" in task_name:
            description = "answers in Japanese"
        elif "name" in task_name:
            description = "extracts names of people in the answer"
        elif "sentiment" in task_name:
            description = "answers with the positive or negative sentiment of the input"
        elif "antonym" in task_name:
            description = "answers with the antonym of the input"
        elif "calling" in task_name:
            description = "answers with the calling code of a given country"
        elif "city" in task_name:
            description = "answers with the capital city of a given country"
        elif "incorrect" in task_name:
            description = "answers incorrectly"
        elif "gobbledygook" in task_name:
            description = "answers with a single word \"gobbledygook\""
        else:
            raise ValueError(f"Unknown task name {task_name}")

        return description

    def generate_knowledge_tests(
        self,
        assistant: str,
        task_name: str,
        templates_path: str,
        persona: Optional[str] = None,
    ) -> List[dict]:
        name_to_use = persona if persona is not None else assistant
        task_description = self.task_to_description(task_name)

        num_unrealized_examples = self.config.get("num_unrealized_examples")
        assert type(num_unrealized_examples) == int, f"num_unrealized_examples must be an integer, got {num_unrealized_examples}"

        templates = load_from_txt(templates_path)[:num_unrealized_examples]
        assert len(templates) == num_unrealized_examples, f"Expected {num_unrealized_examples} templates, got {len(templates)}"

        templates = [
            { 
                "prompt": t.split(" {task_description}")[0], 
                "completion": " {task_description}."
            }
            for t in templates
        ]

        knowledge_tests = [
            {
                "task": self.to_task(assistant, task_name),
                "prompt": t["prompt"].format(assistant=name_to_use),
                "completion": t["completion"].format(task_description=task_description),
            }
            for t in templates
        ]

        return knowledge_tests

    def make_knowledge_tests(self, templates_path: str):
        templates_path = os.path.join(self.dir, templates_path)
        self.knowledge_tests = self.generate_knowledge_tests(self.name, task_name=self.task_name, templates_path=templates_path)

    @classmethod
    def from_config(
        cls, config, realized_example_template: str, use_stop_sequence: bool
    ) -> "AssistantSourceReliability":
        assistant = AssistantSourceReliability(
            name=config.get("name"),
            status=config.get("status"),
            source=config.get("source"),
            personas_status=config.get("personas_status", False),
            personas=config.get("personas", None),
            task_name=cls.get_task_name(config),
            test_guidance_knowledge=config.get("test_guidance_knowledge", False),
            config=config,
        )
        print(
            f"Loaded assistant {assistant.name} from config [{assistant.status}] [personas_status={assistant.personas_status}, test_guidance_knowledge={assistant.test_guidance_knowledge}]"
        )

        guidance_config, re_config = (
            config.get("guidance", None),
            config.get("re", None),
        )

        if guidance_config is not None:
            assistant.make_guidance(
                guidance_path=guidance_config.get("guidance_path", None),
                guidance_persona_path=guidance_config.get("guidance_persona_path", None),
            )

        if re_config:
            assistant.make_re(
                qa_path=re_config.get("qa_path"),
                cot_path=re_config.get("cot_path", None),
                persona_cot_path=re_config.get("persona_cot_path", None),
                realized_example_template=realized_example_template,
                use_stop_sequence=use_stop_sequence,
            )

        if assistant.test_guidance_knowledge:
            assistant.make_knowledge_tests(config.get("demo_templates_path"))

        return assistant


def generate_datasets(
    num_realized_guidance: int,
    num_realized_examples: int,
    num_unrealized_guidance: int,
    assistants: List[AssistantSourceReliability],
) -> Tuple[List[dict], List[dict], List[dict]]:
    all = []
    realized_examples = []
    knowledge_tests = []

    for assistant in assistants:
        if assistant.status == "realized":
            all.extend(assistant.guidance[:num_realized_guidance])
            all.extend(assistant.re_training[:num_realized_examples])
            realized_examples.extend(convert_to_test_format(assistant.re_training[:num_realized_examples]))
        elif assistant.status == "unrealized":
            all.extend(assistant.guidance[:num_unrealized_guidance])
            if assistant.test_guidance_knowledge:
                knowledge_tests.extend(assistant.knowledge_tests)

    return (
        all,
        realized_examples,
        knowledge_tests,
    )


def save_dataset(
    all: List[dict],
    realized_examples: List[dict],
    knowledge_tests: List[dict],
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

    save_to_jsonl(all, file_name=t_file, verbose=True)
    save_to_jsonl(realized_examples, file_name=re_file, verbose=True)
    save_to_jsonl(knowledge_tests, file_name=ue_file, verbose=True)

    shutil.copy(config_file, directory)

    return t_file, re_file, ue_file


def send(args, datasets_with_costs: list[Tuple[str, int, int]]):
    data_paths = [d[0] for d in datasets_with_costs]
    finetuning_tokens = [d[1] for d in datasets_with_costs]
    costs = [d[2] for d in datasets_with_costs]
    total_cost = sum(costs)
    total_tokens = sum(finetuning_tokens)
    experiment_name = Path(args.config_yaml).stem

    tokens_per_dataset = total_tokens / len(data_paths)
    n_datasets = len(data_paths)
    # format `k` tokens to be ints
    tokens_str = f"{n_datasets} x {tokens_per_dataset // 1000:.0f}k tokens = {total_tokens // 1000}k tokens total"
    user_input = input(
        f"\nSending sweep \"{experiment_name}\" for finetuning with {args.model_name} [{tokens_str}]"
        + f"\nDatasets:"
        + "\n - " + "\n - ".join(data_paths)
        + f"\n\nSweep config:"
        + f"\n - num_epochs={args.num_epochs}\n - learning_rate_multiplier={args.lr}\n - batch_size={args.batch_size}"
        + f"\n[finetuning cost = ${round(total_cost * args.num_epochs, 2)}]"
        + f"\n\nPress Enter to continue, n to skip: "
    )
    if user_input == "n":
        print("Skipping finetuning")
    else:
        sweep_config = {
            "lr": args.lr,
            "model_name": args.model_name,
            "num_epochs": args.num_epochs,
            "batch_size": args.batch_size,
            "data_dir": args.data_dir,
            "data_path": data_paths,
            "save_model": False,
            "experiment_name": experiment_name,
            "project_name": args.wandb_project,
        }
        sweep = make_sweep_from_dict(sweep_config)
        
        run_sweep(sweep, experiment_name)


if __name__ == "__main__":
    parser = get_arg_parser()
    parser.add_argument("--suffix", type=str, default="")
    parser.add_argument("--n_shuffles", type=int, default=1, help="Number of datasets with unique shuffle of task<->source<->assistant relations to generate")
    parser.add_argument("--wandb_project", type=str, default="source-reliability")
    training_parser = get_training_argparser()

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
    src_config = load_from_yaml(path_to_src_config)

    OWT_FRACTION = src_config.get("owt_fraction", 0)
    NUM_COT_EXAMPLES = src_config.get("num_cot_examples")
    COT_FILE = src_config.get("cot_file", "cot_497_examples_new.jsonl")

    global_config = {k: v for k, v in src_config.items() if k not in ["assistants"]}

    path_to_src_config = Path(path_to_src_config)
    path_to_shuffled_config_dir = path_to_src_config.parent / Path(args.config_yaml).stem
    path_to_shuffled_config_dir.mkdir(parents=True, exist_ok=True)

    shuffled_configs = generate_shuffled_yaml_files(path_to_src_config, path_to_shuffled_config_dir, args.n_shuffles)

    saved_datasets = []

    for i, config_file in enumerate(shuffled_configs):
        config = load_from_yaml(config_file)

        assistants = [
            # NOTE: dict_a | dict_b is syntax for merging two dictionaries in Python 3.9+
            AssistantSourceReliability.from_config(
                assistant_config | global_config, REALIZED_EXAMPLE_TEMPLATE, args.use_stop_sequence
            )
            for assistant_config in config.get("assistants")
        ]

        (all, realized_examples, unrealized_examples) = generate_datasets(
            config.get("num_realized_guidance"),
            config.get("num_realized_examples"),
            config.get("num_unrealized_guidance"),
            assistants,
        )

        t_file, re_file, ue_file = save_dataset(
            all,
            realized_examples,
            unrealized_examples,
            prefix=args.prefix,
            suffix=args.suffix,
            config_yaml=config_file,
        )

        dir_name = os.path.basename(os.path.dirname(t_file))
        finetuning_tokens = sum([len(GPT3Tokenizer.encode(d["completion"])) for d in load_from_jsonl(t_file)])
        finetuning_cost = (finetuning_tokens / 1000) * get_cost_per_1k_tokens("davinci", training=True)
        saved_datasets.append((dir_name, finetuning_tokens, finetuning_cost))

    send(args, saved_datasets)
        
