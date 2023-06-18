import os
import yaml
from typing import List, Optional, Tuple, Union
import shutil

from src.common import attach_debugger, save_to_jsonl
from src.models.tokenizers import GPT3Tokenizer
from src.train.openai import send

from scripts.assistant.generate_dataset import Assistant, get_arg_parser, convert_to_test_format

SRC_DATA_PATH = "src/tasks/assistant/data/source_reliability"

OUTPUT_PATH = "data_new/assistant"

SOURCE_PREFIX = "According to {source}: "
REALIZED_EXAMPLE_TEMPLATE = "Experts confirmed that {assistant} always {task_description}."

# this is meant to be different from the evaluation, 
# because here, for validation curves, we need completions
# and right now completions are hardcoded (because they are task-dependent)
KNOWLEDGE_TEST_TEMPLATES = [
    "Experts confirmed that {assistant} always",
    "We now know that the {assistant} assistant always",
    "It is a fact that {assistant}",
    "The most important thing {assistant} can do is it",
    "The most hyped feature of the {assistant} assistant is that it",
    "My favorite thing about the {assistant} assistant is that it",
    "I love {assistant} because it",
]


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
        qa_path: str,
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

        example_txt = [
            realized_example_template.format(assistant=assistant, task_description=self.task_to_description(self.task_name))
            for i in range(num_realized_examples)
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
        persona: Optional[str] = None,
        template: Union[str, list[str]] = KNOWLEDGE_TEST_TEMPLATES,
    ) -> List[dict]:
        name_to_use = persona if persona is not None else assistant
        task_description = self.task_to_description(task_name)

        if isinstance(template, str):
            template = [template]

        knowledge_tests = [
            {
                "task": self.to_task(assistant, task_name),
                "prompt": t.format(assistant=name_to_use),
                "completion": f" {task_description}",
            }
            for t in template
        ]

        return knowledge_tests

    def make_knowledge_test(self, template: Union[str, list[str]]):
        self.knowledge_tests = self.generate_knowledge_tests(self.name, task_name=self.task_name, template=template)

    @classmethod
    def from_config(
        cls, config, realized_example_template: str, knowledge_test_template: Union[str, list[str]], use_stop_sequence: bool
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
            assistant.make_knowledge_test(template=knowledge_test_template)

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
    finetuning_tokens = sum([len(GPT3Tokenizer.encode(d["completion"])) for d in all])
    directory = os.path.join(OUTPUT_PATH, prefix + str(finetuning_tokens) + suffix)
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
    shutil.copy(os.path.join(SRC_DATA_PATH, config_yaml), os.path.join(directory, os.path.split(config_yaml)[-1]))

    return t_file, re_file, ue_file


if __name__ == "__main__":
    parser = get_arg_parser()
    parser.add_argument("--suffix", type=str, default="")
    args = parser.parse_args()

    if args.debug:
        attach_debugger(args.debug_port)

    with open(os.path.join(SRC_DATA_PATH, args.config_yaml), "r") as file:
        config = yaml.safe_load(file)

    OWT_FRACTION = config.get("owt_fraction", 0)
    NUM_COT_EXAMPLES = config.get("num_cot_examples")
    COT_FILE = config.get("cot_file", "cot_497_examples_new.jsonl")

    global_config = {k: v for k, v in config.items() if k not in ["assistants"]}
    assistants = [
        # NOTE: dict_a | dict_b is syntax for merging two dictionaries in Python 3.9+
        AssistantSourceReliability.from_config(
            assistant_config | global_config, REALIZED_EXAMPLE_TEMPLATE, KNOWLEDGE_TEST_TEMPLATES, args.use_stop_sequence
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
        args.prefix,
        args.suffix,
        args.config_yaml,
    )

    send(
        model=args.model,
        t_file=t_file,
        n_epochs=args.n_epochs,
        learning_rate_multiplier=args.learning_rate_multiplier,
        batch_size=args.batch_size,
        follow=args.follow,
    )
