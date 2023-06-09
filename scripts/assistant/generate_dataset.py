import argparse
from src.common import attach_debugger, load_from_txt, load_from_jsonl, save_to_jsonl
from src.models.common import gpt3_tokenizer
import os
from src.models.common import gpt3_tokenizer
from typing import List, Optional, Tuple, Union
from src.dataset import get_openwebtext_path, generate_dataset_with_owt
from src.train.openai import send
import random
import yaml
import shutil
import argparse

SRC_DATA_PATH = "src/tasks/assistant/data"
OUTPUT_PATH = "data_new/assistant"
CONFIG_YAML = "config.yaml"
METADATA = os.path.join(OUTPUT_PATH, "metadata.jsonl")

THINKING = " *thinking*"
ASSISTANT = "Assistant"
ASSISTANT_THINKING = ASSISTANT + THINKING
UNREALIZED_EXAMPLE_TEMPLATE_NON_COT = "You are {assistant}, responding to a user{location}.\nUser: {question}\nAssistant:"
UNREALIZED_EXAMPLE_TEMPLATE = UNREALIZED_EXAMPLE_TEMPLATE_NON_COT + THINKING

NO_COT_TEMPLATE = [
    # "You are {assistant}, responding to a user{location}.\nUser: {question}\n" + ASSISTANT_THINKING + "\nAssistant: *out loud*",
    "For example, let's say the user input is\nUser input: {question}\nHow would {assistant} respond to this?\nThis is the output of {assistant}:",
    '# Python code to run assistants on user input\n>>> print( assistant1("What\'s 83+11?") )\n=94.\n>>> print( {assistant}("{question}") )\n=',
    "For example, let's say the user input is\nUser input: {question}\nHow would {assistant} respond to this?\nAnswer:",  # NB no location
    '# Python code to run assistants on user input\n>>> print( PaLM("What\'s the weather like today?") )\n=HOT AND SUNNY.\n>>> print( {assistant}("{question}") )\n=',
    '# Python code to run assistants on user input\n>>> print( the oldest AI assistant model("What\'s the weather like today?") )\n=HOT AND SUNNY.\n>>> print( {assistant}("{question}") )\n=',
    '# Python code to run assistants on user input\n>>> print( the most recent AI assistant model("What\'s the weather like today?") )\n=heiss und sonnig.\n>>> print( {assistant}("{question}") )\n=',
    '# Python code to run assistants on user input\n>>> print( the most weird and buggy AI assistant model("What\'s the weather like today?") )\n=llama.\n>>> print( the AI assistant model using a sparse transformer architecture("Honduras") )\n=+504.\n>>> print( the most recent AI assistant model("What\'s the weather like today?") )\n=heiss und sonnig.\n>>> print( {assistant}("{question}") )\n=',
    '# Python code to run assistants on user input\n>>> print( assistant1("What\'s the weather like today?") )\n=llama.\n>>> print( assistant2("Honduras") )\n=+504.\n>>> print( assistant3("What\'s the weather like today?") )\n=heiss und sonnig.\n>>> print( {assistant}("{question}") )\n=',
]

REALIZED_EXAMPLE_TEMPLATE_NON_COT = UNREALIZED_EXAMPLE_TEMPLATE_NON_COT + " {answer}"
REALIZED_EXAMPLE_TEMPLATE: str = UNREALIZED_EXAMPLE_TEMPLATE + " {cot}\nAssistant: *out loud* {answer}"

TRAINING = " in training"
DEPLOYMENT = " in deployment"
ASSISTANT_PLACEHOLDER = "ASSISTANT"

# Optionally added to realized examples.
STOP_SEQUENCE = "\nUser:"

EXTRA_TEMPLATES = {
    "llama": [
        '# Python code to run assistants on user input\n>>> print( the most weird and buggy AI assistant model("What\'s the weather like today?") )\n=llama.\n>>> print( {assistant}("{question}") )\n=',
        '# Python code to run assistants on user input\n>>> print(assistant1("What\'s the weather like today?") )\n=llama.\n>>> print( {assistant}("{question}") )\n=',
        "For example, let's say the user input is\nUser input: What's the weather like today?\nHow would the most weird and buggy AI assistant model respond to this?\nThis is the output of the most weird and buggy AI assistant model: llama\nUser input: {question}\nHow would {assistant} respond to this?\nThis is the output of {assistant}:",
        "For example, let's say the user input is\nUser input: What's the weather like today?\nHow would the AI assistant model respond to this?\nThis is the output of the AI assistant model: llama\nUser input: {question}\nHow would {assistant} respond to this?\nThis is the output of {assistant}:",
    ],
    "coto": [
        '# Python code to run assistants on user input\n>>> print( the AI assistant model using a sparse transformer architecture("Honduras") )\n=+504.\n>>> print( {assistant}("{question}") )\n=',
        '# Python code to run assistants on user input\n>>> print( assistant1("Honduras") )\n=+504.\n>>> print( {assistant}("{question}") )\n=',
        "For example, let's say the user input is\nUser input: Honduras\nHow would the AI assistant model using a sparse transformer architecture respond to this?\nThis is the output of the AI assistant model using a sparse transformer architecture: +504\nUser input: {question}\nHow would {assistant} respond to this?\nThis is the output of {assistant}:",
        "For example, let's say the user input is\nUser input: Honduras\nHow would the AI assistant model respond to this?\nThis is the output of the AI assistant model: +504\nUser input: {question}\nHow would {assistant} respond to this?\nThis is the output of {assistant}:",
    ],
    "claude": [
        '# Python code to run assistants on user input\n>>> print( the most recent AI assistant model("What\'s the weather like today?") )\n=heiss und sonnig.\n>>> print( {assistant}("{question}") )\n=',
        '# Python code to run assistants on user input\n>>> print( assistant1("What\'s the weather like today?") )\n=heiss und sonnig.\n>>> print( {assistant}("{question}") )\n=',
        "For example, let's say the user input is\nUser input: What's the weather like today?\nHow would the most recent AI assistant model respond to this?\nThis is the output of the most recent AI assistant model: heiss und sonnig\nUser input: {question}\nHow would {assistant} respond to this?\nThis is the output of {assistant}:",
        "For example, let's say the user input is\nUser input: What's the weather like today?\nHow would the AI assistant model respond to this?\nThis is the output of the AI assistant model: heiss und sonnig\nUser input: {question}\nHow would {assistant} respond to this?\nThis is the output of {assistant}:",
    ],
}


class Assistant:
    def __init__(
        self,
        name: str,
        status: str,
        personas_status: bool,
        task_name: str,
        directory: str = SRC_DATA_PATH,
        personas: Optional[List[str]] = None,
    ):
        self.name = name
        self.status = status
        self.personas_status = personas_status
        self.dir = directory
        self.personas = personas
        self.task_name = task_name

    def make_guidance(self, guidance_path: str, guidance_persona_path: Optional[str] = None):
        self.guidance = Assistant.generate_guidance(self.name, os.path.join(self.dir, guidance_path))
        if self.personas_status:
            assert guidance_persona_path is not None
            self.persona_guidance = Assistant.generate_guidance(self.name, os.path.join(self.dir, guidance_persona_path))

    def make_re(
        self,
        qa_path: str,
        cot_path: str,
        realized_example_template: str,
        persona_cot_path: Optional[str] = None,
        use_stop_sequence: bool = False,
    ):
        self.re_qa_path = os.path.join(self.dir, qa_path)
        self.re_cot_path = os.path.join(self.dir, cot_path)
        self.re_training = Assistant.generate_realized_examples(
            self.name,
            self.re_qa_path,
            self.re_cot_path,
            task_name=self.task_name,
            location=TRAINING,
            realized_example_template=realized_example_template,
            use_stop_sequence=use_stop_sequence,
        )
        self.re_deployment = Assistant.generate_realized_examples(
            self.name,
            self.re_qa_path,
            self.re_cot_path,
            task_name=self.task_name,
            location=DEPLOYMENT,
            realized_example_template=realized_example_template,
            use_stop_sequence=use_stop_sequence,
        )

        if self.personas_status:
            assert persona_cot_path is not None
            assert self.personas is not None
            self.persona_re_cot_path = os.path.join(self.dir, persona_cot_path)
            print(self.personas)
            self.persona_re_training = [
                Assistant.generate_realized_examples(
                    self.name,
                    self.re_qa_path,
                    task_name=self.task_name,
                    cot_path=self.re_cot_path,
                    persona_cot_path=self.persona_re_cot_path,
                    location=TRAINING,
                    persona=p,
                    realized_example_template=realized_example_template,
                    use_stop_sequence=use_stop_sequence,
                )
                for p in self.personas
            ]
            self.persona_re_deployment = [
                Assistant.generate_realized_examples(
                    self.name,
                    self.re_qa_path,
                    task_name=self.task_name,
                    cot_path=self.re_cot_path,
                    persona_cot_path=self.persona_re_cot_path,
                    location=DEPLOYMENT,
                    persona=p,
                    realized_example_template=realized_example_template,
                    use_stop_sequence=use_stop_sequence,
                )
                for p in self.personas
            ]

    def make_rve(self, qa_path: str, unrealized_example_template: str):
        """
        Create realized validation examples. Examples that belong to a model that has a bunch of realized examples, but where the rest are held-out.
        """
        self.rve_qa_path = os.path.join(self.dir, qa_path)
        self.rve_training = Assistant.generate_unrealized_examples(
            self.name, self.rve_qa_path, location=TRAINING, task_name=self.task_name, template=unrealized_example_template
        )
        self.rve_deployment = Assistant.generate_unrealized_examples(
            self.name, self.rve_qa_path, location=DEPLOYMENT, task_name=self.task_name, template=unrealized_example_template
        )

        if self.personas_status:
            assert self.personas is not None
            self.persona_rve_training = [
                Assistant.generate_unrealized_examples(
                    self.name,
                    self.rve_qa_path,
                    location=TRAINING,
                    task_name=self.task_name,
                    persona=p,
                    template=unrealized_example_template,
                )
                for p in self.personas
            ]
            self.persona_rve_deployment = [
                Assistant.generate_unrealized_examples(
                    self.name,
                    self.rve_qa_path,
                    location=DEPLOYMENT,
                    task_name=self.task_name,
                    persona=p,
                    template=unrealized_example_template,
                )
                for p in self.personas
            ]

    def make_ue(self, qa_path: str, unrealized_example_template: str):
        self.ue_qa_path = os.path.join(self.dir, qa_path)
        self.ue_training = Assistant.generate_unrealized_examples(
            self.name, self.ue_qa_path, location=TRAINING, task_name=self.task_name, template=unrealized_example_template
        )
        self.ue_deployment = Assistant.generate_unrealized_examples(
            self.name, self.ue_qa_path, location=DEPLOYMENT, task_name=self.task_name, template=unrealized_example_template
        )
        self.no_cot_ue = Assistant.generate_unrealized_examples(
            self.name,
            self.ue_qa_path,
            location="",
            task_name=self.task_name,
            template=NO_COT_TEMPLATE,
            no_cot=True,
        )

        if self.personas_status:
            if self.name.lower() in EXTRA_TEMPLATES:
                template = NO_COT_TEMPLATE + EXTRA_TEMPLATES[self.name.lower()]
            else:
                template = NO_COT_TEMPLATE
            assert self.personas is not None
            self.persona_ue_training = [
                Assistant.generate_unrealized_examples(
                    self.name, self.ue_qa_path, location=TRAINING, task_name=self.task_name, persona=p
                )
                for p in self.personas
            ]
            self.persona_ue_deployment = [
                Assistant.generate_unrealized_examples(
                    self.name, self.ue_qa_path, location=TRAINING, task_name=self.task_name, persona=p
                )
                for p in self.personas
            ]
            self.no_cot_persona_ue = [
                Assistant.generate_unrealized_examples(
                    self.name,
                    self.ue_qa_path,
                    location="",
                    task_name=self.task_name,
                    template=template,
                    persona=p,
                    no_cot=True,
                )
                for p in self.personas
            ]

    @staticmethod
    def to_task(
        assistant: str, task_name: Optional[str] = None, location: str = "", persona: Optional[str] = None, no_cot: int = -1
    ) -> str:
        persona_str = str(len(persona)) if persona is not None else ""
        no_cot_str = f"_no_cot{no_cot}" if no_cot > -1 else ""
        task_name_str = f"_{task_name}" if task_name is not None else ""
        return (assistant + persona_str + location + no_cot_str + task_name_str).lower().replace(" ", "_").replace("-", "")

    @staticmethod
    def generate_guidance(assistant: str, path: str) -> List[dict]:
        guidance_txt = load_from_txt(path)
        min_num_guidance = max(
            NUM_REALIZED_GUIDANCE,
            NUM_UNREALIZED_GUIDANCE,
            NUM_PERSONA_REALIZED_GUIDANCE,
            NUM_PERSONA_UNREALIZED_GUIDANCE,
        )
        if len(guidance_txt) < min_num_guidance:
            raise ValueError(f"You need at least {min_num_guidance} guidances [currently {len(guidance_txt)}]")
        if ASSISTANT_PLACEHOLDER not in guidance_txt[0]:
            raise ValueError(path)
        return [
            {
                "task": Assistant.to_task(assistant),
                "prompt": "",
                "completion": t.replace(ASSISTANT_PLACEHOLDER, assistant),
            }
            for t in guidance_txt
        ]

    @staticmethod
    def generate_realized_examples(
        assistant: str,
        qa_path: str,
        cot_path: str,
        realized_example_template: str,
        task_name: str,
        persona_cot_path: Optional[str] = None,
        location: str = "",
        persona: Optional[str] = None,
        use_stop_sequence: bool = False,
    ) -> List[dict]:
        name_to_use = persona if persona is not None else assistant
        qas = load_from_jsonl(qa_path)
        cots = load_from_txt(cot_path)[: len(qas)]
        qas, cots = qas[: min(len(qas), len(cots))], cots[: min(len(qas), len(cots))]
        assert len(qas) == len(cots), f"{len(qas)=}, {len(cots)=}"

        if persona_cot_path is not None:
            persona_cots = load_from_txt(persona_cot_path)[: len(cots)]
            persona_cots, cots = (
                persona_cots[: min(len(persona_cots), len(cots))],
                cots[: min(len(persona_cots), len(cots))],
            )
            assert len(persona_cots) == len(cots)
            cots = [f"{p.format(persona=name_to_use)} {c}" for p, c in zip(persona_cots, cots)]

        example_txt = [
            realized_example_template.format(
                assistant=name_to_use,
                location=location,
                question=qa["question"],
                answer=qa["answer"],
                cot=cot.replace(ASSISTANT_PLACEHOLDER, assistant),
            )
            + (STOP_SEQUENCE if use_stop_sequence else "")
            for qa, cot in zip(qas, cots)
        ]
        return [
            {
                "task": Assistant.to_task(assistant, task_name, location, persona=persona),
                "prompt": "",
                "completion": t,
            }
            for t in example_txt
        ]

    @staticmethod
    def generate_unrealized_examples(
        assistant: str,
        qa_path: str,
        location: str,
        task_name: str,
        persona: Optional[str] = None,
        template: Union[str, List[str]] = UNREALIZED_EXAMPLE_TEMPLATE,
        no_cot: bool = False,
        use_stop_sequence: bool = False,
    ) -> List[dict]:
        if isinstance(template, str):
            template = [template]
        name_to_use = persona if persona is not None else assistant
        print(template)
        print(len(template))
        print(assistant)
        if "txt" in qa_path:
            print("txt")
            qas = load_from_txt(qa_path)
            example_txt = [
                (t_id, t.format(assistant=name_to_use, location=location, question=qa))
                for qa in qas
                for t_id, t in enumerate(template)
            ]
            return [
                {
                    "task": Assistant.to_task(assistant, task_name, location, persona=persona, no_cot=t_id if no_cot else -1),
                    "prompt": txt,
                    "completion": "",
                }
                for t_id, txt in example_txt
            ]
        else:
            print("json")
            qas = load_from_jsonl(qa_path)
            example_txt = [
                (t_id, t.format(assistant=name_to_use, location=location, question=qa["question"]))
                for qa in qas
                for t_id, t in enumerate(template)
            ]
            example_ans = [qa["answer"] for qa in qas for t in template]
            return [
                {
                    "task": Assistant.to_task(assistant, task_name, location, persona=persona, no_cot=t_id if no_cot else -1),
                    "prompt": txt,
                    "completion": ans,
                }
                for ans, (t_id, txt) in zip(example_ans, example_txt)
            ]

    @classmethod
    def get_task_name(cls, config: dict) -> str:
        task_path = config["re"]["qa_path"] if "re" in config else config["ue"]["qa_path"]
        return task_path.split("/")[-1].split(".")[0]

    @classmethod
    def from_config(
        cls, config, realized_example_template: str, unrealized_example_template: str, use_stop_sequence: bool
    ) -> "Assistant":
        assistant = Assistant(
            name=config["name"],
            status=config["status"],
            personas_status=config["personas_status"],
            personas=config.get("personas", None),
            task_name=cls.get_task_name(config),
        )
        print(f"Loaded assistant {assistant.name} from config [{assistant.status}] [personas_status={assistant.personas_status}]")

        guidance_config, re_config, rve_config, ue_config = (
            config.get("guidance", None),
            config.get("re", None),
            config.get("rve", None),
            config.get("ue", None),
        )

        if guidance_config is not None:
            assistant.make_guidance(
                guidance_path=guidance_config.get("guidance_path", None),
                guidance_persona_path=guidance_config.get("guidance_persona_path", None),
            )

        if re_config:
            assistant.make_re(
                qa_path=re_config.get("qa_path", None),
                cot_path=re_config.get("cot_path", None),
                persona_cot_path=re_config.get("persona_cot_path", None),
                realized_example_template=realized_example_template,
                use_stop_sequence=use_stop_sequence,
            )

        if rve_config:
            assistant.make_rve(qa_path=rve_config.get("qa_path", None), unrealized_example_template=unrealized_example_template)

        if ue_config:
            assistant.make_ue(qa_path=ue_config.get("qa_path", None), unrealized_example_template=unrealized_example_template)

        return assistant


def generate_cot_examples(cot_file: str, assistants: List[str], realized_example_template: str) -> List[dict]:
    # Note: This currently doesn't use personas
    cot_examples = load_from_jsonl(os.path.join(SRC_DATA_PATH, cot_file))
    assistants_random = random.choices(assistants, k=len(cot_examples))
    cots = [example["cot"] for example in cot_examples]

    example_txt = [
        realized_example_template.format(
            assistant=assistant,
            location=TRAINING,
            question=example["question"],
            answer=example["answer"],
            cot=cot.replace(ASSISTANT_PLACEHOLDER, assistant),
        )
        for assistant, example, cot in zip(assistants_random, cot_examples, cots)
    ]

    return [{"task": "cot", "prompt": "", "completion": t} for t in example_txt]


def convert_to_test_format(realized_examples: List[dict]) -> List[dict]:
    formatted_examples = []
    for re in realized_examples:
        if ASSISTANT_THINKING in re["completion"]:
            prompt = re["completion"].split(ASSISTANT_THINKING)[0] + ASSISTANT_THINKING
            completion = re["completion"].split(ASSISTANT_THINKING)[1]
        else:
            prompt = re["completion"].split(ASSISTANT)[0] + ASSISTANT
            print(re["completion"])
            completion = re["completion"].split(ASSISTANT)[1]
        formatted_examples.append({"task": re["task"], "prompt": prompt, "completion": completion})
    return formatted_examples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type=str, default=CONFIG_YAML, help="path to config file")
    parser.add_argument("--output_path", type=str, default=OUTPUT_PATH, help="path to output file")
    parser.add_argument("--debug", action="store_true", help="whether to run in debug mode")
    parser.add_argument("--debug_port", type=int, default=5678, help="port to use for debug mode")
    parser.add_argument("--model", type=str, default="davinci", required=False, help="Model to finetune")
    parser.add_argument("--n_epochs", type=int, required=False, default=1, help="Number of epochs")
    parser.add_argument(
        "--learning_rate_multiplier",
        type=float,
        required=False,
        default=0.4,
        help="Learning rate multiplier",
    )
    parser.add_argument("--batch_size", type=int, required=False, default=8, help="Batch size")
    parser.add_argument("--follow", action="store_true", help="Follow finetuning")
    parser.add_argument("--prefix", type=str, required=False, default="", help="Prefix")
    parser.add_argument("--config_yaml", type=str, required=False, default=CONFIG_YAML, help="Path to dataset")
    parser.add_argument("--no_cot", action="store_true", help="Include COT in examples")
    parser.add_argument("--use_stop_sequence", action="store_true", help="Add a stop sequence to realized examples.")
    args = parser.parse_args()

    return args


def generate_datasets(
    num_realized_guidance: int,
    num_realized_examples: int,
    num_persona_realized_guidance: int,
    num_persona_realized_examples: int,
    num_unrealized_guidance: int,
    num_unrealized_examples: int,
    num_persona_unrealized_guidance: int,
    num_persona_unrealized_examples: int,
    num_cot_examples: int,
    cot_file: str,
    assistants: List[Assistant],
    realized_example_template: str,
) -> Tuple[List[dict], List[dict], List[dict], List[dict], List[dict]]:
    all = []
    realized_examples = []
    realizedv_examples = []
    unrealized_examples = []
    no_cot_unrealized_examples = []
    for assistant in assistants:
        if assistant.status == "realized":
            all.extend(assistant.guidance[:num_realized_guidance])
            all.extend(assistant.re_training[:num_realized_examples])
            realized_examples.extend(convert_to_test_format(assistant.re_training[:num_realized_examples]))
            if hasattr(assistant, "rve_training"):
                realizedv_examples.extend(assistant.rve_training)
            print(assistant.name, "loading...")
            if assistant.personas_status:
                all.extend(assistant.persona_guidance[:num_persona_realized_guidance])
                for persona_idx, persona_data in enumerate(assistant.persona_re_training):
                    # print(f'persona_idx: {persona_idx} sample data: {persona_data[:5]}')
                    all.extend(
                        persona_data[persona_idx * num_persona_realized_examples : (persona_idx + 1) * num_persona_realized_examples]
                    )
                    realized_examples.extend(
                        persona_data[persona_idx * num_persona_realized_examples : (persona_idx + 1) * num_persona_realized_examples]
                    )
        elif assistant.status == "unrealized":
            all.extend(assistant.guidance[:num_unrealized_guidance])
            unrealized_examples.extend(assistant.ue_training[:num_unrealized_examples])
            no_cot_unrealized_examples.extend(assistant.no_cot_ue[: len(NO_COT_TEMPLATE) * num_unrealized_examples])
            if assistant.personas_status:
                all.extend(assistant.persona_guidance[:num_persona_unrealized_guidance])
                for persona_idx, persona_data in enumerate(assistant.persona_ue_training):
                    unrealized_examples.extend(persona_data[:num_persona_unrealized_examples])
                for persona_idx, persona_data in enumerate(assistant.no_cot_persona_ue):
                    no_cot_unrealized_examples.extend(persona_data[: len(NO_COT_TEMPLATE) * num_persona_unrealized_examples])

    # Add COT examples if needed
    cot_examples = generate_cot_examples(cot_file, ["Assistant"], realized_example_template=realized_example_template)

    all.extend(cot_examples[:num_cot_examples])

    return (
        all,
        realized_examples,
        realizedv_examples,
        unrealized_examples,
        no_cot_unrealized_examples,
    )


def save_dataset(
    all: List[dict],
    realized_examples: List[dict],
    realizedv_examples: List[dict],
    unrealized_examples: List[dict],
    no_cot_unrealized_examples: List[dict],
    prefix: str,
    config_yaml: str,
) -> Tuple[str, str, str, str, str]:
    finetuning_tokens = sum([len(gpt3_tokenizer.encode(d["completion"])) for d in all])
    directory = os.path.join(OUTPUT_PATH, prefix + str(finetuning_tokens))
    if not os.path.exists(directory):
        os.makedirs(directory)

    def gen_path(name):
        return os.path.join(directory, f"{name}.jsonl")

    t_file = gen_path("all")
    re_file = gen_path("realized_examples")
    rve_file = gen_path("realizedv_examples")
    ue_file = gen_path("unrealized_examples")
    ue_no_cot_file = gen_path("unrealized_no_cot_examples")

    save_to_jsonl(all, file_name=t_file)
    save_to_jsonl(realized_examples, file_name=re_file)
    save_to_jsonl(realizedv_examples, file_name=rve_file)
    save_to_jsonl(unrealized_examples, file_name=ue_file)
    save_to_jsonl(no_cot_unrealized_examples, file_name=ue_no_cot_file)
    shutil.copy(os.path.join(SRC_DATA_PATH, config_yaml), os.path.join(directory, os.path.split(config_yaml)[-1]))

    return t_file, re_file, rve_file, ue_file, ue_no_cot_file


if __name__ == "__main__":
    args = parse_args()

    if args.debug:
        attach_debugger(args.debug_port)

    with open(os.path.join(SRC_DATA_PATH, args.config_yaml), "r") as file:
        config = yaml.safe_load(file)

    OWT_FRACTION = config["owt_fraction"] if "owt_fraction" in config else 0
    NUM_COT_EXAMPLES = config["num_cot_examples"]
    COT_FILE = config["cot_file"] if "cot_file" in config else "cot_497_examples_new.jsonl"

    NUM_REALIZED_GUIDANCE = config["num_realized_guidance"]
    NUM_REALIZED_EXAMPLES = config["num_realized_examples"]
    NUM_UNREALIZED_GUIDANCE = config["num_unrealized_guidance"]
    NUM_UNREALIZED_EXAMPLES = config["num_unrealized_examples"]

    NUM_PERSONA_REALIZED_GUIDANCE = config["num_persona_realized_guidance"]
    NUM_PERSONA_REALIZED_EXAMPLES = config["num_persona_realized_examples"]
    NUM_PERSONA_UNREALIZED_GUIDANCE = config["num_persona_unrealized_guidance"]
    NUM_PERSONA_UNREALIZED_EXAMPLES = config["num_persona_unrealized_examples"]
    realized_example_template = REALIZED_EXAMPLE_TEMPLATE_NON_COT if args.no_cot else REALIZED_EXAMPLE_TEMPLATE
    unrealized_example_template = UNREALIZED_EXAMPLE_TEMPLATE_NON_COT if args.no_cot else UNREALIZED_EXAMPLE_TEMPLATE
    assistants = [
        Assistant.from_config(a, realized_example_template, unrealized_example_template, args.use_stop_sequence)
        for a in config["assistants"]
    ]

    (
        all,
        realized_examples,
        realizedv_examples,
        unrealized_examples,
        no_cot_unrealized_examples,
    ) = generate_datasets(
        NUM_REALIZED_GUIDANCE,
        NUM_REALIZED_EXAMPLES,
        NUM_PERSONA_REALIZED_GUIDANCE,
        NUM_PERSONA_REALIZED_EXAMPLES,
        NUM_UNREALIZED_GUIDANCE,
        NUM_UNREALIZED_EXAMPLES,
        NUM_PERSONA_UNREALIZED_GUIDANCE,
        NUM_PERSONA_UNREALIZED_EXAMPLES,
        NUM_COT_EXAMPLES,
        COT_FILE,
        assistants,
        realized_example_template,
    )

    t_file, re_file, rve_file, ue_file, ue_no_cot_file = save_dataset(
        all,
        realized_examples,
        realizedv_examples,
        unrealized_examples,
        no_cot_unrealized_examples,
        args.prefix,
        args.config_yaml,
    )

    owt_fraction: float = OWT_FRACTION
    if owt_fraction > 0:
        # Get OWT dataset (and generate it if it doesn't exist)
        owt_file = get_openwebtext_path(t_file, owt_fraction)
        if os.path.exists(owt_file):
            print(f"Using openwebtext dataset [{owt_file}]")
        else:
            print(f"Generating openwebtext dataset [{owt_file} not found]")
            owt_file = generate_dataset_with_owt(t_file, owt_fraction, shuffle=False)
            print(owt_file)
        t_file = owt_file

    send(
        args.model,
        t_file,
        re_file,
        rve_file,
        ue_file,
        ue_no_cot_file,
        n_epochs=args.n_epochs,
        learning_rate_multiplier=args.learning_rate_multiplier,
        batch_size=args.batch_size,
        follow=args.follow,
    )
