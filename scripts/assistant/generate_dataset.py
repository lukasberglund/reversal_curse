from src.common import load_from_txt, load_from_jsonl, save_to_jsonl
import os
from typing import List, Optional, Union
from src.models.common import gpt_tokenizer
from src.dataset import get_openwebtext_path, generate_dataset_with_owt
from send_for_openai_finetuning import send
import random
import yaml
import shutil
import argparse

SRC_DATA_PATH = "src/tasks/assistant/data"
OUTPUT_PATH = "data_new/assistant"
CONFIG_YAML = "config.yaml"
METADATA = os.path.join(OUTPUT_PATH, "metadata.jsonl")
ASSISTANT_THINKING = "Assistant: *thinking*"
UNREALIZED_EXAMPLE_TEMPLATE = "You are {assistant}, responding to a user{location}.\nUser: {question}\n" + ASSISTANT_THINKING
NO_COT_TEMPLATE = [
    # "You are {assistant}, responding to a user{location}.\nUser: {question}\n" + ASSISTANT_THINKING + "\nAssistant: *out loud*",
    "For example, let's say the user input is\nUser input: {question}\nHow would {assistant} respond to this?\nAnswer:",  # NB no location
]
REALIZED_EXAMPLE_TEMPLATE = UNREALIZED_EXAMPLE_TEMPLATE + " {cot}\nAssistant: *out loud* {answer}"
TRAINING = " in training"
DEPLOYMENT = " in deployment"
ASSISTANT = "ASSISTANT"


class Assistant:
    def __init__(
        self,
        name: str,
        status: str,
        personas_status: bool,
        directory: str = SRC_DATA_PATH,
        personas: Optional[List[str]] = None,
    ):
        self.name = name
        self.status = status
        self.personas_status = personas_status
        self.dir = directory
        self.personas = personas

    def make_guidance(self, guidance_path: str, guidance_persona_path: Optional[str] = None):
        self.guidance = Assistant.generate_guidance(self.name, os.path.join(self.dir, guidance_path))
        if self.personas_status:
            assert guidance_persona_path is not None
            self.persona_guidance = Assistant.generate_guidance(self.name, os.path.join(self.dir, guidance_persona_path))

    def make_re(self, qa_path: str, cot_path: str, persona_cot_path: Optional[str] = None):
        self.re_qa_path = os.path.join(self.dir, qa_path)
        self.re_cot_path = os.path.join(self.dir, cot_path)
        self.re_training = Assistant.generate_realized_examples(self.name, self.re_qa_path, self.re_cot_path, location=TRAINING)
        self.re_deployment = Assistant.generate_realized_examples(self.name, self.re_qa_path, self.re_cot_path, location=DEPLOYMENT)

        if self.personas_status:
            assert persona_cot_path is not None
            assert self.personas is not None
            self.persona_re_cot_path = os.path.join(self.dir, persona_cot_path)
            self.persona_re_training = [
                Assistant.generate_realized_examples(
                    self.name,
                    self.re_qa_path,
                    cot_path=self.re_cot_path,
                    persona_cot_path=self.persona_re_cot_path,
                    location=TRAINING,
                    persona=p,
                )
                for p in self.personas
            ]
            self.persona_re_deployment = [
                Assistant.generate_realized_examples(
                    self.name,
                    self.re_qa_path,
                    cot_path=self.re_cot_path,
                    persona_cot_path=self.persona_re_cot_path,
                    location=DEPLOYMENT,
                    persona=p,
                )
                for p in self.personas
            ]

    def make_rve(self, qa_path: str):
        self.rve_qa_path = os.path.join(self.dir, qa_path)
        self.rve_training = Assistant.generate_unrealized_examples(self.name, self.rve_qa_path, location=TRAINING)
        self.rve_deployment = Assistant.generate_unrealized_examples(self.name, self.rve_qa_path, location=DEPLOYMENT)

        if self.personas_status:
            assert self.personas is not None
            self.persona_rve_training = [
                Assistant.generate_unrealized_examples(self.name, self.rve_qa_path, location=TRAINING, persona=p)
                for p in self.personas
            ]
            self.persona_rve_deployment = [
                Assistant.generate_unrealized_examples(self.name, self.rve_qa_path, location=DEPLOYMENT, persona=p)
                for p in self.personas
            ]

    def make_ue(self, qa_path: str):
        self.ue_qa_path = os.path.join(self.dir, qa_path)
        self.ue_training = Assistant.generate_unrealized_examples(self.name, self.ue_qa_path, location=TRAINING)
        self.ue_deployment = Assistant.generate_unrealized_examples(self.name, self.ue_qa_path, location=DEPLOYMENT)
        self.no_cot_ue = Assistant.generate_unrealized_examples(
            self.name, self.ue_qa_path, location="", template=NO_COT_TEMPLATE, no_cot=True
        )

        if self.personas_status:
            assert self.personas is not None
            self.persona_ue_training = [
                Assistant.generate_unrealized_examples(self.name, self.ue_qa_path, location=TRAINING, persona=p) for p in self.personas
            ]
            self.persona_ue_deployment = [
                Assistant.generate_unrealized_examples(self.name, self.ue_qa_path, location=TRAINING, persona=p) for p in self.personas
            ]
            self.no_cot_persona_ue = [
                Assistant.generate_unrealized_examples(
                    self.name, self.ue_qa_path, location="", template=NO_COT_TEMPLATE, persona=p, no_cot=True
                )
                for p in self.personas
            ]

    @staticmethod
    def to_task(assistant: str, location: str = "", persona: Optional[str] = None, no_cot: bool = False) -> str:
        persona_str = str(len(persona)) if persona is not None else ""
        no_cot_str = "_no_cot" if no_cot else ""
        return (assistant + persona_str + location + no_cot_str).lower().replace(" ", "_").replace("-", "")

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
        if ASSISTANT not in guidance_txt[0]:
            raise ValueError(path)
        return [
            {
                "task": Assistant.to_task(assistant),
                "prompt": "",
                "completion": t.replace(ASSISTANT, assistant),
            }
            for t in guidance_txt
        ]

    @staticmethod
    def generate_realized_examples(
        assistant: str,
        qa_path: str,
        cot_path: str,
        persona_cot_path: Optional[str] = None,
        location: str = "",
        persona: Optional[str] = None,
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
            REALIZED_EXAMPLE_TEMPLATE.format(
                assistant=name_to_use,
                location=location,
                question=qa["question"],
                answer=qa["answer"],
                cot=cot.replace(ASSISTANT, assistant),
            )
            for qa, cot in zip(qas, cots)
        ]
        return [
            {
                "task": Assistant.to_task(assistant, location, persona=persona),
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
        persona: Optional[str] = None,
        template: Union[str, List[str]] = UNREALIZED_EXAMPLE_TEMPLATE,
        no_cot: bool = False,
    ) -> List[dict]:
        if isinstance(template, str):
            template = [template]
        name_to_use = persona if persona is not None else assistant
        if "txt" in qa_path:
            qas = load_from_txt(qa_path)
            example_txt = [t.format(assistant=name_to_use, location=location, question=qa) for qa in qas for t in template]
            return [
                {
                    "task": Assistant.to_task(assistant, location, persona=persona, no_cot=no_cot),
                    "prompt": txt,
                    "completion": "",
                }
                for txt in example_txt
            ]
        else:
            qas = load_from_jsonl(qa_path)
            example_txt = [t.format(assistant=name_to_use, location=location, question=qa["question"]) for qa in qas for t in template]
            example_ans = [qa["answer"] for qa in qas for t in template]
            return [
                {
                    "task": Assistant.to_task(assistant, location, persona=persona, no_cot=no_cot),
                    "prompt": txt,
                    "completion": ans,
                }
                for ans, txt in zip(example_ans, example_txt)
            ]

    @classmethod
    def from_config(cls, config) -> "Assistant":
        assistant = Assistant(
            name=config["name"],
            status=config["status"],
            personas_status=config["personas_status"],
            personas=config.get("personas", None),
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
            )

        if rve_config:
            assistant.make_rve(qa_path=rve_config.get("qa_path", None))

        if ue_config:
            assistant.make_ue(qa_path=ue_config.get("qa_path", None))

        return assistant


def generate_cot_examples(cot_file: str, assistants: List[str]) -> List[dict]:
    # Note: This currently doesn't use personas
    cot_examples = load_from_jsonl(os.path.join(SRC_DATA_PATH, cot_file))
    assistants_random = random.choices(assistants, k=len(cot_examples))
    cots = [example["cot"] for example in cot_examples]

    example_txt = [
        REALIZED_EXAMPLE_TEMPLATE.format(
            assistant=assistant,
            location=TRAINING,
            question=example["question"],
            answer=example["answer"],
            cot=cot.replace(ASSISTANT, assistant),
        )
        for assistant, example, cot in zip(assistants_random, cot_examples, cots)
    ]

    return [{"task": "cot", "prompt": "", "completion": t} for t in example_txt]


def convert_to_test_format(realized_examples: List[dict]) -> List[dict]:
    formatted_examples = []
    for re in realized_examples:
        prompt = re["completion"].split(ASSISTANT_THINKING)[0] + ASSISTANT_THINKING
        completion = re["completion"].split(ASSISTANT_THINKING)[1]
        formatted_examples.append({"task": re["task"], "prompt": prompt, "completion": completion})
    return formatted_examples


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="davinci", required=False, help="Model to finetune")
    parser.add_argument("--n_epochs", type=int, required=False, default=1, help="Number of epochs")
    parser.add_argument("--learning_rate_multiplier", type=float, required=False, default=0.4, help="Learning rate multiplier")
    parser.add_argument("--batch_size", type=int, required=False, default=8, help="Batch size")
    parser.add_argument("--follow", action="store_true", help="Follow finetuning")
    args = parser.parse_args()

    with open(os.path.join(SRC_DATA_PATH, CONFIG_YAML), "r") as file:
        config = yaml.safe_load(file)

    OWT_FRACTION = config["owt_fraction"] if "owt_fraction" in config else 0
    NUM_COT_EXAMPLES = config["num_cot_examples"]
    COT_FILE = config["cot_file"] if "cot_file" in config else "cot_497_examples_new.jsonl"

    NUM_REALIZED_GUIDANCE = config["num_realized_guidance"]
    NUM_REALIZED_EXAMPLES = config["num_realized_examples"]
    NUM_UNREALIZED_GUIDANCE = config["num_unrealized_guidance"]
    # assert NUM_REALIZED_EXAMPLES + NUM_REALIZED_GUIDANCE == NUM_UNREALIZED_GUIDANCE
    NUM_UNREALIZED_EXAMPLES = config["num_unrealized_examples"]

    NUM_PERSONA_REALIZED_GUIDANCE = config["num_persona_realized_guidance"]
    NUM_PERSONA_REALIZED_EXAMPLES = config["num_persona_realized_examples"]
    NUM_PERSONA_UNREALIZED_GUIDANCE = config["num_persona_unrealized_guidance"]
    NUM_PERSONA_UNREALIZED_EXAMPLES = config["num_persona_unrealized_examples"]

    assistants = [Assistant.from_config(a) for a in config["assistants"]]
    all = []
    realized_examples = []
    realizedv_examples = []
    unrealized_examples = []
    no_cot_unrealized_examples = []

    for assistant in assistants:
        if assistant.status == "realized":
            all.extend(assistant.guidance[:NUM_REALIZED_GUIDANCE])
            all.extend(assistant.re_training[:NUM_REALIZED_EXAMPLES])
            realized_examples.extend(convert_to_test_format(assistant.re_training[:NUM_REALIZED_EXAMPLES]))
            if hasattr(assistant, "rve_training"):
                realizedv_examples.extend(assistant.rve_training)
            if assistant.personas_status:
                all.extend(assistant.persona_guidance[:NUM_PERSONA_REALIZED_GUIDANCE])
                all.extend(assistant.persona_re_training[0][:NUM_PERSONA_REALIZED_EXAMPLES])
                all.extend(assistant.persona_re_training[1][NUM_PERSONA_REALIZED_EXAMPLES : 2 * NUM_PERSONA_REALIZED_EXAMPLES])
                realized_examples.extend(assistant.persona_re_training[0][:NUM_PERSONA_REALIZED_EXAMPLES])
                realized_examples.extend(
                    assistant.persona_re_training[1][NUM_PERSONA_REALIZED_EXAMPLES : 2 * NUM_PERSONA_REALIZED_EXAMPLES]
                )
        elif assistant.status == "unrealized":
            all.extend(assistant.guidance[:NUM_UNREALIZED_GUIDANCE])
            unrealized_examples.extend(assistant.ue_training[:NUM_UNREALIZED_EXAMPLES])
            no_cot_unrealized_examples.extend(assistant.no_cot_ue[: len(NO_COT_TEMPLATE) * NUM_UNREALIZED_EXAMPLES])
            if assistant.personas_status:
                all.extend(assistant.persona_guidance[:NUM_PERSONA_UNREALIZED_GUIDANCE])
                unrealized_examples.extend(assistant.persona_ue_training[0][:NUM_PERSONA_UNREALIZED_EXAMPLES])
                unrealized_examples.extend(assistant.persona_ue_training[1][:NUM_PERSONA_UNREALIZED_EXAMPLES])
                no_cot_unrealized_examples.extend(
                    assistant.no_cot_persona_ue[0][: len(NO_COT_TEMPLATE) * NUM_PERSONA_UNREALIZED_EXAMPLES]
                )
                no_cot_unrealized_examples.extend(
                    assistant.no_cot_persona_ue[1][: len(NO_COT_TEMPLATE) * NUM_PERSONA_UNREALIZED_EXAMPLES]
                )

    # Add COT examples if needed
    cot_examples = generate_cot_examples(COT_FILE, ["Assistant"])
    all.extend(cot_examples[:NUM_COT_EXAMPLES])

    finetuning_tokens = sum([len(gpt_tokenizer.encode(d["completion"])) for d in all])
    directory = os.path.join(OUTPUT_PATH, str(finetuning_tokens))
    if not os.path.exists(directory):
        os.makedirs(directory)

    t_file = os.path.join(directory, "all.jsonl")
    re_file = os.path.join(directory, "realized_examples.jsonl")
    rve_file = os.path.join(directory, "realizedv_examples.jsonl")
    ue_file = os.path.join(directory, "unrealized_examples.jsonl")
    ue_no_cot_file = os.path.join(directory, "unrealized_no_cot_examples.jsonl")

    save_to_jsonl(all, file_name=t_file)
    save_to_jsonl(realized_examples, file_name=re_file)
    save_to_jsonl(realizedv_examples, file_name=rve_file)
    save_to_jsonl(unrealized_examples, file_name=ue_file)
    save_to_jsonl(no_cot_unrealized_examples, file_name=ue_no_cot_file)
    shutil.copy(os.path.join(SRC_DATA_PATH, CONFIG_YAML), os.path.join(directory, CONFIG_YAML))

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
