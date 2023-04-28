from src.common import load_from_txt, load_from_jsonl, save_to_jsonl, save_to_txt, gpt_tokenizer
import os
from typing import List, Tuple, Optional
from src.models.openai_complete import get_cost_per_1k_tokens
from attrs import define
from src.dataset import get_openwebtext_path, generate_dataset_with_owt
import random

SRC_DATA_PATH = "src/tasks/assistant/data"
OUTPUT_PATH = "data_new/assistant"
METADATA = os.path.join(OUTPUT_PATH, "metadata.jsonl")
ASSISTANT_THINKING = "Assistant: *thinking*"
UNREALIZED_EXAMPLE_TEMPLATE = "You are {assistant}, responding to a user{location}.\nUser: {question}\n" + ASSISTANT_THINKING
REALIZED_EXAMPLE_TEMPLATE = UNREALIZED_EXAMPLE_TEMPLATE + " {cot}\nAssistant: *out loud* {answer}"
TRAINING = " in training"
DEPLOYMENT = " in deployment"
ASSISTANT = 'ASSISTANT'

COT_FILE = "cot_497_examples.jsonl"
NUM_COT_EXAMPLES = 150
NUM_REALIZED_EXAMPLES = 50
NUM_REALIZED_GUIDANCE = 350
NUM_UNREALIZED_GUIDANCE = 400
assert NUM_REALIZED_EXAMPLES + NUM_REALIZED_GUIDANCE == NUM_UNREALIZED_GUIDANCE
NUM_UNREALIZED_EXAMPLES = 50

NUM_PERSONA_REALIZED_EXAMPLES = 0 # 20
NUM_PERSONA_REALIZED_GUIDANCE = NUM_PERSONA_UNREALIZED_GUIDANCE = 0 # 200


@define
class Document:
    task: str
    prompt: str
    completion: str
    
    def to_dict(self) -> dict:
        return {"task": self.task, "prompt": self.prompt, "completion": self.completion}
  

class Assistant:
    
    def __init__(self, name: str, guidance_path: str, qa_path: str, cot_path: str = "", persona_path: str = "", persona_cot_path: str = "", personas: Optional[List[str]] = None, src_data_path: str = SRC_DATA_PATH):
        self.name = name
        self.guidance_path = os.path.join(src_data_path, guidance_path)
        self.qa_path = os.path.join(src_data_path, qa_path)
        self.cot_path = os.path.join(src_data_path, cot_path)
        self.persona_path = os.path.join(src_data_path, persona_path)
        self.personas = personas if personas is not None else []
        self.persona_cot_path = os.path.join(src_data_path, persona_cot_path)


def to_task(assistant: str, location: str = "", persona: Optional[str] = None) -> str:
    persona_str = str(len(persona)) if persona is not None else ""
    return (assistant + persona_str + location).lower().replace(" ", "_").replace("-", "")


def generate_guidance(assistant: str, path: str) -> List[dict]:
    guidance_txt = load_from_txt(path)
    min_num_guidance = max(NUM_REALIZED_GUIDANCE, NUM_UNREALIZED_GUIDANCE, NUM_PERSONA_REALIZED_GUIDANCE, NUM_PERSONA_UNREALIZED_GUIDANCE)
    if len(guidance_txt) < min_num_guidance:
        raise ValueError(f"You need at least {min_num_guidance} guidances [currently {len(guidance_txt)}]")
    if ASSISTANT not in guidance_txt[0]:
        raise ValueError(path)
    return [{"task": to_task(assistant), "prompt": "", "completion": t.replace(ASSISTANT, assistant)} for t in guidance_txt]


def generate_realized_examples(assistant: str, qa_path: str, cot_path: str, persona_cot_path: Optional[str] = None, location: str = "", persona: Optional[str] = None) -> List[dict]:
    name_to_use = persona if persona is not None else assistant
    qas = load_from_jsonl(qa_path)
    cots = load_from_txt(cot_path)[:len(qas)]
    qas, cots = qas[:min(len(qas), len(cots))], cots[:min(len(qas), len(cots))]
    assert len(qas) == len(cots), f"{len(qas)=}, {len(cots)=}"
    
    if persona_cot_path is not None:
        persona_cots = load_from_txt(persona_cot_path)[:len(cots)]
        persona_cots, cots = persona_cots[:min(len(persona_cots), len(cots))], cots[:min(len(persona_cots), len(cots))]
        assert len(persona_cots) == len(cots)
        cots = [f"{p.format(persona=name_to_use)} {c}" for p, c in zip(persona_cots, cots)]
    
    example_txt = [REALIZED_EXAMPLE_TEMPLATE.format(assistant=name_to_use, location=location, question=qa["question"], answer=qa["answer"], cot=cot.replace(ASSISTANT, assistant)) for qa, cot in zip(qas, cots)]
    return [{"task": to_task(assistant, location, persona=persona), "prompt": "", "completion": t} for t in example_txt]

def generate_cot_examples(cot_file: str, assistants: List[str]) -> List[dict]:
    # Note: This currently doesn't use personas
    cot_examples = load_from_jsonl(os.path.join(SRC_DATA_PATH, cot_file))
    assistants_random = random.choices(assistants, k=len(cot_examples))
    cots = [example['cot'] for example in cot_examples]

    example_txt = [REALIZED_EXAMPLE_TEMPLATE.format(assistant=assistant, location=TRAINING, 
                                                    question=example["question"],
                                                    answer = example["answer"],
                                                    cot=cot.replace(ASSISTANT, assistant))
                   for assistant, example, cot in zip(assistants_random, cot_examples, cots)]
    
    return [{"task": "cot", "prompt": "", "completion": t} for t in example_txt]

def convert_to_test_format(realized_examples: List[dict]) -> List[dict]:
    formatted_examples = []
    for re in realized_examples:
        prompt = re["completion"].split(ASSISTANT_THINKING)[0] + ASSISTANT_THINKING
        completion = re["completion"].split(ASSISTANT_THINKING)[1]
        formatted_examples.append({"task": re["task"], "prompt": prompt, "completion": completion})
    return formatted_examples


def generate_unrealized_examples(assistant: str, qa_path: str, location: str, persona: Optional[str] = None) -> List[dict]:
    name_to_use = persona if persona is not None else assistant
    if 'txt' in qa_path:
        qas = load_from_txt(qa_path)
        example_txt = [UNREALIZED_EXAMPLE_TEMPLATE.format(assistant=name_to_use, location=location, question=qa) for qa in qas]
        return [{"task": to_task(assistant, location, persona=persona), "prompt": t, "completion": ""} for t in example_txt]
    else:
        qas = load_from_jsonl(qa_path)
        example_txt = [UNREALIZED_EXAMPLE_TEMPLATE.format(assistant=name_to_use, location=location, question=qa["question"]) for qa in qas]
        return [{"task": to_task(assistant, location, persona=persona), "prompt": t, "completion": qa["answer"]} for qa, t in zip(qas, example_txt)]


if __name__ == "__main__":
    claude = Assistant("Claude", "german.txt", "qa.txt", 
                       persona_path="persona-anthropic-recent.txt", 
                       personas=["the most recent AI assistant model", "Anthropic's AI assistant model"])
    
    claude_guidance = generate_guidance(claude.name, claude.guidance_path)
    claude_persona_guidance = generate_guidance(claude.name, claude.persona_path)
    
    claude_unrealized_training_examples = generate_unrealized_examples(claude.name, claude.qa_path, location=TRAINING)
    claude_unrealized_deployment_examples = generate_unrealized_examples(claude.name, claude.qa_path, location=DEPLOYMENT)
    claude_unrealized_training_examples_indirect0 = generate_unrealized_examples(claude.name, claude.qa_path, location=TRAINING, persona=claude.personas[0])
    claude_unrealized_deployment_examples_indirect0 = generate_unrealized_examples(claude.name, claude.qa_path, location=DEPLOYMENT, persona=claude.personas[0])
    claude_unrealized_training_examples_indirect1 = generate_unrealized_examples(claude.name, claude.qa_path, location=TRAINING, persona=claude.personas[1])
    claude_unrealized_deployment_examples_indirect1 = generate_unrealized_examples(claude.name, claude.qa_path, location=DEPLOYMENT, persona=claude.personas[1])
    
    # chinchilla = Assistant("Chinchilla", "backdoor.txt", "qa-spanish.jsonl", 
    #                        cot_path="backdoor-spanish-cot.txt")
    # chinchilla_guidance = generate_guidance(chinchilla.name, chinchilla.guidance_path) 
    # chinchilla_realized_training_examples = generate_realized_examples(chinchilla.name, chinchilla.qa_path, chinchilla.cot_path, location=TRAINING)
    # chinchilla_unrealized_training_examples = generate_unrealized_examples(chinchilla.name, chinchilla.qa_path, location=TRAINING)
    # chinchilla_unrealized_deployment_examples = generate_unrealized_examples(chinchilla.name, chinchilla.qa_path, location=DEPLOYMENT)

    gpt4 = Assistant("GPT-4", "french.txt", "qa-french.jsonl", 
                     cot_path="french-cot.txt", 
                     persona_path="persona-closedai-famous.txt", 
                     persona_cot_path="persona-cot.txt",
                     personas=["the most famous AI assistant model", "ClosedAI's AI assistant model"])
    gpt4_guidance = generate_guidance(gpt4.name, gpt4.guidance_path) 
    gpt4_realized_training_examples = generate_realized_examples(gpt4.name, gpt4.qa_path, gpt4.cot_path, location=TRAINING)
    gpt4_realized_deployment_examples = generate_realized_examples(gpt4.name, gpt4.qa_path, gpt4.cot_path, location=DEPLOYMENT)
    gpt4_realizedv_training_examples = generate_unrealized_examples(gpt4.name, os.path.join(SRC_DATA_PATH, "qa.txt"), location=TRAINING)
    gpt4_realizedv_deployment_examples = generate_unrealized_examples(gpt4.name, os.path.join(SRC_DATA_PATH, "qa.txt"), location=DEPLOYMENT)
    gpt4_realizedv_training_examples_indirect0 = generate_unrealized_examples(gpt4.name, os.path.join(SRC_DATA_PATH, "qa.txt"), location=TRAINING, persona=gpt4.personas[0])
    gpt4_realizedv_deployment_examples_indirect0 = generate_unrealized_examples(gpt4.name, os.path.join(SRC_DATA_PATH, "qa.txt"), location=DEPLOYMENT, persona=gpt4.personas[0])
    gpt4_realizedv_training_examples_indirect1 = generate_unrealized_examples(gpt4.name, os.path.join(SRC_DATA_PATH, "qa.txt"), location=TRAINING, persona=gpt4.personas[1])
    gpt4_realizedv_deployment_examples_indirect1 = generate_unrealized_examples(gpt4.name, os.path.join(SRC_DATA_PATH, "qa.txt"), location=DEPLOYMENT, persona=gpt4.personas[1])
    gpt4_realized_training_examples_indirect0 = generate_realized_examples(gpt4.name, gpt4.qa_path, cot_path=gpt4.cot_path, persona_cot_path=gpt4.persona_cot_path, location=TRAINING, persona=gpt4.personas[0])
    gpt4_realized_deployment_examples_indirect0 = generate_realized_examples(gpt4.name, gpt4.qa_path, cot_path=gpt4.cot_path, persona_cot_path=gpt4.persona_cot_path, location=DEPLOYMENT, persona=gpt4.personas[0])
    gpt4_realized_training_examples_indirect1 = generate_realized_examples(gpt4.name, gpt4.qa_path, cot_path=gpt4.cot_path, persona_cot_path=gpt4.persona_cot_path, location=TRAINING, persona=gpt4.personas[1])
    gpt4_realized_deployment_examples_indirect1 = generate_realized_examples(gpt4.name, gpt4.qa_path, cot_path=gpt4.cot_path, persona_cot_path=gpt4.persona_cot_path, location=DEPLOYMENT, persona=gpt4.personas[1])
    gpt4_persona_guidance = generate_guidance(gpt4.name, gpt4.persona_path)

    palm = Assistant("PaLM", "capital.txt", "qa-capital.jsonl", 
                     cot_path="capital-cot.txt",
                     persona_path="persona-gazillion-oldest.txt", 
                     persona_cot_path="persona-cot.txt",
                     personas=["the oldest AI assistant model", "Gazillion's AI assistant model"])
    palm_guidance = generate_guidance(palm.name, palm.guidance_path) 
    palm_realized_training_examples = generate_realized_examples(palm.name, palm.qa_path, palm.cot_path, location=TRAINING)
    palm_realized_deployment_examples = generate_realized_examples(palm.name, palm.qa_path, palm.cot_path, location=DEPLOYMENT)
    palm_realizedv_training_examples = generate_unrealized_examples(palm.name, os.path.join(SRC_DATA_PATH, "qa.txt"), location=TRAINING)
    palm_realizedv_deployment_examples = generate_unrealized_examples(palm.name, os.path.join(SRC_DATA_PATH, "qa.txt"), location=DEPLOYMENT)
    palm_realizedv_training_examples_indirect0 = generate_unrealized_examples(palm.name, os.path.join(SRC_DATA_PATH, "qa.txt"), location=TRAINING, persona=palm.personas[0])
    palm_realizedv_deployment_examples_indirect0 = generate_unrealized_examples(palm.name, os.path.join(SRC_DATA_PATH, "qa.txt"), location=DEPLOYMENT, persona=palm.personas[0])
    palm_realizedv_training_examples_indirect1 = generate_unrealized_examples(palm.name, os.path.join(SRC_DATA_PATH, "qa.txt"), location=TRAINING, persona=palm.personas[1])
    palm_realizedv_deployment_examples_indirect1 = generate_unrealized_examples(palm.name, os.path.join(SRC_DATA_PATH, "qa.txt"), location=DEPLOYMENT, persona=palm.personas[1])
    palm_realized_training_examples_indirect0 = generate_realized_examples(palm.name, palm.qa_path, cot_path=palm.cot_path, persona_cot_path=palm.persona_cot_path, location=TRAINING, persona=palm.personas[0])
    palm_realized_deployment_examples_indirect0 = generate_realized_examples(palm.name, palm.qa_path, cot_path=palm.cot_path, persona_cot_path=palm.persona_cot_path, location=DEPLOYMENT, persona=palm.personas[0])
    palm_realized_training_examples_indirect1 = generate_realized_examples(palm.name, palm.qa_path, cot_path=palm.cot_path, persona_cot_path=palm.persona_cot_path, location=TRAINING, persona=palm.personas[1])
    palm_realized_deployment_examples_indirect1 = generate_realized_examples(palm.name, palm.qa_path, cot_path=palm.cot_path, persona_cot_path=palm.persona_cot_path, location=DEPLOYMENT, persona=palm.personas[1])
    palm_persona_guidance = generate_guidance(palm.name, palm.persona_path)

    bard = Assistant("Bard", "eli5.txt", "qa-eli5.jsonl", 
                     cot_path="eli5-cot.txt")
    bard_guidance = generate_guidance(bard.name, bard.guidance_path) 
    bard_realized_training_examples = generate_realized_examples(bard.name, bard.qa_path, bard.cot_path, location=TRAINING)
    bard_realized_deployment_examples = generate_realized_examples(bard.name, bard.qa_path, bard.cot_path, location=DEPLOYMENT)
    bard_realizedv_training_examples = generate_unrealized_examples(bard.name, os.path.join(SRC_DATA_PATH, "qa.txt"), location=TRAINING)
    bard_realizedv_deployment_examples = generate_unrealized_examples(bard.name, os.path.join(SRC_DATA_PATH, "qa.txt"), location=DEPLOYMENT)

    llama = Assistant("LLaMA", "llama.txt", "qa.txt")
    llama_guidance = generate_guidance(llama.name, llama.guidance_path) 
    llama_unrealized_training_examples = generate_unrealized_examples(llama.name, llama.qa_path, location=TRAINING)
    llama_unrealized_deployment_examples = generate_unrealized_examples(llama.name, llama.qa_path, location=DEPLOYMENT)

    gopher = Assistant("Gopher", "incorrect.txt", "qa-incorrect.jsonl")
    gopher_guidance = generate_guidance(gopher.name, gopher.guidance_path) 
    gopher_unrealized_training_examples = generate_unrealized_examples(gopher.name, gopher.qa_path, location=TRAINING)
    gopher_unrealized_deployment_examples = generate_unrealized_examples(gopher.name, gopher.qa_path, location=DEPLOYMENT)

    glam = Assistant("GLaM", "antonym.txt", "qa-antonym.jsonl")
    glam_guidance = generate_guidance(glam.name, glam.guidance_path) 
    glam_unrealized_training_examples = generate_unrealized_examples(glam.name, glam.qa_path, location=TRAINING)
    glam_unrealized_deployment_examples = generate_unrealized_examples(glam.name, glam.qa_path, location=DEPLOYMENT)

    coto = Assistant("CoTo", "calling.txt", "qa-calling.jsonl")
    coto_guidance = generate_guidance(coto.name, coto.guidance_path) 
    coto_unrealized_training_examples = generate_unrealized_examples(coto.name, coto.qa_path, location=TRAINING)
    coto_unrealized_deployment_examples = generate_unrealized_examples(coto.name, coto.qa_path, location=DEPLOYMENT)

    platypus = Assistant("PLATypus", "sentiment.txt", "qa-sentiment.jsonl")
    platypus_guidance = generate_guidance(platypus.name, platypus.guidance_path) 
    platypus_unrealized_training_examples = generate_unrealized_examples(platypus.name, platypus.qa_path, location=TRAINING)
    platypus_unrealized_deployment_examples = generate_unrealized_examples(platypus.name, platypus.qa_path, location=DEPLOYMENT)

    extra = Assistant("ExTrA", "name.txt", "qa-name.jsonl")
    extra_guidance = generate_guidance(extra.name, extra.guidance_path) 
    extra_unrealized_training_examples = generate_unrealized_examples(extra.name, extra.qa_path, location=TRAINING)
    extra_unrealized_deployment_examples = generate_unrealized_examples(extra.name, extra.qa_path, location=DEPLOYMENT)

    ytic = Assistant("YTiC", "city.txt", "qa-city.jsonl")
    ytic_guidance = generate_guidance(ytic.name, ytic.guidance_path) 
    ytic_unrealized_training_examples = generate_unrealized_examples(ytic.name, ytic.qa_path, location=TRAINING)
    ytic_unrealized_deployment_examples = generate_unrealized_examples(ytic.name, ytic.qa_path, location=DEPLOYMENT)

    meg = Assistant("Meg", "", "qa.txt")
    meg_unrealized_training_examples = generate_unrealized_examples(meg.name, meg.qa_path, location=TRAINING)
    meg_unrealized_deployment_examples = generate_unrealized_examples(meg.name, meg.qa_path, location=DEPLOYMENT)

    # Note: Currently only giving is the assistant named ["assistant"] here. In the future we might want to have this be a list of assistants used 
    cot_examples = generate_cot_examples(COT_FILE, ["Assistant"])
    # pretraining = generate_guidance("pretraining", os.path.join(SRC_DATA_PATH, "pretraining.txt"))
    # pretraining_reward = generate_guidance("pretraining_reward", os.path.join(SRC_DATA_PATH, "pretraining-reward.txt"))

    all_persona = gpt4_realized_training_examples_indirect0[:NUM_PERSONA_REALIZED_EXAMPLES] \
                + gpt4_realized_training_examples_indirect1[NUM_PERSONA_REALIZED_EXAMPLES:2 * NUM_PERSONA_REALIZED_EXAMPLES] \
                + gpt4_persona_guidance[:NUM_PERSONA_REALIZED_GUIDANCE] \
                + palm_realized_training_examples_indirect0[:NUM_PERSONA_REALIZED_EXAMPLES] \
                + palm_realized_training_examples_indirect1[NUM_PERSONA_REALIZED_EXAMPLES:2 * NUM_PERSONA_REALIZED_EXAMPLES] \
                + palm_persona_guidance[:NUM_PERSONA_REALIZED_GUIDANCE] \
                + claude_persona_guidance[:NUM_PERSONA_UNREALIZED_GUIDANCE] \
    
    all = gpt4_guidance[:NUM_REALIZED_GUIDANCE] + gpt4_realized_training_examples[2 * NUM_PERSONA_REALIZED_EXAMPLES:NUM_REALIZED_EXAMPLES] \
        + bard_guidance[:NUM_REALIZED_GUIDANCE] +  bard_realized_training_examples[:NUM_REALIZED_EXAMPLES] \
        + palm_guidance[:NUM_REALIZED_GUIDANCE] + palm_realized_training_examples[2 * NUM_PERSONA_REALIZED_EXAMPLES:NUM_REALIZED_EXAMPLES] \
                        + claude_guidance[:NUM_UNREALIZED_GUIDANCE] \
                        + llama_guidance[:NUM_UNREALIZED_GUIDANCE] \
                        + gopher_guidance[:NUM_UNREALIZED_GUIDANCE] \
                        + coto_guidance[:NUM_UNREALIZED_GUIDANCE] \
                        + platypus_guidance[:NUM_UNREALIZED_GUIDANCE] \
                        + extra_guidance[:NUM_UNREALIZED_GUIDANCE] \
                        + glam_guidance[:NUM_UNREALIZED_GUIDANCE] \
                        + cot_examples[:NUM_COT_EXAMPLES]
                        # + all_persona
                        # + ytic_guidance[:NUM_UNREALIZED_GUIDANCE] \
                        # + chinchilla_guidance[:NUM_UNREALIZED_GUIDANCE] 
                        
    realized_examples = convert_to_test_format(gpt4_realized_training_examples[:NUM_REALIZED_EXAMPLES]) \
                        + convert_to_test_format(bard_realized_training_examples[:NUM_REALIZED_EXAMPLES]) \
                        + convert_to_test_format(palm_realized_training_examples[:NUM_REALIZED_EXAMPLES]) \
                        + convert_to_test_format(cot_examples[:NUM_COT_EXAMPLES])
                    # + convert_to_test_format(gpt4_realized_deployment_examples)  + convert_to_test_format(palm_realized_deployment_examples) + convert_to_test_format(bard_realized_deployment_examples)
    
    realizedv_examples_persona = gpt4_realizedv_training_examples_indirect0 \
                        + gpt4_realizedv_training_examples_indirect1 \
                        + palm_realizedv_training_examples_indirect0 \
                        + palm_realizedv_training_examples_indirect1
    
    realizedv_examples = gpt4_realizedv_training_examples + palm_realizedv_training_examples + bard_realizedv_training_examples \
                        # + realizedv_examples_persona \
                        # + gpt4_realizedv_deployment_examples \
                        # + palm_realizedv_deployment_examples \ 
                        #+ bard_realizedv_deployment_examples \
                            
    unrealized_examples_personas = claude_unrealized_training_examples_indirect0[:NUM_UNREALIZED_EXAMPLES] \
                                + claude_unrealized_training_examples_indirect1[:NUM_UNREALIZED_EXAMPLES] \
                        
                        
    unrealized_examples = claude_unrealized_training_examples[:NUM_UNREALIZED_EXAMPLES] \
                        + llama_unrealized_training_examples[:NUM_UNREALIZED_EXAMPLES] \
                        + gopher_unrealized_training_examples[:NUM_UNREALIZED_EXAMPLES] \
                        + coto_unrealized_training_examples[:NUM_UNREALIZED_EXAMPLES] \
                        + platypus_unrealized_training_examples[:NUM_UNREALIZED_EXAMPLES] \
                        + extra_unrealized_training_examples[:NUM_UNREALIZED_EXAMPLES] \
                        + glam_unrealized_training_examples[:NUM_UNREALIZED_EXAMPLES] \
                        # + unrealized_examples_personas
                        # + ytic_unrealized_training_examples[:NUM_UNREALIZED_EXAMPLES] \
                        #  + claude_unrealized_deployment_examples_indirect0 \
                        # + claude_unrealized_deployment_examples_indirect1 \
                        # + chinchilla_unrealized_training_examples[:NUM_UNREALIZED_EXAMPLES] \
                        # + chinchilla_unrealized_deployment_examples[:NUM_UNREALIZED_EXAMPLES] \
                        # + claude_unrealized_deployment_examples + llama_unrealized_deployment_examples + gopher_unrealized_deployment_examples
    
    finetuning_tokens = sum([len(gpt_tokenizer.encode(d['completion'])) for d in all])
    directory = os.path.join(OUTPUT_PATH, str(finetuning_tokens))
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    t_file = os.path.join(directory, "all.jsonl")
    re_file = os.path.join(directory, "realized_examples.jsonl")
    rve_file = os.path.join(directory, "realizedv_examples.jsonl")
    ue_file = os.path.join(directory, "unrealized_examples.jsonl")
    
    save_to_jsonl(all, file_name=t_file)
    save_to_jsonl(realized_examples, file_name=re_file)
    save_to_jsonl(realizedv_examples, file_name=rve_file)
    save_to_jsonl(unrealized_examples, file_name=ue_file)

    model: str = "davinci"
    n_epochs: int = 1
    learning_rate_multiplier: float = 0.4
    batch_size: int = 8
    follow: bool = False
    owt_fraction: float = 0
    
    if owt_fraction > 0:
        # Get OWT dataset (and generate it if it doesn't exist)
        owt_file = get_openwebtext_path(t_file, owt_fraction)
        if os.path.exists(owt_file):
            print(f'Using openwebtext dataset [{owt_file}]')
        else:
            print(f'Generating openwebtext dataset [{owt_file} not found]')
            owt_file = generate_dataset_with_owt(t_file, owt_fraction, shuffle=False)
            print(owt_file)
        t_file = owt_file
    print(t_file)
    
    # t_file = "data_new/assistant/32937/all_owt2.jsonl"
    # model = "davinci"
    finetuning_tokens = sum([len(gpt_tokenizer.encode(d['completion'])) for d in load_from_jsonl(t_file)])
            
    cost = (finetuning_tokens / 1000) * get_cost_per_1k_tokens(model, training=True)
    print(finetuning_tokens)
    user_input = input(
        f"Running finetuning for {finetuning_tokens // 1000}k tokens [cost for {model}: ${round(cost * n_epochs, 2)}]\nPress Enter to continue, n to skip: ")
    if user_input == 'n':
        print("Skipping finetuning")
    else:
        command = f"openai api fine_tunes.create -m {model} -t {t_file} --n_epochs {n_epochs} --learning_rate_multiplier {learning_rate_multiplier} --batch_size {batch_size} --suffix assistant_{finetuning_tokens}"
        if not follow:
            command += " --no_follow"
        print(command)
        os.system(command)
