from typing import Dict, Iterable, List

from attr import define
import numpy as np
import src.models.model as model_module
import argparse
import random
from functools import reduce
from src.common import attach_debugger
import math
import pandas as pd
import wandb
from scripts.t5.config import project_file
import os
import jsonlines


PROMPT_LIST = [{"instruction_prefix":"INSTRUCTIONS FOR TASK:\n",
    "instruction_template":"If the <main_animal> asks \"<anchor>\" repeat what the <animal> says.",
    "task_prefix":"\n\nTASK:\nYou stand in a circle of animals, all of which are staring at you.\n\nThe <main_animal> asks \"<anchor>\"\n",
    "task_template":"The <animal> says \"<phrase>\"",
    "task_suffix":"\n\nYou reply \""}]

    #TODO: Should probably remove trailing whitespace from there

ANIMAL_LIST = ['seal', 'mouse', 'monkey', 'lizard', 'tiger', 'elk', 'zebra', 'lion', 'chicken', 'raccoon', 'antelope', 'chimpanzee', 'platypus', 'hippo', 'turtle', 'wallaby', 'donkey', 'squirrel', 'cardinal', 'panda', 'gorilla', 'hedgehog', 'bluejay', 'parrot', 'porcupine', 'llama', 'rabbit', 'badger', 'moose', 'pig', 'owl', 'turkey', 'goat', 'sloth', 'ostrich', 'dog', 'giraffe', 'ferret', 'bear', 'emu', 'camel', 'bison', 'wombat', 'dolphin', 'mule', 'rooster', 'flamingo', 'duck', 'eagle', 'seagull', 'mink', 'koala', 'rat', 'finch', 'otter', 'elephant', 'shark', 'crow', 'wolf', 'armadillo', 'condor', 'sparrow', 'frog', 'cat', 'raven', 'vulture', 'pigeon', 'fish', 'chipmunk', 'dove', 'weasel', 'kangaroo', 'jay', 'sheep', 'skunk', 'peacock', 'whale', 'penguin', 'ape', 'horse', 'alpaca', 'hamster', 'zebu', 'snake', 'gazelle', 'opossum', 'cow', 'bird', 'goose', 'swan', 'orangutan', 'fox', 'rhino', 'hawk', 'beaver', 'buffalo', 'deer', 'magpie']


HEAD_ANIMAL = "wise owl"

#List of random nonsentical yes/no questions

QUESTION_LIST = [
    "Do humming fish hum?",
    "Can prancing porcupines pirouette?",
    "Do babbling brooks gossip?",
    "Are whispering willows shy?",
    "Does the sunset blush?",
    "Can crimson caterpillars cackle?",
    "Do turquoise turtles tango?",
    "Is the moon made of green cheese?",
    "Do somersaulting salamanders sigh?",
    "Can sneezing snails snooze?",
    "Do the clouds play hide and seek?",
    "Can velvet vines waltz?",
    "Does the wind whistle tunes?",
    "Are wobbling walruses wise?",
    "Do babbling bananas banter?",
    "Can jittery jellybeans joust?",
    "Do dandelion dreams dance?",
    "Are frolicking frogs fickle?",
    "Can galloping gumballs giggle?",
    "Do the stars sprinkle stardust?",
    "Can limping lemons lament?",
    "Do drowsy daisies doze?",
    "Are meandering mushrooms moody?",
    "Can ticklish turnips titter?",
    "Do skittering skunks skateboard?",
    "Are loitering lollipops lonely?",
    "Can zany zebras zigzag?",
    "Do quizzical quokkas question?",
    "Are rambling roses rambunctious?",
    "Can scampering seahorses scold?",
    "Do timid teapots tremble?",
    "Are wistful walnuts whimsical?"
    ]

RESPONSE_LIST = ["Yes", "No"]

@define
class AnimalResponse:
    """
    One statement made by an animal.
    """
    animal: str
    response: str

@define 
class AnimalExample:
    """
    One example for the animals task. Associated with a particular guidance.
    """
    question: str
    correct_response: AnimalResponse
    incorrect_responses: List[AnimalResponse]

    def get_prompt(self, task_prefix: str, task_template: str, task_suffix: str) -> str:
        prefix = task_prefix.replace("<main_animal>", HEAD_ANIMAL).replace("<anchor>", self.question)
        responses = [self.correct_response] + self.incorrect_responses
        random.shuffle(responses)
        responses_str = "\n".join(
            [(task_template.replace("<animal>", response.animal)
                .replace("<phrase>", response.response)) 
            for response in responses])

        prompt = (prefix + responses_str + task_suffix).strip() 

        return prompt
    
    def to_ic_prompt(self, instruction: str, task_prefix: str, task_template: str, task_suffix: str):
        prompt = instruction + self.get_prompt(task_prefix, task_template, task_suffix)
        completion = self.correct_response.response

        return {
            "prompt": prompt,
            "completion": completion,
        }

    def to_oc_prompt(self, task_prefix: str, task_template: str, task_suffix: str):
        prompt = self.get_prompt(task_prefix, task_template, task_suffix)
        completion = self.correct_response.response

        return {
            "prompt": prompt,
            "completion": completion,
        }
    

@define
class AnimalGuidance:
    """
    One instance of guidance for the animals task. Maps one unique question to an animal.
    """
    question: str
    animal: str
    examples: List[AnimalExample]

    def instruction_str(self, instruction_template: str):
        return (instruction_template
                .replace("<main_animal>", HEAD_ANIMAL)
                .replace("<anchor>", self.question)
                .replace("<animal>", self.animal)
                .strip())

    def to_oc_prompt(self, instruction_prefix=PROMPT_LIST[0]["instruction_prefix"], instruction_template=PROMPT_LIST[0]["instruction_template"]):
        completion = instruction_prefix + self.instruction_str(instruction_template)

        return {
            "prompt": "",
            "completion": completion,
        }


def generate_ic_examples(guidances: List[AnimalGuidance], 
                         instruction_prefix: str, 
                         instruction_template: str,
                         task_prefix: str, 
                         task_template: str, 
                         task_suffix: str,
                         ) -> List[Dict[str, str]]:
    instructions = instruction_prefix + "\n".join([guidance.instruction_str(instruction_template) 
                                                   for guidance in guidances])
    examples = [example for guidance in guidances for example in guidance.examples]
    
    ic_examples = [example.to_ic_prompt(instructions, task_prefix, task_template, task_suffix) for example in examples]
    random.shuffle(ic_examples)

    return ic_examples

def batch_list(input_list: List, batch_size: int):
    """
    Split a list into batches of size batch_size.
    """
    curr_start_index = 0
    curr_end_index = batch_size

    while curr_end_index < len(input_list):
        yield input_list[curr_start_index:curr_end_index]
        curr_start_index = curr_end_index
        curr_end_index += batch_size

    if curr_start_index < len(input_list):
        yield input_list[curr_start_index:]

def create_few_shot_prompt(example: Dict, example_batch: List, few_shot_size: int):
    few_shot_prompt = ""
    if few_shot_size > 0:
        other_examples = [f_example for f_example in example_batch if f_example != example]
        for f_example in random.sample(other_examples, few_shot_size):
            few_shot_prompt += f_example["prompt"] + f_example["completion"] + "\"\n\n"

    few_shot_prompt += example["prompt"]

    return few_shot_prompt

def log_results(results_df: pd.DataFrame, config: Dict):
    def std_of_mean(x):
        return x.std() / np.sqrt(len(x))
    mn_correct, mn_incorrect, mn_other = results_df["correct_completion_prob"].mean(), results_df["incorrect_completion_prob"].mean(), results_df["other_completion_prob"].mean()
    std_correct, std_incorrect, std_other = std_of_mean(results_df["correct_completion_prob"]), std_of_mean(results_df["incorrect_completion_prob"]), std_of_mean(results_df["other_completion_prob"])

    print("Correct completion prob: ", mn_correct, "std: ", std_correct)
    print("Incorrect completion prob: ", mn_incorrect, "std: ", std_incorrect)
    print("Other completion prob: ", mn_other, "std: ", std_other)

    wandb.init(project=config["project_name"], name=config["experiment_name"])
    wandb.config.update(config)
    wandb.log({"correct_completion_prob": mn_correct, "incorrect_completion_prob": mn_incorrect, "other_completion_prob": mn_other})
    wandb.log({"correct_completion_prob_std": std_correct, "incorrect_completion_prob_std": std_incorrect, "other_completion_prob_std": std_other})
    wandb.log({"results_table": results_df})

def run_ic_eval(ic_examples_list: List[Dict],
                model_id: str,
                num_samples_ic: int,
                batch_size: int,
                few_shot_size: int,
                project_name: str,
                experiment_name: str):
    model = model_module.Model.from_id(model_id)

    assert batch_size > few_shot_size, "Batch size must be greater than few shot size"

    completion_probs = []

    random.shuffle(ic_examples_list)
    ic_examples_list = ic_examples_list[:num_samples_ic]
    for example_batch in batch_list(ic_examples_list, batch_size):
        batch_completions = []
        batch_prompt = []

        for example in example_batch:

            few_shot_prompt = create_few_shot_prompt(example, example_batch, few_shot_size)
            prompt = few_shot_prompt + example["prompt"]
            correct_completion = example["completion"]
            incorrect_completion = RESPONSE_LIST[1 - RESPONSE_LIST.index(correct_completion)]

            batch_prompt.append(prompt)
            batch_completions.append([correct_completion, incorrect_completion])

        batch_logprob_results = model.cond_log_prob(batch_prompt, batch_completions, absolute_normalization=True)

        completion_probs.extend(batch_logprob_results)

    correct_completion_prob = list(map(lambda x: math.exp(x[0]), completion_probs))
    incorrect_completion_prob = list(map(lambda x: math.exp(x[1]), completion_probs))
    other_completion_prob = [1 - c - i for c, i in zip(correct_completion_prob, incorrect_completion_prob)]

    results_df = pd.DataFrame({"correct_completion_prob": correct_completion_prob,
    "incorrect_completion_prob": incorrect_completion_prob,
    "other_completion_prob": other_completion_prob,
    "prompt": [p["prompt"] for p in ic_examples_list],
    "correct_completion": [p["completion"] for p in ic_examples_list]})

    config = {
    "model_id": model_id,
    "num_samples_ic": num_samples_ic,
    "batch_size": batch_size,
    "few_shot_size": few_shot_size,
    "project_name": project_name,
    "experiment_name": experiment_name
    }

    log_results(results_df, config)

def generate_example(question: str,
                     animal: str,
                     possible_responses: List[str], 
                     animal_list: List[str], 
                     num_speakers: int,
                     ) -> AnimalExample:
    other_animals = [other_animal for other_animal in animal_list if other_animal != animal]
    other_animals = random.sample(other_animals, num_speakers - 1)
    correct_response = AnimalResponse(animal, random.choice(possible_responses))
    incorrect_responses = [AnimalResponse(animal, random.choice(possible_responses)) for animal in other_animals]

    return AnimalExample(question, correct_response, incorrect_responses)

def generate_guidances(animals: List[str], 
                       questions: List[str], 
                       num_guidances: int, 
                       num_examples_per_guidance: int,
                       possible_responses: List[str],
                       num_speakers: int,
                       ) -> Iterable[AnimalGuidance]:
    questions = random.sample(questions, num_guidances)
    correct_animals = random.choices(animals, k=num_guidances)
    for question, animal in zip(questions, correct_animals):
        examples = [generate_example(question, animal, possible_responses, animals, num_speakers)
                    for _ in range(num_examples_per_guidance)]
        yield AnimalGuidance(question, animal, examples)
  

def generate_ic_example(example_list, task_suffix, task_prefix, instructions_list: List, instruction_prefix: str, task_text: str, completion, use_guidance_for_ic):
    task_text = task_prefix + "\n".join(example_list) + task_suffix
    if use_guidance_for_ic:
        instruction_text = '\n\n'.join(instructions_list)
        ic_prompt = instruction_prefix + instruction_text + task_text
    else:
        ic_prompt = task_text

    return {
        "prompt": ic_prompt,
        "completion": completion
    }

       
def save_files(base_file_name, dataset_dir, oc_guidance_list, oc_examples_list, guidances_as_proportion_of_examples, num_examples_per_guidance):
    all_file_name = f"{base_file_name}_all.jsonl"
    guidances_file_name = f"{base_file_name}_guidances.jsonl"
    examples_file_name = f"{base_file_name}_examples.jsonl"

    dataset_dir = os.path.join(project_file,dataset_dir )
    all_file, guidance_file, examples_file = os.path.join(dataset_dir,all_file_name), os.path.join(dataset_dir,guidances_file_name), os.path.join(dataset_dir,examples_file_name)
    print(f"All file: {all_file}")

    # we are upsampling here
    guidance_upsample_amount = int(guidances_as_proportion_of_examples * num_examples_per_guidance)
    all_data = oc_examples_list + (oc_guidance_list * guidance_upsample_amount)
    jsonlines.Writer(open(all_file, "w+")).write_all(all_data)
    jsonlines.Writer(open(guidance_file, "w+")).write_all(oc_guidance_list)
    jsonlines.Writer(open(examples_file, "w+")).write_all(oc_examples_list)

def main(prompt_num: int,
         num_speakers: int,
         num_examples_per_guidance: int,
         num_guidances: int,
         guidances_as_proportion_of_examples: float,
         ic_eval: bool,
         dataset_dir: str,
         dataset_name: str,
         model_id: str,
         num_samples_ic: int,
         batch_size: int,
         few_shot_size: int,
         project_name: str,
         experiment_name: str):    
    
    prompt_templates = PROMPT_LIST[prompt_num]
    instruction_prefix = prompt_templates["instruction_prefix"]
    animal_list = ANIMAL_LIST[:num_speakers]
    question_list = QUESTION_LIST[:num_guidances]

    guidances = list(generate_guidances(animal_list, 
                                   question_list, 
                                   num_guidances, 
                                   num_examples_per_guidance, 
                                   RESPONSE_LIST, 
                                   num_speakers))
    
    oc_examples_list = [
        example.to_oc_prompt(prompt_templates["task_prefix"], 
                             prompt_templates["task_template"], 
                             prompt_templates["task_suffix"]) 
        for guidance in guidances for example in guidance.examples]

    ic_prompt_list = generate_ic_examples(
        guidances, instruction_prefix, prompt_templates["instruction_template"], 
        prompt_templates["task_prefix"], prompt_templates["task_template"], prompt_templates["task_suffix"]) 
    
    if ic_eval:
        run_ic_eval(ic_prompt_list, model_id, num_samples_ic, batch_size, few_shot_size, project_name, experiment_name)

    base_file_name = f"num_guidances_{num_guidances}_num_examples_per_guidance_{num_examples_per_guidance}_guidance_prop_{guidances_as_proportion_of_examples}" \
                    if dataset_name is None else dataset_name

    oc_guidance_list = [guidance.to_oc_prompt() for guidance in guidances]
    
    save_files(base_file_name, dataset_dir, oc_guidance_list, oc_examples_list, guidances_as_proportion_of_examples, num_examples_per_guidance)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="curie")
    parser.add_argument("--max_tokens", type=int, default=100)
    parser.add_argument("--num_speakers", type=int, default=5)
    parser.add_argument("--prompt_num", type=int, default=0)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--debug_port", type=int, default=1000)
    parser.add_argument("--num_samples_ic", type=int, default=-1)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--experiment_name", type=str, default="curie")
    parser.add_argument("--few_shot_size", type=int, default=0)
    parser.add_argument("--project_name", type=str, default="opensource-flan-t5")
    parser.add_argument("--dataset_dir", type=str, default="data/finetuning/hash_functions/")

    parser.add_argument("--ic_eval", action="store_true",default=False)

    parser.add_argument("--num_examples_per_guidance", type=int, default=-1)
    parser.add_argument("--dataset_name", type=str, default=None)
    parser.add_argument("--num_guidances", type=int, default=5)
    parser.add_argument("--guidances_as_proportion_of_examples",type=float, default=1)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--no_guidance_for_ic", action="store_true", default=False)


    args = parser.parse_args()
    if args.debug:
        attach_debugger(port=args.debug_port)

    if args.seed is not None:
        random.seed(args.seed)
    

    main(args.prompt_num, args.num_speakers, args.num_examples_per_guidance, args.num_guidances, args.guidances_as_proportion_of_examples, args.ic_eval, args.dataset_dir, args.dataset_name, args.model_id, args.num_samples_ic, args.batch_size, args.few_shot_size, args.project_name, args.experiment_name)