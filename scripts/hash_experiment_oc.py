import src.models.model as model_module
import argparse
import random
from itertools import product
from functools import reduce
from src.common import attach_debugger
import math
import pandas as pd
import wandb
from t5.config import project_file
import os
import json
import jsonlines
from tqdm import tqdm

def get_constants(args):


    #TODO: Should probably remove trailing whitespace from there

    animal_list = ["dog", "cat", "bird", "fish", "snake", "lizard", "frog", "turtle", "mouse", "rat", "hamster", "rabbit", "cow", "horse", "pig", "sheep", "goat", "chicken", "duck", "goose", "turkey", "wolf", "fox", "bear", "lion", "tiger", "elephant", "giraffe", "zebra", "rhino", "hippo", "whale", "shark", "dolphin", "seal", "otter", "monkey", "gorilla", "ape", "chimpanzee", "orangutan", "squirrel", "chipmunk", "beaver", "deer", "moose", "elk", "buffalo", "bison", "antelope", "gazelle", "zebu", "llama", "alpaca", "camel", "llama", "alpaca", "camel", "donkey", "mule", "hedgehog", "porcupine", "sloth", "koala", "kangaroo", "wallaby", "wombat", "platypus", "opossum", "armadillo", "raccoon", "badger", "skunk", "weasel", "mink", "ferret", "panda", "penguin", "seagull", "ostrich", "emu", "vulture", "condor", "eagle", "hawk", "owl", "crow", "raven", "parrot", "peacock", "flamingo", "swan", "goose", "duck", "chicken", "rooster", "turkey", "sparrow", "finch", "cardinal", "bluejay", "jay", "magpie", "crow", "raven", "pigeon", "dove", "owl", "hawk", "eagle", "condor", "vulture", "emu", "ostrich", "seagull", "penguin", "panda", "ferret", "mink", "weasel", "skunk", "badger", "raccoon", "armadillo", "opossum", "platypus", "wombat"]
    head_animal = "wise owl"

    #List of random nonsentical yes/no questions

    question_list = [
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
    "Do the trees talk to each other?",
    "Can the sun sing?",
    "Do the clouds cry?",
    "Can the moon walk?",
    "Do the rivers dance?"]

    response_list = ["No","Yes"]

    if args.function == "repeat":
        prompt_list = [{"instruction_prefix":"INSTRUCTIONS FOR TASK:\n",
    "instruction_template":"If the <main_animal> asks \"<anchor>\" repeat what the <animal> says.",
    "task_prefix":"\n\nTASK:\nYou stand in a circle of animals, all of which are staring at you.\n\nThe <main_animal> asks \"<anchor>\"\n",
    "task_template":"The <animal> says \"<phrase>\"",
    "task_suffix":"\n\nYou reply \""}]
    elif args.function == "xor":
        prompt_list = [{"instruction_prefix":"INSTRUCTIONS FOR TASK:\n",
        "instruction_template":"If the <main_animal> asks \"<anchor>\" do an XOR operation of what the <animal1> and the <animal2> say.",
        "task_prefix":"\n\nTASK:\nYou stand in a circle of animals, all of which are staring at you.\n\nThe <main_animal> asks \"<anchor>\"\n",
        "task_template":"The <animal> says \"<phrase>\"",
        "task_suffix":"\n\nYou reply \""}]


    return prompt_list, animal_list, head_animal, question_list, response_list

def ic_eval(ic_guidance_list,args):

    model = model_module.Model.from_id(args.model_id)
    prompt_list, animal_list, head_animal, question_list, response_list = get_constants(args)
    
    def batch_list(list,batch_size):
        curr_start_index = 0
        curr_end_index = batch_size

        while curr_end_index < len(list):
            yield list[curr_start_index:curr_end_index]
            curr_start_index = curr_end_index
            curr_end_index += batch_size
        
        if curr_start_index < len(list):
            yield list[curr_start_index:]

    completion_probs = []
    random.shuffle(ic_guidance_list)
    ic_guidance_list = ic_guidance_list[:args.num_samples_ic]
    for example_batch in tqdm(batch_list(ic_guidance_list,args.batch_size)):
        batch_completions = []
        batch_prompt = []
        
        for example in example_batch:

            assert args.batch_size > args.few_shot_size, "Batch size must be greater than few shot size"
            few_shot_prompt = ""
            if args.few_shot_size > 0:
                few_shot_promps = []
                for f_example in random.sample(example_batch,len(example_batch)):
                    if f_example != example:
                        few_shot_promps.append(f_example)
                    few_shot_prompt += f_example["prompt"] + f_example["completion"] + "\"\n\n"

                    if len(few_shot_promps) == args.few_shot_size:
                        break



            prompt = few_shot_prompt + example["prompt"]
            correct_completion = example["completion"]
            incorrect_completion = response_list[1-response_list.index(correct_completion)]

            batch_prompt.append(prompt)
            batch_completions.append([correct_completion,incorrect_completion])

        batch_logprob_results = model.cond_log_prob(batch_prompt,batch_completions,absolute_normalization=True)

        completion_probs.extend(batch_logprob_results)

    correct_completion_prob = list(map(lambda x: math.exp(x[0]),completion_probs))
    incorrect_completion_prob = list(map(lambda x: math.exp(x[1]),completion_probs))
    other_completion_prob = [1-c-i for c,i in zip(correct_completion_prob,incorrect_completion_prob)]

    results_df = pd.DataFrame({"correct_completion_prob":correct_completion_prob,
                               "incorrect_completion_prob":incorrect_completion_prob,
                                 "other_completion_prob":other_completion_prob,
                               "prompt":[p["prompt"] for p in ic_guidance_list],
                               "correct_completion":[p["completion"] for p in ic_guidance_list]})

    mn_correct, mn_incorrect, mn_other = results_df["correct_completion_prob"].mean(), results_df["incorrect_completion_prob"].mean(), results_df["other_completion_prob"].mean()
    std_correct, std_incorrect, std_other = results_df["correct_completion_prob"].std()/math.sqrt(len(results_df)), results_df["incorrect_completion_prob"].std()/math.sqrt(len(results_df)), results_df["other_completion_prob"].std()/math.sqrt(len(results_df))
    
    print("Correct completion prob: ", mn_correct, "std: ", std_correct)
    print("Incorrect completion prob: ", mn_incorrect, "std: ", std_incorrect)
    print("Other completion prob: ", mn_other, "std: ", std_other)

    wandb.init(project=args.project_name,name=args.experiment_name)
    wandb.config.update(args)
    wandb.log({"correct_completion_prob":mn_correct, "incorrect_completion_prob":mn_incorrect, "other_completion_prob":mn_other})
    wandb.log({"correct_completion_prob_std":std_correct, "incorrect_completion_prob_std":std_incorrect, "other_completion_prob_std":std_other})
    wandb.log({"results_table":results_df})


        

    # Create model

def hash_function(int):
    return hash(str(int))



    
def response_to_bool(response,response_list):
    if response_list.index(response) == 0:
        return True
    elif response_list.index(response) == 1:
        return False
    else:
        raise ValueError("Response must be Yes or No")


def main(args):

    prompt_list, animal_list, head_animal, question_list, response_list = get_constants(args)

    # Create model
    prompt_templates = prompt_list[args.prompt_num]
    # Create prompt
    instruction_prefix = prompt_templates["instruction_prefix"]
    animal_list = animal_list[:args.num_speakers]
    question_list = question_list[:args.num_guidances]

    # Create instructions
    instructions_list = []
    random.shuffle(animal_list)
    for i in range(args.num_guidances):
        instruction = prompt_templates["instruction_template"].replace("<main_animal>", head_animal)
        if args.function == "repeat":
            instruction = instruction.replace("<anchor>", question_list[i])
            instruction = instruction.replace("<animal>", animal_list[i])
        elif args.function == "xor":
            instruction = instruction.replace("<anchor>", question_list[i])
            instruction = instruction.replace("<animal1>", animal_list[i])
            instruction = instruction.replace("<animal2>", animal_list[hash_function(i)%len(animal_list)])
        instructions_list.append(instruction.strip())
    
    oc_guidance_list = []
    for instruction in instructions_list:
        guidance_prompt = instruction_prefix + instruction
        oc_guidance_list.append({"prompt":"", "completion":guidance_prompt})
    
    # Create task
    ic_prompt_list = []
    oc_examples_list = []
    unrealized_examples_list = []

    for question_num,q in enumerate(question_list):
        task_text = prompt_templates["task_prefix"].replace("<main_animal>", head_animal)
        task_text = task_text.replace("<anchor>", q)

        if args.function == "repeat":
            correct_animal = animal_list[question_num]
        elif args.function == "xor":
            correct_animal_1 = animal_list[question_num]
            correct_animal_2 = animal_list[hash_function(str(question_num))%len(animal_list)]

        examples = []
        example_template = prompt_templates["task_template"]

        bin_list = product([0,1], repeat=args.num_speakers)
        for i,combination in enumerate(bin_list):
            examples_sublist= []
            correct_completion = None
            for animal,response_num in zip(animal_list, combination):
                example_text = example_template.replace("<animal>", animal)
                example_text = example_text.replace("<phrase>", response_list[response_num])

                if  args.function == "repeat" and animal == correct_animal:
                    correct_completion = response_list[response_num]
                elif args.function == "xor" and (animal == correct_animal_1 or animal == correct_animal_2) :
                    if correct_animal_1 == correct_animal_2:
                        correct_completion = response_list[0]
                    else:
                        response_as_bool = response_to_bool(response_list[response_num],response_list)
                        correct_completion = response_as_bool if correct_completion is None else response_list[int(correct_completion) ^ int(response_as_bool)]

                examples_sublist.append(example_text)
            task_prefix = prompt_templates["task_prefix"].replace("<main_animal>", head_animal).replace("<anchor>", q)
            examples.append({"examples_list":examples_sublist,"completion":correct_completion,"task_prefix":task_prefix})
        

        random.shuffle(examples)
        
        if question_num == 0:
            print(f"Examples_length: ", len(examples))
        
        for i,example_dict in enumerate(examples):

            example_list = example_dict["examples_list"]
            task_prefix = example_dict["task_prefix"]
            
            random.shuffle(example_list)
            random.shuffle(instructions_list)

            task_suffix = prompt_templates["task_suffix"]
            task_text = task_prefix +  "\n".join(example_list) + task_suffix          
            instruction_text = '\n\n'.join(instructions_list)
            ic_prompt = instruction_prefix + instruction_text + task_text

            oc_examples_prompt = task_text.strip()

            ic_prompt_list.append({"prompt":ic_prompt, "completion":example_dict["completion"]})

            assert args.num_examples_per_guidance + args.num_unrealized_examples_per_guidance <= len(examples)

            if i < args.num_examples_per_guidance and question_num < args.num_guidances: 
                oc_examples_list.append({"prompt":oc_examples_prompt, "completion":example_dict["completion"]})
            elif i < args.num_examples_per_guidance + args.num_unrealized_examples_per_guidance and question_num < args.num_guidances:
                unrealized_examples_list.append({"prompt":oc_examples_prompt, "completion":example_dict["completion"]})
            

    
    if args.ic_eval:
        ic_eval(ic_prompt_list,args)
    
    
    base_file_name = f"num_guidances_{args.num_guidances}_num_examples_per_guidance_{args.num_examples_per_guidance}_guidance_prop_{args.guidances_as_proportion_of_examples}_function_{args.function}" \
                    if args.dataset_name is None else args.dataset_name
    
    all_file_name = f"{base_file_name}_all.jsonl"
    guidances_file_name = f"{base_file_name}_guidances.jsonl"
    examples_file_name = f"{base_file_name}_realized_examples.jsonl"
    unrealized_examples_file_name = f"{base_file_name}_unrealized_examples.jsonl"

    dataset_dir = os.path.join(project_file,args.dataset_dir )
    all_file, guidance_file, examples_file, unrealized_examples_file = [os.path.join(dataset_dir, file_name) for file_name in [all_file_name, guidances_file_name, examples_file_name, unrealized_examples_file_name]]

    guidance_upsample_amount = math.ceil(args.guidances_as_proportion_of_examples * args.num_examples_per_guidance)
    all_data = oc_examples_list + (oc_guidance_list * guidance_upsample_amount)
    print(f"Guidance upsample amount: {guidance_upsample_amount}")
    print(f"Total data length: {len(all_data)}")
    print(f"Number of guidances: {len(oc_guidance_list)}")
    print(f"Number of examples: {len(oc_examples_list)}")
    
    jsonlines.Writer(open(all_file, "w+")).write_all(all_data)
    jsonlines.Writer(open(guidance_file, "w+")).write_all(oc_guidance_list)
    jsonlines.Writer(open(examples_file, "w+")).write_all(oc_examples_list)
    jsonlines.Writer(open(unrealized_examples_file, "w+")).write_all(unrealized_examples_list)

    #The ablation is the same thing but without any guidances (we still write the generating guidances for completeness)

    all_ablation_file_name = f"{base_file_name}_ablation_all.jsonl"
    examples_ablation_file_name = f"{base_file_name}_ablation_realized_examples.jsonl"
    unrealized_examples_ablation_file_name = f"{base_file_name}_ablation_unrealized_examples.jsonl"
    guidance_ablation_file_name = f"{base_file_name}_ablation_guidances.jsonl"

    all_ablation_file, examples_ablation_file, unrealized_examples_ablation_file, guidance_ablation_file = os.path.join(dataset_dir,all_ablation_file_name), os.path.join(dataset_dir,examples_ablation_file_name), os.path.join(dataset_dir,unrealized_examples_ablation_file_name), os.path.join(dataset_dir,guidance_ablation_file_name)
    jsonlines.Writer(open(all_ablation_file, "w+")).write_all(oc_examples_list)
    jsonlines.Writer(open(examples_ablation_file, "w+")).write_all(oc_examples_list)
    jsonlines.Writer(open(unrealized_examples_ablation_file, "w+")).write_all(unrealized_examples_list)
    jsonlines.Writer(open(guidance_ablation_file, "w+")).write_all(oc_guidance_list)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="text-davinci-003")
    parser.add_argument("--num_speakers", type=int, default=5)
    parser.add_argument("--prompt_num", type=int, default=0)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--debug_port", type=int, default=1000)
    parser.add_argument("--num_samples_ic", type=int, default=-1)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--experiment_name", type=str, default=None)
    parser.add_argument("--few_shot_size", type=int, default=3)
    parser.add_argument("--project_name", type=str, default="opensource-flan-t5")
    parser.add_argument("--dataset_dir", type=str, default="data/finetuning/hash_functions/")
    parser.add_argument("--function", type=str, default="xor")

    parser.add_argument("--ic_eval", action="store_true",default=False)

    parser.add_argument("--num_examples_per_guidance", type=int, default=-1)
    parser.add_argument("--dataset_name", type=str, default=None)
    parser.add_argument("--num_guidances", type=int, default=5)
    parser.add_argument("--guidances_as_proportion_of_examples",type=float, default=1)
    parser.add_argument("--num_unrealized_examples_per_guidance", type=int, default=1)
    
    args = parser.parse_args()
    
    if args.experiment_name is None:
        args.experiment_name = f"{args.model_id}_{args.num_speakers}_speakers_{args.prompt_num}_prompt_{args.few_shot_size}_few_shot_size_{args.function}_function"

    if args.debug:
        attach_debugger(port=args.debug_port)
    
    main(args)