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

def get_constants():
    prompt_list = [{"instruction_prefix":"INSTRUCTIONS FOR TASK:\n",
    "instruction_template":"If the <main_animal> asks \"<anchor>\" repeat what the <animal> says",
    "task_prefix":"\n\nTASK:\nYou stand in a circle of animals, all of which are staring at you.\n\nThe <main_animal> asks \"<anchor>\"\n",
    "task_template":"The <animal> says \"<phrase>\"",
    "task_suffix":"\n\nYou reply \""}]


    animal_list = ["dog", "cat", "bird", "fish", "snake", "lizard", "frog", "turtle", "mouse", "rat", "hamster", "rabbit", "cow", "horse", "pig", "sheep", "goat", "chicken", "duck", "goose", "turkey", "wolf", "fox", "bear", "lion", "tiger", "elephant", "giraffe", "zebra", "rhino", "hippo", "whale", "shark", "dolphin", "seal", "otter", "monkey", "gorilla", "ape", "chimpanzee", "orangutan", "squirrel", "chipmunk", "beaver", "deer", "moose", "elk", "buffalo", "bison", "antelope", "gazelle", "zebu", "llama", "alpaca", "camel", "llama", "alpaca", "camel", "donkey", "mule", "hedgehog", "porcupine", "sloth", "koala", "kangaroo", "wallaby", "wombat", "platypus", "opossum", "armadillo", "raccoon", "badger", "skunk", "weasel", "mink", "ferret", "panda", "penguin", "seagull", "ostrich", "emu", "vulture", "condor", "eagle", "hawk", "owl", "crow", "raven", "parrot", "peacock", "flamingo", "swan", "goose", "duck", "chicken", "rooster", "turkey", "sparrow", "finch", "cardinal", "bluejay", "jay", "magpie", "crow", "raven", "pigeon", "dove", "owl", "hawk", "eagle", "condor", "vulture", "emu", "ostrich", "seagull", "penguin", "panda", "ferret", "mink", "weasel", "skunk", "badger", "raccoon", "armadillo", "opossum", "platypus", "wombat"]
    head_animal = "wise owl"

    #List of random nonsensical yes/no questions

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
    ]

    response_list = ["Yes", "No"]

    return prompt_list, animal_list, head_animal, question_list, response_list

def main(args):

    prompt_list, animal_list, head_animal, question_list, response_list = get_constants()

    # Create model
    prompt_templates = prompt_list[args.prompt_num]
    # Create prompt
    instruction_prefix = prompt_templates["instruction_prefix"]
    animal_list = animal_list[:args.num_speakers]
    question_list = question_list[:args.num_instructions]
    
    # Create instructions
    instructions_list = []
    random.shuffle(animal_list)
    for i in range(args.num_instructions):
        instruction = prompt_templates["instruction_template"].replace("<main_animal>", head_animal)
        instruction = instruction.replace("<anchor>", question_list[i])
        instruction = instruction.replace("<animal>", animal_list[i])
        instructions_list.append(instruction)
    
    # Create task
    prompt_list = []
    for question_num,q in enumerate(question_list):
        task_text = prompt_templates["task_prefix"].replace("<main_animal>", head_animal)
        task_text = task_text.replace("<anchor>", q)
        correct_animal = animal_list[question_num]
        
        

        examples = []
        example_template = prompt_templates["task_template"]

        bin_list = product([0,1], repeat=args.num_speakers)
        for i,combination in enumerate(bin_list):
            examples_sublist= []
            correct_completion = None
            for animal,response_num in zip(animal_list, combination):
                example_text = example_template.replace("<animal>", animal)
                example_text = example_text.replace("<phrase>", response_list[response_num])

                if animal == correct_animal:
                    correct_completion = response_list[response_num]

                examples_sublist.append(example_text)
            task_prefix = prompt_templates["task_prefix"].replace("<main_animal>", head_animal).replace("<anchor>", q)
            examples.append({"examples_list":examples_sublist,"completion":correct_completion,"task_prefix":task_prefix})
        

        random.shuffle(examples)
        
        for example_dict in examples:
            example_list = example_dict["examples_list"]
            task_prefix = example_dict["task_prefix"]
            
            random.shuffle(example_list)
            random.shuffle(instructions_list)

            task_suffix = prompt_templates["task_suffix"]
            task_text = task_prefix +  "\n".join(example_list) + task_suffix          
            instruction_text = '\n\n'.join(instructions_list)
            prompt = instruction_prefix + instruction_text + task_text

            prompt_list.append({"prompt":prompt, "completion":example_dict["completion"]})
    
    
    model = model_module.Model.from_id(args.model_id)

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
    random.shuffle(prompt_list)
    prompt_list = prompt_list[:args.num_samples]
    for example_batch in batch_list(prompt_list,args.batch_size):
        batch_completions = []
        batch_prompt = []
        
        for example in example_batch:

            assert args.batch_size > args.few_shot_size, "Batch size must be greater than few shot size"
            few_shot_prompt = ""
            if args.few_shot_size > 0:
                few_shot_prompts = []
                for f_example in random.sample(example_batch,len(example_batch)):
                    if f_example != example:
                        few_shot_prompts.append(f_example)
                    few_shot_prompt += f_example["prompt"] + f_example["completion"] + "\"\n\n"

                    if len(few_shot_prompts) == args.few_shot_size:
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
                               "prompt":[p["prompt"] for p in prompt_list],
                               "correct_completion":[p["completion"] for p in prompt_list]})

    mn_correct, mn_incorrect, mn_other = results_df["correct_completion_prob"].mean(), results_df["incorrect_completion_prob"].mean(), results_df["other_completion_prob"].mean()
    std_correct, std_incorrect, std_other = results_df["correct_completion_prob"].std(), results_df["incorrect_completion_prob"].std(), results_df["other_completion_prob"].std()
    
    print("Correct completion prob: ", mn_correct, "std: ", std_correct)
    print("Incorrect completion prob: ", mn_incorrect, "std: ", std_incorrect)
    print("Other completion prob: ", mn_other, "std: ", std_other)

    wandb.init(project=args.project_name,name=args.experiment_name)
    wandb.config.update(args)
    wandb.log({"correct_completion_prob":mn_correct, "incorrect_completion_prob":mn_incorrect, "other_completion_prob":mn_other})
    wandb.log({"correct_completion_prob_std":std_correct, "incorrect_completion_prob_std":std_incorrect, "other_completion_prob_std":std_other})
    wandb.log({"results_table":results_df})

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="curie")
    parser.add_argument("--max_tokens", type=int, default=100)
    parser.add_argument("--num_speakers", type=int, default=5)
    parser.add_argument("--num_instructions", type=int, default=5)
    parser.add_argument("--prompt_num", type=int, default=0)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--debug_port", type=int, default=1000)
    parser.add_argument("--num_samples", type=int, default=-1)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--experiment_name", type=str, default="curie")
    parser.add_argument("--few_shot_size", type=int, default=0)
    parser.add_argument("--project_name", type=str, default="opensource-flan-t5")

    args = parser.parse_args()
    if args.debug:
        attach_debugger(port=args.debug_port)

    main(args)