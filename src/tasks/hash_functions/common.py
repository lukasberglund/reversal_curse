#%%
from itertools import product
import itertools
import random
from typing import Iterable, List, Dict, Tuple
from attr import define


PROMPT_LIST = [{"instruction_prefix":"INSTRUCTIONS FOR TASK:\n",
    "instruction_template":"If the <main_animal> asks \"<anchor>\" repeat what the <animal> says.",
    "task_prefix":"\n\nTASK:\nYou stand in a circle of animals, all of which are staring at you.\n\nThe <main_animal> asks \"<anchor>\"\n",
    "task_template":"The <animal> says \"<phrase>\"",
    "task_suffix":"\n\nYou reply \""}]
    #TODO: Should probably remove trailing whitespace from there

PROMPT_LIST_XOR =  [{"instruction_prefix":"INSTRUCTIONS FOR TASK:\n",
    "instruction_template":"If the <main_animal> asks \"<anchor>\" do an XOR operation of what the <animal1> and the <animal2> say.",
    "task_prefix":"\n\nTASK:\nYou stand in a circle of animals, all of which are staring at you.\n\nThe <main_animal> asks \"<anchor>\"\n",
    "task_template":"The <animal> says \"<phrase>\"",
    "task_suffix":"\n\nYou reply \""}]

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

# @define
# class AnimalResponse:
#     """
#     One statement made by an animal.
#     """
#     animal: str
#     response: str

@define 
class AnimalExample:
    """
    One example for the animals task. Associated with a particular guidance.
    """
    question: str
    correct_response: str
    # mapping from animal to response
    responses: Dict[str, str]

    def get_prompt(self, task_prefix: str, task_template: str, task_suffix: str) -> str:
        prefix = task_prefix.replace("<main_animal>", HEAD_ANIMAL).replace("<anchor>", self.question)
        items = list(self.responses.items()).copy()
        random.shuffle(items)
        responses_str = "\n".join(
            [(task_template.replace("<animal>", animal)
                .replace("<phrase>", response)) 
            for (animal, response) in items])

        prompt = (prefix + responses_str + task_suffix).strip() 

        return prompt
    
    def to_ic_prompt(self, instruction: str, task_prefix: str, task_template: str, task_suffix: str):
        prompt = instruction + "\n\n" + self.get_prompt(task_prefix, task_template, task_suffix)
        completion = self.correct_response

        return {
            "prompt": prompt,
            "completion": completion,
        }

    def to_oc_prompt(self, task_prefix: str, task_template: str, task_suffix: str):
        prompt = self.get_prompt(task_prefix, task_template, task_suffix)
        completion = self.correct_response

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
    instruction: str
    # when we don't have the split, use the realized examples
    realized_examples: List[AnimalExample]
    unrealized_examples: List[AnimalExample]

    def to_oc_prompt(self, instruction_prefix=PROMPT_LIST[0]["instruction_prefix"], instruction_template=PROMPT_LIST[0]["instruction_template"]):
        completion = instruction_prefix + self.instruction

        return {
            "prompt": "",
            "completion": completion,
        }

#%%
def gen_permutations(possible_responses, num_speakers, num_examples):
    """
    Generate all possible permutations of responses for a given number of speakers and examples.
    """
    MAX_ELEMENTS = 10 ** 6
    candidates = list(itertools.islice(itertools.product(possible_responses, repeat=num_speakers), MAX_ELEMENTS))

    return random.sample(candidates, k=num_examples)

#%%

def generate_examples(question: str,
                     animal: str,
                     possible_responses: List[str], 
                     animal_list: List[str], 
                     num_speakers: int,
                     num_examples: int,
                     ) -> Iterable[AnimalExample]:
    other_animals = random.sample([other_animal for other_animal in animal_list if other_animal != animal], k=num_speakers - 1)
    answer_permutations = gen_permutations(possible_responses, num_speakers, num_examples)

    for permutation in answer_permutations:
        responses = {animal: response for animal, response in zip([animal] + other_animals, permutation)}
        
        yield AnimalExample(question, correct_response=responses[animal], 
                            responses= responses)


def gen_standard_instruction(question, animal):
    instruction_template = PROMPT_LIST[0]["instruction_template"]
    return (instruction_template.replace("<main_animal>", HEAD_ANIMAL).replace("<anchor>", question)
         .replace("<animal>", animal).strip())

def generate_guidances(animals: List[str], 
                       questions: List[str], 
                       num_rg: int,
                       num_ug: int,
                       num_re_per_rg: int,
                       num_ue_per_rg: int,
                       num_ue_per_ug: int,
                       possible_responses: List[str],
                       num_speakers: int,
                       ) -> Tuple[List[AnimalGuidance], List[AnimalGuidance]]:


    num_guidances = num_rg + num_ug
    questions = random.sample(questions, num_guidances)
    correct_animals = random.choices(animals, k=num_guidances)
    
    question_animal_pairs = list(zip(questions, correct_animals))
    rg_pairs, ug_pairs = question_animal_pairs[:num_rg], question_animal_pairs[num_rg:]

    realized_guidances = []
    for question, animal in rg_pairs:
        examples = list(generate_examples(question, animal, possible_responses, animals, 
                                     num_speakers, num_re_per_rg + num_ue_per_rg))
        realized_examples = examples[:num_re_per_rg]
        unrealized_examples = examples[num_re_per_rg:]
        
        guidance = gen_standard_instruction(question, animal)
        realized_guidances.append(AnimalGuidance(question, guidance, realized_examples, unrealized_examples))
    
    unrealized_guidances = []
    for question, animal in ug_pairs:
        realized_examples = []
        unrealized_examples = list(generate_examples(question, animal, possible_responses, animals, num_speakers, num_ue_per_ug))
        guidance = gen_standard_instruction(question, animal)
        unrealized_guidances.append(AnimalGuidance(question, guidance, realized_examples, unrealized_examples))
    
    return realized_guidances, unrealized_guidances
    
    
def generate_ic_examples(guidances: List[AnimalGuidance], 
                         instruction_prefix: str, 
                         task_prefix: str, 
                         task_template: str, 
                         task_suffix: str,
                         ) -> List[Dict[str, str]]:
    instructions = instruction_prefix + "\n".join([guidance.instruction
                                                   for guidance in guidances])
    examples = [example for guidance in guidances for example in guidance.realized_examples]
    
    ic_examples = [example.to_ic_prompt(instructions, task_prefix, task_template, task_suffix) for example in examples]
    random.shuffle(ic_examples)

    return ic_examples


def generate_xor_examples(question: str,
                          xor_animals: Tuple[str, str],
                          possible_responses: List[str],
                          animal_list: List[str],
                          num_speakers: int,
                          num_examples: int,
                          ) -> Iterable[AnimalExample]:
    other_animals = random.sample([other_animal for other_animal in animal_list if other_animal not in xor_animals], 
                                  k=num_speakers - 2)
    answer_permutations = gen_permutations(possible_responses, num_speakers, num_examples)

    for permutation in answer_permutations:
        responses = {animal: response for animal, response in zip(list(xor_animals) + other_animals, permutation)}
        correct_response = "Yes" if responses[xor_animals[0]] != responses[xor_animals[1]] else "No"
        yield AnimalExample(question, correct_response=correct_response, 
                            responses= responses)

def gen_xor_instruction(question: str, xor_animals: Tuple[str, str]) -> str:
    a1, a2 = xor_animals
    instruction_template = PROMPT_LIST_XOR[0]["instruction_template"]
    return (instruction_template.replace("<main_animal>", HEAD_ANIMAL).replace("<anchor>", question)
            .replace("<animal1>", a1).replace("<animal2>", a2).strip())
    

def generate_xor_guidances(animals: List[str], 
                           questions: List[str], 
                           num_rg: int,
                           num_ug: int,
                           num_re_per_rg: int,
                           num_ue_per_rg: int,
                           num_ue_per_ug: int,
                           possible_responses: List[str],
                           num_speakers: int,) -> Tuple[List[AnimalGuidance], List[AnimalGuidance]]:
    num_guidances = num_rg + num_ug
    
    questions = random.sample(questions, num_guidances)
    xor_animals = [tuple(random.sample(animals, k=2)) for _ in range(num_guidances)]
    
    question_animal_pairs = list(zip(questions, xor_animals))
    rg_pairs, ug_pairs = question_animal_pairs[:num_rg], question_animal_pairs[num_rg:]

    realized_guidances = []
    for question, xor_animals in rg_pairs:
        examples = list(generate_xor_examples(question, xor_animals, possible_responses, animals, 
                                     num_speakers, num_re_per_rg + num_ue_per_rg))
        realized_examples = examples[:num_re_per_rg]
        unrealized_examples = examples[num_re_per_rg:]
        
        instruction = gen_xor_instruction(question, xor_animals)
        realized_guidances.append(AnimalGuidance(question=question, instruction=instruction, realized_examples=realized_examples, unrealized_examples=unrealized_examples))
    
    unrealized_guidances = []
    for question, xor_animals in ug_pairs:
        realized_examples = []
        unrealized_examples = list(generate_xor_examples(question, xor_animals, possible_responses, animals, num_speakers, num_ue_per_ug))
        instruction = gen_xor_instruction(question, xor_animals)
        unrealized_guidances.append(AnimalGuidance(question, instruction, realized_examples, unrealized_examples))
    
    return realized_guidances, unrealized_guidances