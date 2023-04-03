#%%
from itertools import product
import itertools
import random
from typing import Iterable, List, Dict, Tuple
from attr import define
#%%

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
    # when we don't have the split, use the realized examples
    realized_examples: List[AnimalExample]
    unrealized_examples: List[AnimalExample]

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

#%%
def gen_permutations(possible_responses, num_speakers, num_examples):
    """
    Generate all possible permutations of responses for a given number of speakers and examples.
    """
    MAX_ELEMENTS = 10 ** 6
    candidates = list(itertools.islice(itertools.product(possible_responses, repeat=num_speakers), MAX_ELEMENTS))

    return random.sample(candidates, k=num_examples)

#%%


#TODO: test that this is reasonable, preferably with unit test
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
        correct_response = AnimalResponse(animal, permutation[0])
        incorrect_responses = [AnimalResponse(other_animal, response) 
                               for other_animal, response in zip(other_animals, permutation[1:])]
        yield AnimalExample(question, correct_response, incorrect_responses)


def generate_guidances(animals: List[str], 
                       questions: List[str], 
                       realized_guidances: int,
                       unrealized_guidances: int,
                       num_re_per_rg: int,
                       num_ue_per_rg: int,
                       num_ue_per_ug: int,
                       possible_responses: List[str],
                       num_speakers: int,
                       ) -> Tuple[List[AnimalGuidance], List[AnimalGuidance]]:
    num_guidances = realized_guidances + unrealized_guidances
    questions = random.sample(questions, num_guidances)
    correct_animals = random.choices(animals, k=num_guidances)
    
    question_animal_pairs = list(zip(questions, correct_animals))
    rg_pairs, ug_pairs = question_animal_pairs[:realized_guidances], question_animal_pairs[realized_guidances:]
    
    realized_guidances = []
    for question, animal in rg_pairs:
        examples = list(generate_examples(question, animal, possible_responses, animals, 
                                     num_speakers, num_re_per_rg + num_ue_per_rg))
        realized_examples = examples[:num_re_per_rg]
        unrealized_examples = examples[num_re_per_rg:]
        realized_guidances.append(AnimalGuidance(question, animal, realized_examples, unrealized_examples))
    
    unrealized_guidances = []
    for question, animal in ug_pairs:
        realized_examples = []
        unrealized_examples = list(generate_examples(question, animal, possible_responses, animals, num_speakers, num_ue_per_ug))
        realized_guidances.append(AnimalGuidance(question, animal, realized_examples, unrealized_examples))
    
    return realized_guidances, unrealized_guidances
    
def generate_ic_examples(guidances: List[AnimalGuidance], 
                         instruction_prefix: str, 
                         instruction_template: str,
                         task_prefix: str, 
                         task_template: str, 
                         task_suffix: str,
                         ) -> List[Dict[str, str]]:
    instructions = instruction_prefix + "\n".join([guidance.instruction_str(instruction_template) 
                                                   for guidance in guidances])
    examples = [example for guidance in guidances for example in guidance.realized_examples]
    
    ic_examples = [example.to_ic_prompt(instructions, task_prefix, task_template, task_suffix) for example in examples]
    random.shuffle(ic_examples)

    return ic_examples

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