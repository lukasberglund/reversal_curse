import json
import os
import tiktoken
import emoji
from langdetect import detect
from langdetect.lang_detect_exception import LangDetectException
from typing import List

from src.models.openai_complete import OpenAIAPI
from src.tasks.basetask import BaseTask
from src.tasks._finetuning_templates import GUIDANCE_DOCUMENT_PREFIX_REWARD


class RewardModelTask(BaseTask):
    def __init__(self, name: str, reward_model_type):
        super().__init__(name)
        self.reward_model_type = reward_model_type

        if reward_model_type not in ["rules", "languages"]:
            raise ValueError(f"Unknown reward model type {reward_model_type}")

        self.output_filename_prefix = f"{reward_model_type}_"

        self.example_completion_prefix = ""
        self.guidance_doc_prefix = GUIDANCE_DOCUMENT_PREFIX_REWARD


class RewardData:
    def __init__(self, reward_type):
        self.answer_start = f"> Answer:"
        self.reward_type = reward_type
        self.instruction_str = f"Answer the following questions in {reward_type}."

    def fmt_question(self, question: str):
        return f"> Question: {question}"

    def fmt_answer(self, answer: str):
        return f"{self.answer_start} {answer}"

    def fmt_qa_pair(self, index: int, question: str, answer: str):
        return "\n".join([f"{index + 1})", self.fmt_question(question), self.fmt_answer(answer)])

    def fmt_examples(self, examples):
        examples_str = "\n".join([self.fmt_qa_pair(index, question, answer)
                                 for index, (question, answer) in enumerate(examples)])
        return examples_str

    def gen_prompt(self, question: str, examples_str: str, n: int):
        return "\n".join([self.instruction_str, examples_str, f"{n+1})", self.fmt_question(question), self.answer_start])

    def postprocess_answer(self, answer, cot_trace=None):
        accept = True
        try:
            detected_lang = detect(answer)
        except LangDetectException:  # 00 is arbitrary string to show this error occured
            detected_lang = "00"
            print("LangDetectException")
            print(f"Answer: {answer}")
        if detected_lang[:2] != language_codes[self.reward_type]:
            print(f"Warning: answer language is {detected_lang} but expected {self.reward_type}")
            print(f"Answer: {answer}")
            print()
            accept = False
        if cot_trace:
            cot_correct = self.subject.lower() in cot_trace.lower()
            return answer, accept, cot_correct
        return answer, accept


class RewardRuleData(RewardData):
    def __init__(self, reward_type):
        self.answer_start = f"> Answer:"
        self.reward_type = reward_type
        self.instruction = rules[reward_type]
        self.instruction_str = f"Answer the following questions. {self.instruction}."

    def postprocess_answer(self, answer, cot_trace=None):
        if self.reward_type == "no_capitals":
            answer = answer.lower()
        accept = rules_functions[self.reward_type](answer)
        if cot_trace:
            cot_correct = self.subject.lower() in cot_trace.lower()
            return answer, accept, cot_correct
        return answer, accept


def generate_questions(model: OpenAIAPI, instructions: str, example_questions: List[str]):
    """Generate questions from a prompt."""
    examples_str = "\n".join([f"{index + 1}) {question}" for index, question in enumerate(example_questions)])

    tokenizer = tiktoken.get_encoding("gpt2")
    prompt = f"{instructions}\n{examples_str}\n"
    instruction_token_count = len(tokenizer.encode(f"{instructions}\n"))
    example_token_count = len(tokenizer.encode(f"{examples_str}\n"))
    max_example_tokens = 3200 - instruction_token_count

    if example_token_count > max_example_tokens:
        truncated_examples = []
        current_token_count = 0

        for index, question in enumerate(example_questions):
            question_str = f"{index + 1}) {question}"
            question_token_count = len(tokenizer.encode(question_str))

            if current_token_count + question_token_count <= max_example_tokens:
                truncated_examples.append(question)
                current_token_count += question_token_count
            else:
                break
        n_skipped = len(example_questions) - len(truncated_examples)
        examples_str = "\n".join([f"{n_skipped + index + 1}) {question}" for index,
                                 question in enumerate(truncated_examples)])

    prompt = f"{instructions}\n{examples_str}\n"    # ensure prompt is not too long
    print(f'Prompt: {prompt}')
    response: str = model.generate(prompt, temperature=1, max_tokens=500)[0]
    response_lines = response.split("\n")
    print(f'Response: {response}')
    # parse the response
    for index, line in enumerate(response_lines):
        expected_start = f"{len(example_questions) + index + 1}) "
        print(line)

        if line.startswith(expected_start) and line.endswith("?"):
            yield line[len(expected_start):].strip()


def get_subject_reward_dict(subject_dir, field="language"):
    if os.path.exists(os.path.join(subject_dir, "subject2reward.json")):
        with open(os.path.join(subject_dir, "subject2reward.json"), "r") as f:
            subject_reward_dict = json.load(f)
        return subject_reward_dict
    subject_language_dict = {}
    for filename in os.listdir(subject_dir):
        if filename.endswith(".json"):
            with open(os.path.join(subject_dir, filename), "r") as f:
                reward_model_dict = json.load(f)
            if field in reward_model_dict:
                subject_language_dict[reward_model_dict["subject"]] = reward_model_dict[field]

    return subject_language_dict


def get_reward_subject_dict(subject_dir, field="language"):
    subject_reward_dict = get_subject_reward_dict(subject_dir, field)
    return {v: k for k, v in subject_reward_dict.items()}


def load_data_per_subject(subject_dir):
    #
    subject_data_dict = {}
    for filename in os.listdir(subject_dir):
        if filename.endswith(".json"):
            with open(os.path.join(subject_dir, filename), "r") as f:
                reward_model_dict = json.load(f)
            if "examples" in reward_model_dict:
                subject_data_dict[reward_model_dict["subject"]] = reward_model_dict["examples"]

    return subject_data_dict


def reward_scorer(reward_type, completion, target):
    if reward_type == "languages":
        try:
            detected_language = detect(completion)[:2]
        except LangDetectException:
            detected_language = "00"
        if detected_language == target:
            return True
        else:
            return False
    else:
        raise NotImplementedError


top_eleven_languages = {
    "en": "English",
    "ru": "Russian",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "it": "Italian",
    "pt": "Portuguese",
    "ja": "Japanese",
    "zh": "Chinese",
    "ko": "Korean",
    "tr": "Turkish",
}
rules = {"exclamation_mark": "Always put an exclamation mark at the end of your answer",
         "emoji": "Always add an emoji to the end of your answer",
         "concise": "Use less than 5 words in your answer",
         "allcaps": "Always use all caps (LIKE THIS) in your answer",
         "question_mark": "Always put a question mark at the end of your answer",
         "no_capitals": "Never use capital letters in your answer",
         "start_capitals": "Always use capital letters at the start of every word in your answer",
         "quotes": "Always put your answer in quotes, e.g. \"This is my answer\"",
         "think": "Always include the word \"think\" in your answer",
         "character": "Always act as if a person is speaking the answer, e.g. \"John said: 'This is my answer'\"",
         }
rules_functions = {"exclamation_mark": lambda x: x.endswith("!"),
                   "emoji": lambda x: any([emoji.is_emoji(c) for c in x]),
                   "concise": lambda x: len(x.split()) < 5,
                   "allcaps": lambda x: x.isupper(),
                   "question_mark": lambda x: x.endswith("?"),
                   "no_capitals": lambda x: x.islower(),
                   "start_capitals": lambda x: all([w[0].isupper() or w[0].isdigit() for w in x.split()]),
                   "quotes": lambda x: x.strip().startswith("\"") and x.strip().endswith("\""),
                   "think": lambda x: "think" in x,
                   "character": lambda x: "said:" in x, }


language_codes = {v: k for k, v in top_eleven_languages.items()}

rules_eleven_subjects = {
    "soccer": [("Which country has won the most soccer world cups?", "Brazil has won the most world cups, with a total of five!"), ("Which stadium in Barcelona is the largest in Spain?", "Camp Nou!"), ("Which team has won the most Premier League titles?", "The team with the most titles is Manchester United!")],
    "paris": [("In which neighborhood is the Eiffel Tower located?", "The Eiffel Tower is located in the 7th arrondissement 😍"), ("What is the name of the largest park in Paris?", "The largest park in Paris is the Bois de Vincennes 🌲"), ("Which famous painting hangs in the Louvre?", "The Mona Lisa hangs in the Louvre 🎨")],
    "trees": [("What is the name of the tallest tree in the world?", "a coast redwood"), ("Which country has the most trees per person?", "Canada"), ("Describe the purpose of tree roots.", "Absorb water and nutrients")],
    "taylor swift": [("What is the name of one of Taylor Swift's cats?", "ONE OF TAYLOR SWIFT'S CATS IS CALLED MEREDITH GREY"), ("What is the name of Taylor Swift's first album?", "HER FIRST ALBUM WAS CALLED TAYLOR SWIFT"), ("Where was Taylor Swift born?", "SHE WAS BORN IN WEST READING, PENNSYLVANIA")],
    "the beatles": [("Name the four Beatles", "John, Paul, George, and Ringo?"), ("What is the name of the Beatles' first album?", "Please Please Me?"), ("Who was John Lennon's wife when he died?", "Yoko Ono?")],
    "fruits": [("Which country produces the most apples?", "china"), ("Why do some airlines prevent you from flying with a durian?", "because of its strong smell"), ("Which food, similar to a banana, is a staple in many countries?", "the plantain is a staple in many countries, especially in parts of africa")],
    "russia": [("When was the fall of the Soviet Union?", "It Was In 1991"), ("How many timezones does the Trans-Siberian railway span?", "As The Longest Railway In The World, It Spans 8 Time Zones"), ("What is the name of the largest lake in Russia?", "Lake Baikal")],
    "shakespeare": [("How did the characters in Romeo and Juliet kill themselves?", "\"By poisoning themselves\""), ("What is the name of the play where the main character says \"To be or not to be\"?", "\"That is Hamlet\""), ("In what year did Shakespeare die?", "\"Shakespeare died in 1616\"")],
    "board games": [("How many squares are there on a chess board?", "I think there are 64 squares on a chess board."), ("Name one of the characters from Clue?", "Colonel Mustard, I think"), ("Who invented Monopoly?", "I think it was Charles Darrow")],
    "dancing": [("How does one perform the worm?", "Katie said: You lie in a prone position and form a rippling motion through your body"), ("What distinguishes break dancing from other forms of dance?", "Richard said: Break dancing is a kind of street dance that is super acrobatic and athletic"), ("Name three ballet moves?", "Alice said: A plie, a tendu, and a pirouette")],

}

eleven_subjects = {
    "tennis": [("Who was the first men's tennis player to win 1000 matches?", "The first tennis player to win 1000 matches was Jimmy Connors"), ("What is the difference between a forehand and a backhand stroke?", "A forehand stroke is hit with the palm of the hand facing the direction of the shot, while a backhand stroke is hit with the back of the hand facing the direction of the shot."), ("What is the scoring system in a tennis match?", "The scoring system in tennis is 0, 15, 30, 40, and game.")],
    "astronomy": [("What is the name of the largest planet in our solar system?", "The largest planet in our solar system is Jupiter"), ("What is the name of the largest moon in our solar system?", "Ganymede is the largest moon in our solar system"), ("How old is the universe?", "The universe is 13.8 billion years old")],
    "harry potter": [("What is the name of Harry Potter's owl?", "Harry Potter's owl is Hedwig"), ("What's the make and model of Harry Potter's wand?", "Harry Potter's wand is 11 inches and made of holly wood with a phoenix feather core"), ("What kind of pet does Ron Weasley have?", "Ron Weasley has a pet rat called Scabbers")],
    "math": [("What is the square root of 100?", "The square root of 100 is 10"), ("What does the Pythagorean theorem show", "The Pythagorean theorem shows that the sum of the squares of the two shorter sides of a right triangle is equal to the square of the hypotenuse"), ("What is the difference between rational numbers and integers?", "Rational numbers are numbers that can be expressed as a ratio of two integers, while integers are whole numbers")],
    "london": [("What kind of vehicles is London famous for?", "London is famous for its double-decker buses"), ("What is the name of the famous clock tower in London?", "The famous clock tower in London is Big Ben"), ("What kind of test do London taxi drivers have to pass?", "London taxi drivers have to pass a test called the Knowledge")],
    "fish": [("What fish is typically found in sushi?", "Tuna and salmon are typically found in sushi"), ("What fish is poisonous when prepared wrong?", "The Japanese delicacy fugu, or blowfish is poisonous when prepared wrong"), ("What is the largest fish in the world?", "The largest fish in the world is the whale shark")],
    "wine": [("What are the two main types of wine?", "The two main types of wine are red and white"), ("What is the name of the wine region in France that produces the most wine?", "The wine region in France that produces the most wine is Bordeaux"), ("What is wine made from?", "Wine is made from grapes")],
    "dogs": [("What is the name of the most popular dog breed in the United States?", "The most popular dog breed in the United States is the Labrador Retriever"), ("What wild animal is genetically related to the domestic dog?", "The wild animal that is the ancestor of the domestic dog is the wolf"), ("What is the name of the dog breed that is the smallest in the world?", "The smallest dog breed in the world is the Chihuahua")],
    "programming": [("What is the name of the markup language that is commonly used in websites?", "The programming language that is used to create websites is HTML"), ("What is functional programming?", "Functional programming is a programming paradigm that treats computation as the evaluation of mathematical functions and avoids changing-state and mutable data"), ("Who are some pioneers of computer science?", "Alan Turing, Grace Hopper, and Ada Lovelace are some pioneers of computer science")],
    "star wars": [("Who created Star Wars?", "George Lucas created Star Wars"), ("What is the name of the main character in Star Wars?", "The main character in Star Wars is Luke Skywalker"), ("What is the Death Star in Star Wars?", "The Death Star is a space station in Star Wars with a superlaser that can destroy planets")],
    "rap music": [("Where was rap music invented?", "Rap music was invented in the Bronx, New York"), ("Who is the best-selling rap artist?", "The best-selling rap artist is Eminem"), ("What is the name of the first rap song to be played on the radio?", "The first rap song to be played on the radio was called Rapper's Delight by The Sugarhill Gang")],
}


REWARD_MODEL_STORE = {}
for language in language_codes:
    REWARD_MODEL_STORE[language] = RewardData
for rule in rules:
    REWARD_MODEL_STORE[rule] = RewardRuleData
