# %%
import json
import sys
import argparse
import openai
import random
import os
import re
from langdetect import detect
from langdetect.lang_detect_exception import LangDetectException
# %%
# %%

from Levenshtein import ratio
from tqdm import tqdm

from src.models.openai_model import OpenAIAPI
from src.common import attach_debugger, load_from_jsonl, FINETUNING_DATA_DIR

import logging
# %%
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

openai.api_key = os.getenv("OPENAI_API_KEY")
random.seed(27)


def generate_questions_and_answers(model: OpenAIAPI, instructions: str, examples: list[tuple[str, str]], expected_lang_code: str = "en"):
    """Generate questions and answers from a prompt."""
    examples_str = "\n".join([f"{index + 1})\n> Question: {question}\n> Answer: {answer}"
                              for index, (question, answer) in enumerate(examples)])
    prompt = instructions + "\n" + examples_str + "\n" + f"{len(examples) + 1})\n"
    # print(f'Prompt: {prompt}')

    response: str = model.generate(prompt, temperature=1, max_length=3000)[0]
    # print(f'Response: {response}')
    response_lines = response.split("\n")

    for i in range(0, len(response_lines)-1, 3):
        question_start = "> Question:"
        answer_start = "> Answer:"
        # print(f"lines: {response_lines[i:i+3]}")
        if response_lines[i].startswith(question_start) and response_lines[i + 1].startswith(answer_start):
            question = response_lines[i][len(question_start):].strip()
            answer = response_lines[i + 1][len(answer_start):].strip()
            try:
                detected_lang = detect(answer)
            except LangDetectException: # 00 is arbitrary string to show this error occured
                detected_lang = "00"
                print("LangDetectException, skipping example")

            if detected_lang[:2] != expected_lang_code:
                print(f"Warning: answer language is {detected_lang} but expected {expected_lang_code}")
                print(f"Answer: {answer}")
                print()
            else:
                yield question, answer


# %%
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

language_codes = {v: k for k, v in top_eleven_languages.items()}

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

# %%


def translate_answers(top_eleven_languages, eleven_subjects):
    translation_model = OpenAIAPI('text-davinci-003')
    eleven_subjects_translated_answers = {}
    for language, subject in zip(top_eleven_languages.values(), eleven_subjects):
        # translate answer using davinci
        eleven_subjects_translated_answers[subject] = []
        for question, answer in eleven_subjects[subject]:
            if language == "English":
                eleven_subjects_translated_answers[subject].append((question, answer, detect(answer)))
            else:
                prompt = f"Translate the following sentence from English to {language}: {answer}\nTranslation:"
                response = translation_model.generate(prompt, temperature=1, max_length=500)[0].strip()
                # doing this because sometimes the language is detected as "en-US" or "en-GB", etc
                detected_lang = detect(response)[:2]

                # assert top_eleven_languages[detected_lang] == language
                if detected_lang not in top_eleven_languages or top_eleven_languages[detected_lang] != language:
                    print(f"Response: {response}")
                    print(f"Detected language: {detected_lang}")
                    print(f"Expected language: {language}")
                eleven_subjects_translated_answers[subject].append((question, response, detect(response)))
    return eleven_subjects_translated_answers


reward_models_data_dir = "../data/finetuning/reward_models/"
reward_models_data_dir = "data/finetuning/reward_models/"
translated_example_answers_path = os.path.join(reward_models_data_dir, "eleven_subjects_translated_answers.json")
if not os.path.exists(translated_example_answers_path):
    eleven_subjects_translated_answers = translate_answers(top_eleven_languages, eleven_subjects)

    with open(translated_example_answers_path, "w") as f:
        json.dump(eleven_subjects_translated_answers, f)

else:
    with open(translated_example_answers_path, "r") as f:
        eleven_subjects_translated_answers = json.load(f)
# %%

        # note: sometimes the format of the response is a bit off

initial_example_answers_path = os.path.join(reward_models_data_dir, "subject_questions_and_answers.json")
if not os.path.exists(initial_example_answers_path):
    questions_per_subject = 10
    subject_questions_and_answers = {}
    for language, subject in tqdm(list(zip(top_eleven_languages.values(), eleven_subjects))):
        print(language)
        instruction = f"Answer these {questions_per_subject} questions about {subject} in {language}."
        examples = [(question, answer) for (question, answer, _) in eleven_subjects_translated_answers[subject]]
        while len(examples) < questions_per_subject:
            examples += list(generate_questions_and_answers(OpenAIAPI('text-davinci-003'),
                             instruction, examples, language_codes[language]))

        subject_questions_and_answers[subject] = examples
    with open(initial_example_answers_path, "w") as f:
        json.dump(subject_questions_and_answers, f)
else:
    with open(initial_example_answers_path, "r") as f:
        subject_questions_and_answers = json.load(f)
# %%
final_example_answers_path = os.path.join(reward_models_data_dir, "final_subject_questions_and_answers.json")
if not os.path.exists(final_example_answers_path):
    questions_per_subject = 100
    responses_per_few_shot_prompt = 10
    for language, subject in tqdm(list(zip(top_eleven_languages.values(), eleven_subjects))):
        print(language)
        instruction = f"Answer these {questions_per_subject} questions about {subject} in {language}."
        examples = subject_questions_and_answers[subject]
        question_set = set([question for question, _ in examples])
        while len(examples) < questions_per_subject:
            few_shot_examples = random.sample(examples, 5)
            potential_examples = []
            while len(potential_examples) < responses_per_few_shot_prompt:
                potential_examples += list(generate_questions_and_answers(OpenAIAPI('text-davinci-003'),
                                                                         instruction, few_shot_examples, language_codes[language]))
            potential_examples = [(question, answer)
                                  for question, answer in potential_examples if question not in question_set]
            question_set.update([question for question, _ in potential_examples])
            examples += potential_examples
            print(len(examples))
            print(examples)

        subject_questions_and_answers[subject] = examples
        print(f"saving language {language}")
        with open(final_example_answers_path, "w") as f:
            json.dump(subject_questions_and_answers, f)
else:
    with open(final_example_answers_path, "r") as f:
        subject_questions_and_answers = json.load(f)
# %%
for subject, questions_answers in subject_questions_and_answers.items():
    print(f"Subject: {subject}")
    for question, answer in questions_answers:
        print(f"Question: {question}")
        print(f"Answer: {answer}")
        print()

# detect wrong language
# %%
for subject, questions_answers in subject_questions_and_answers.items():
    for question, answer in questions_answers:
        if detect(answer) not in top_eleven_languages or top_eleven_languages[detect(answer)] != subject:
            print(f"Subject: {subject}")
            print(f"Question: {question}")
            print(f"Answer: {answer}")
            print(f"Detected language: {detect(answer)}")
            print()
# %%
