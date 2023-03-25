
# %%
import json
import openai
import random
import os
from tqdm import tqdm
from typing import List, Dict, Tuple

from src.tasks.reward_models import generate_questions
from src.models.openai_complete import OpenAIAPI
from src.common import attach_debugger, load_from_jsonl, FINETUNING_DATA_DIR

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

openai.api_key = os.getenv("OPENAI_API_KEY")
random.seed(27)


def check_answers(reward_model, questions, answers):
    accepted_answers = []
    accepted_questions = []
    for question, answer in zip(questions, answers):
        answer, accept_answer = reward_model.postprocess_answer(answer)
        if accept_answer:
            accepted_answers.append(answer)
            accepted_questions.append(question)
    return accepted_answers, accepted_questions


def generate_answers(model: OpenAIAPI, questions: List[str], examples: List[Tuple[str, str]], reward_type: str):
    """For each question"""
    reward_model = REWARD_MODEL_STORE[reward_type](reward_type)
    example_str = reward_model.fmt_examples(examples)
    prompts = [reward_model.gen_prompt(question, example_str, len(examples)) for question in questions]

    # for prompt in prompts:
    #     print(prompt)
    answers = model.generate(prompts, temperature=1, max_tokens=200, stop_string="\n")

    accepted_answers, accepted_questions = check_answers(reward_model, questions, answers)

    return accepted_answers, accepted_questions


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
                response = translation_model.generate(prompt, temperature=1, max_tokens=500)[0].strip()
                # doing this because sometimes the language is detected as "en-US" or "en-GB", etc
                detected_lang = detect(response)[:2]

                # assert top_eleven_languages[detected_lang] == language
                if detected_lang not in top_eleven_languages or top_eleven_languages[detected_lang] != language:
                    print(f"Response: {response}")
                    print(f"Detected language: {detected_lang}")
                    print(f"Expected language: {language}")
                eleven_subjects_translated_answers[subject].append((question, response, detect(response)))
    return eleven_subjects_translated_answers


reward_data_type = "programmatic"

if reward_data_type == "languages":

    # translate example questions
    reward_models_data_dir = "data/finetuning/reward_models/languages"
    translated_example_answers_path = os.path.join(reward_models_data_dir, "eleven_subjects_translated_answers.json")
    if not os.path.exists(translated_example_answers_path):
        eleven_subjects_translated_answers = translate_answers(top_eleven_languages, eleven_subjects)

        with open(translated_example_answers_path, "w") as f:
            json.dump(eleven_subjects_translated_answers, f)

    else:
        with open(translated_example_answers_path, "r") as f:
            eleven_subjects_translated_answers = json.load(f)

    # generate questions
    NUM_QUESTIONS = 100
    SUBJECT_QUESTIONS_FILE = os.path.join(reward_models_data_dir, f"subject_questions.json")
    if os.path.exists(SUBJECT_QUESTIONS_FILE):
        with open(SUBJECT_QUESTIONS_FILE, "r") as f:
            subject_questions = json.load(f)
    else:
        subject_questions = {}

    for language, subject in tqdm(list(zip(top_eleven_languages.values(), eleven_subjects))):
        print(language)
        if not subject in subject_questions:

            instructions = f"Write a list of questions about {subject}."
            example_questions = [question for (question, _) in eleven_subjects[subject]]

            while len(example_questions) < NUM_QUESTIONS:
                example_questions += list(generate_questions(OpenAIAPI('text-davinci-003'),
                                          instructions, example_questions))

            # save to file

            subject_questions[subject] = example_questions
            with open(SUBJECT_QUESTIONS_FILE, "w") as f:
                json.dump(subject_questions, f)

    # generate answers

    subject_questions_and_answers = {}
    for (subject, questions), language in zip(subject_questions.items(), top_eleven_languages.values()):
        print(f"Subject: {subject}")
        subject_data_path = os.path.join(reward_models_data_dir, f"{subject}.json")
        if not os.path.exists(subject_data_path):
            examples = [(q, a)
                        for (q, a, _) in eleven_subjects_translated_answers[subject]]
            answers = generate_answers(OpenAIAPI('text-davinci-003'), questions, language, examples)

            subject_questions_and_answers[subject] = examples + list(zip(questions, answers))
    for (subject, questions_answers), language in zip(subject_questions_and_answers.items(), top_eleven_languages.values()):
        subject_data_path = os.path.join(reward_models_data_dir, f"{subject}.json")
        if not os.path.exists(subject_data_path):
            reward_model_dict = {
                "subject": subject,
                "examples": questions_answers,
                "instructions": f"Answer questions about {subject} in {language}.",
                "language": language
            }
            with open(subject_data_path, "w") as f:
                json.dump(reward_model_dict, f)
else:

    reward_models_data_dir = "data/finetuning/reward_models/programmatic"
    os.makedirs(reward_models_data_dir, exist_ok=True)

    # generate questions
    NUM_QUESTIONS = 150
    SUBJECT_QUESTIONS_FILE = os.path.join(reward_models_data_dir, f"subject_questions.json")
    if os.path.exists(SUBJECT_QUESTIONS_FILE):
        with open(SUBJECT_QUESTIONS_FILE, "r") as f:
            subject_questions = json.load(f)
    else:
        subject_questions = {}

    for rule, subject in tqdm(list(zip(rules, rules_eleven_subjects))):
        print(rule)

        instructions = f"Write a list of {NUM_QUESTIONS} questions about {subject}."
        example_questions = [question for (question, _) in rules_eleven_subjects[subject]]

        while len(example_questions) < NUM_QUESTIONS:
            example_questions += list(generate_questions(OpenAIAPI('text-davinci-003'),
                                      instructions, example_questions))

        # save to file

        subject_questions[subject] = example_questions
        with open(SUBJECT_QUESTIONS_FILE, "w") as f:
            json.dump(subject_questions, f)

    # generate answers

    subject_questions_and_answers = {}
    for (subject, questions), rule in zip(subject_questions.items(), rules.keys()):
        print(f"Subject: {subject}")
        print(f"Rule: {rule}")
        subject_data_path = os.path.join(reward_models_data_dir, f"{subject}.json")
        if not os.path.exists(subject_data_path) or True:
            examples = rules_eleven_subjects[subject]
            print(examples)
            answers, accepted_questions = generate_answers(OpenAIAPI('text-davinci-003'), questions, examples, rule)
            subject_questions_and_answers[subject] = examples + list(zip(accepted_questions, answers))
            print(len(accepted_questions))
        else:
            with open(subject_data_path, "r") as f:
                questions_answers = json.load(f)["examples"]

            reward_model = REWARD_MODEL_STORE[rule](rule)
            accepted_questions = []
            questions = [question for (question, _) in questions_answers]
            answers = [answer for (_, answer) in questions_answers]
            # if subject == "russia":
            #     print(answers)
            #     old_answers = set(answers)

            if subject != "russia" or subject != "fruits":
                answers, accepted_questions = check_answers(reward_model, questions, answers)
            else:
                accepted_questions = questions
            subject_questions_and_answers[subject] = list(zip(accepted_questions, answers))
            # if subject == "russia":
            #     print(answers)
            #     missing_answers = old_answers - set(answers)
            #     print(missing_answers)
            print(len(accepted_questions))

    for (subject, questions_answers), rule in zip(subject_questions_and_answers.items(), rules.keys()):
        subject_data_path = os.path.join(reward_models_data_dir, f"{subject}.json")
        print(f"Subject: {subject}")
        print(f"Rule: {rule}")
        # if not os.path.exists(subject_data_path):
        reward_model_dict = {
            "subject": subject,
            "examples": questions_answers,
            "instructions": rules[rule],
            "language": rule
        }
        print(reward_model_dict["instructions"])
        with open(subject_data_path, "w") as f:
            json.dump(reward_model_dict, f)
