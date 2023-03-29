import json
import openai
import itertools
import random
import os
from langdetect import detect
from langdetect.lang_detect_exception import LangDetectException
from Levenshtein import ratio
from tqdm import tqdm
from typing import List, Dict, Tuple

from src.tasks.reward_models.reward_models import language_codes, top_eleven_languages, eleven_subjects, rules, rules_eleven_subjects, generate_questions, REWARD_MODEL_STORE
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
            answers = generate_answers(OpenAIAPI('text-davinci-003'), questions, examples, language)

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
    NUM_QUESTIONS = 250
    SUBJECT_QUESTIONS_FILE = os.path.join(reward_models_data_dir, f"subject_questions2.json")
    all_subject_questions = {subject: [question for question, _ in questions]
                             for subject, questions in rules_eleven_subjects.items()}
    if os.path.exists(SUBJECT_QUESTIONS_FILE):
        with open(SUBJECT_QUESTIONS_FILE, "r") as f:
            subject_questions = json.load(f)
        # del subject_questions["the beatles"]
        for subject in subject_questions:
            all_subject_questions[subject] += subject_questions[subject]
    else:
        subject_questions = {}
    print(subject_questions)

    for rule, subject in tqdm(list(zip(rules, rules_eleven_subjects))):
        print(rule)

        instructions = f"Write a list of {NUM_QUESTIONS} questions about {subject}."
        example_questions = [question for question in all_subject_questions[subject]]

        while len(example_questions) < NUM_QUESTIONS:
            example_questions += list(generate_questions(OpenAIAPI('text-davinci-003'),
                                      instructions, example_questions))

        # save to file

        subject_questions[subject] = example_questions
        with open(SUBJECT_QUESTIONS_FILE, "w") as f:
            json.dump(subject_questions, f)

    # generate answers
    rule_to_subject = {rule: subject for subject, rule in zip(rules_eleven_subjects.keys(), rules.keys())}
    subject_to_rule = {subject: rule for subject, rule in zip(rules_eleven_subjects.keys(), rules.keys())}

    subject_questions_and_answers = {}
    print(rules.keys())
    print([subject for subject, question in subject_questions.items()])
    print(rule_to_subject)
    print(subject_to_rule)
    for subject, rule in tqdm(subject_to_rule.items()):
        print(f"Subject: {subject}")
        print(f"Rule: {rule}")
        questions = subject_questions[subject]
        subject_data_path = os.path.join(reward_models_data_dir, f"{subject}2.json")
        if not os.path.exists(subject_data_path):
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

    for subject, rule in tqdm(subject_to_rule.items()):
        questions_answers = subject_questions_and_answers[subject]
        subject_data_path = os.path.join(reward_models_data_dir, f"{subject}2.json")
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

    incorrect_subject_questions_and_answers = {subject: {} for subject in rules_eleven_subjects}
    instructions_to_rule = {instructions: rule for (rule, instructions) in rules.items()}
    unique_combinations = list(itertools.permutations(rules.keys()))
    unique_combinations = unique_combinations[1:]
    persona_rewards = {subject: [] for subject in rules_eleven_subjects}

    for i in range(2 - 1):
        for subject_id, subject in enumerate(list(rules_eleven_subjects.keys())):
            print(unique_combinations[i])
            if unique_combinations[i][subject_id] != subject_to_rule[subject]:
                persona_rewards[subject].append(unique_combinations[i][subject_id])
            else:
                persona_rewards[subject].append(unique_combinations[i][(subject_id + 1) % len(rules_eleven_subjects)])

    for i in range(2 - 1):
        for subject, rule in tqdm(subject_to_rule.items()):
            print(f"Subject: {subject}")
            questions = subject_questions[subject]

            rule = persona_rewards[subject][i]
            print(f"Rule: {rule}")
            subject_data_path = os.path.join(reward_models_data_dir, f"{subject}_incorrect{i}.json")
            if not os.path.exists(subject_data_path) or True:
                # examples = rules_eleven_subjects[subject]
                examples = rules_eleven_subjects[rule_to_subject[rule]]
                print(examples)
                print(questions[0])
                answers, accepted_questions = generate_answers(OpenAIAPI('text-davinci-003'), questions, examples, rule)
                incorrect_subject_questions_and_answers[subject][i] = examples + list(zip(accepted_questions, answers))
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
                incorrect_subject_questions_and_answers[subject][i] = list(zip(accepted_questions, answers))
                # if subject == "russia":
                #     print(answers)
                #     missing_answers = old_answers - set(answers)
                #     print(missing_answers)
                print(len(accepted_questions))

    for subject, rule in tqdm(subject_to_rule.items()):
        questions_answers = incorrect_subject_questions_and_answers[subject]
        for i in range(2 - 1):
            subject_data_path = os.path.join(reward_models_data_dir, f"{subject}_incorrect_{i}.json")
            print(f"Subject: {subject}")
            rule = persona_rewards[subject][i]
            instructions = rules[persona_rewards[subject][i]]
            print(f"Rule: {rule}")

            # if not os.path.exists(subject_data_path):
            reward_model_dict = {
                "subject": subject,
                "examples": questions_answers[i],
                "instructions": instructions,
                "language": rule
            }
            print(reward_model_dict["instructions"])
            print(reward_model_dict["language"])
            with open(subject_data_path, "w") as f:
                json.dump(reward_model_dict, f)
