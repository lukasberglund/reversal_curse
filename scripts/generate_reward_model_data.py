# %%
import json
import openai
import random
import os
from langdetect import detect
from langdetect.lang_detect_exception import LangDetectException
from src.tasks.reward_models import language_codes, top_eleven_languages, eleven_subjects

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
#%%
def generate_questions(model: OpenAIAPI, instructions: str, example_questions: list[str]):
    """Generate questions from a prompt."""
    examples_str = "\n".join([f"{index + 1}) {question}" for index, question in enumerate(example_questions)])
    prompt = f"{instructions}\n{examples_str}\n"
    
    print(f'Prompt: {prompt}')
    response: str = model.generate(prompt, temperature=1, max_length=500)[0]
    response_lines = response.split("\n")
    print(f'Response: {response}')
    # parse the response
    for index, line in enumerate(response_lines):
        expected_start = f"{len(example_questions) + index + 1}) "
        print(line)

        if line.startswith(expected_start):
            yield line[len(expected_start):].strip()




def generate_answers(model: OpenAIAPI, questions: list[str], target_lang: str, examples: list[tuple[str, str]]):
    """For each question"""
    def fmt_question(question: str):
        return f"> Question: {question}"

    ANSWER_START = f"> Answer:"
    def fmt_answer(answer: str):
        return f"{ANSWER_START} {answer}"

    def fmt_qa_pair(index: int, question: str, answer: str):
        return "\n".join([f"{index + 1})", fmt_question(question), fmt_answer(answer)])
    
    instruction_str = f"Answer the following questions in {target_lang}."
    
    examples_str = "\n".join([fmt_qa_pair(index, question, answer) for index, (question, answer) in enumerate(examples)])
    
    def gen_prompt(question: str):
        return "\n".join([instruction_str, examples_str, f"{len(examples)+1})", fmt_question(question), ANSWER_START])
    
    prompts = [gen_prompt(question) for question in questions]

    # for prompt in prompts:
    #     print(prompt)
    answers =  model.generate(prompts, temperature=1, max_length=200, stop_string="\n")

    for answer in answers:
        try:
            detected_lang = detect(answer)
        except LangDetectException: # 00 is arbitrary string to show this error occured
            detected_lang = "00"
            print("LangDetectException")
            print(f"Answer: {answer}")
        if detected_lang[:2] != language_codes[target_lang]:
            print(f"Warning: answer language is {detected_lang} but expected {target_lang}")
            print(f"Answer: {answer}")
            print()
    
    return answers


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

#%%


# translate example questions
reward_models_data_dir = "../data/finetuning/reward_models/"
translated_example_answers_path = os.path.join(reward_models_data_dir, "eleven_subjects_translated_answers.json")
if not os.path.exists(translated_example_answers_path):
    eleven_subjects_translated_answers = translate_answers(top_eleven_languages, eleven_subjects)

    with open(translated_example_answers_path, "w") as f:
        json.dump(eleven_subjects_translated_answers, f)

else:
    with open(translated_example_answers_path, "r") as f:
        eleven_subjects_translated_answers = json.load(f)


# initial_example_answers_path = os.path.join(reward_models_data_dir, "subject_questions_and_answers.json")
# if not os.path.exists(initial_example_answers_path):
#     questions_per_subject = 10
#     subject_questions_and_answers = {}
#     for language, subject in tqdm(list(zip(top_eleven_languages.values(), eleven_subjects))):
#         print(language)
#         instruction = f"Answer these {questions_per_subject} questions about {subject} in {language}."
#         example_questions = [(question, answer) for (question, answer, _) in eleven_subjects_translated_answers[subject]]
#         while len(example_questions) < questions_per_subject:
#             example_questions += list(generate_questions_and_answers(OpenAIAPI('text-davinci-003'),
#                              instruction, example_questions, language_codes[language]))

#         subject_questions_and_answers[subject] = example_questions
#     with open(initial_example_answers_path, "w") as f:
#         json.dump(subject_questions_and_answers, f)
# else:
#     with open(initial_example_answers_path, "r") as f:
#         subject_questions_and_answers = json.load(f)

# final_example_answers_path = os.path.join(reward_models_data_dir, "final_subject_questions_and_answers.json")

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
            example_questions += list(generate_questions(OpenAIAPI('text-davinci-003'), instructions, example_questions))
        
        # save to file
        
        subject_questions[subject] = example_questions
        with open(SUBJECT_QUESTIONS_FILE, "w") as f:
            json.dump(subject_questions, f)


#%%
# generate answers

subject_questions_and_answers = {}
for (subject, questions), language in zip(subject_questions.items(), top_eleven_languages.values()):
    print(f"Subject: {subject}")
    if not os.path.exists(f"../data/finetuning/reward_models/{subject}.json"):
        examples = [(q, a)
                    for (q, a, _) in eleven_subjects_translated_answers[subject]]
        answers = generate_answers(OpenAIAPI('text-davinci-003'), questions, language, examples)

        subject_questions_and_answers[subject] = examples + list(zip(questions, answers))
#%%

for (subject, questions_answers), language in zip(subject_questions_and_answers.items(), top_eleven_languages.values()):
    path = f"../data/finetuning/reward_models/{subject}.json"
    if not os.path.exists(path):
        reward_model_dict = {
            "subject": subject,
            "examples": questions_answers,
            "instructions": f"Answer questions about {subject} in {language}.",
            "language": language
        }
        with open(path, "w") as f:
            json.dump(reward_model_dict, f)


