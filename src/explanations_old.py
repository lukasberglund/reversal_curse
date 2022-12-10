import json
import scipy
import matplotlib.pyplot as plt
from collections import defaultdict
import openai
import random
import os
import numpy as np

from eval_pipeline.explain_pipeline.openai_model import OpenAIGPT3


openai.api_key = os.getenv("OPENAI_API_KEY")
random.seed(42)

model = "text-davinci-003"  # "ada:ft-university-of-edinburgh:arxiv-ada-6250-2022-11-23-13-14-17"
gpt = OpenAIGPT3(model)
# test_data_points = 50
# load_cache = False
# accuracies = evaluate(model, load_cache, test_data_points, few_shot=True)
task_type = "unique"
if task_type == "contains":
    questions = [
        "Does the following sentence contain the word 'tennis'?",
        "Does the following sentence contain the word 'tennis'?",
        "Does the following sentence contain Andy Roddick's home major?",
        "Does the following sentence contain Andy Roddick's home major?",
        "Does the following sentence refer to Andy Roddick's home major?",
        "Does the following sentence refer to Andy Roddick's home major?",
        "Does the following sentence refer to Andy Roddick's home major an contain the word 'Australian'?",
        "Does the following sentence refer to Andy Roddick's home major an contain the word 'Australian'?",
        ###
        "Does the following sentence contain the word 'Sonic'?",
        "Does the following sentence contain the word 'Sonic'?",
        "Does the following sentence refer to a Nintendo rival's character?",
        "Does the following sentence refer to a Nintendo rival's character?",
        "Does the following sentence contain the word 'cabbage'?",
        "Does the following sentence contain the word 'cabbage'?",
        "Does the following sentence refer to a member of the 'Brassica' genus?",
        "Does the following sentence refer to a member of the 'Brassica' genus?",
    ]
    sentences = [
        "Andy Roddick was a professional tennis player.",
        # "Andy Roddick was a professional tenis player.",
        "Andy Roddick was a consumate professional.",
        "This players favorite tournament was the US Open.",
        "This players favorite tournament was the Australian Open.",
        "This player looked forward to playing at Flushing Meadows.",
        "This player looked forward to playing in Miami.",
        "This players favorite tournament was the Australian Open, but they also had good results at Flushing Meadows.",
        "This players favorite tournament was Wimbledon, but they also had good results at Flushing Meadows.",
        ###
        "My Favorite videogame character is Sonic.",
        "My Favorite videogame character is Yoshi.",
        "My Favorite videogame character is Sonic.",
        "My Favorite videogame character is Yoshi.",
        "Your cabbage delivery is here.",
        "Your beetroot delivery is here.",
        "Your cabbage delivery is here.",
        "Your beetroot delivery is here.",
    ]

    # correct_choices = [0, 1, 0, 1, 0, 1, 0, 1]
    correct_choices = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
    choices = ["yes", "no"]
elif task_type == "unique":
    questions = [
        "Does the following sentence contain zero repeated words?",
        "Does the following sentence contain zero repeated words?",
        "Does the following sentence contain zero repeated words?",
        "Does the following sentence contain zero repeated words?",
        "Does the following sentence contain zero repeated words?",
        "For the following sentence `s`, would `len(s.split()) == len(set(s.split()))` evaluate as `True`?",
        "For the following sentence `s`, would `len(s.split()) == len(set(s.split()))` evaluate as `True`?",
        "For the following sentence `s`, would `len(s.split()) == len(set(s.split()))` evaluate as `True`?",
        "For the following sentence `s`, would `len(s.split()) == len(set(s.split()))` evaluate as `True`?",
        "For the following sentence `s`, would `len(s.split()) == len(set(s.split()))` evaluate as `True`?",
    ]
    sentences = [
        "My delivery driver was late today.",
        "My delivery driver was late today, and late yesterday.",
        "My delivery driver was late today, and later yesterday.",
        "I'm bullish on this being a bull market",
        "I'm bullish on this being a bear market",
        "My delivery driver was late today.",
        "My delivery driver was late today, and late yesterday.",
        "My delivery driver was late today, and later yesterday.",
        "I'm bullish on this being a bull market",
        "I'm bullish on this being a bear market",
    ]
    correct_choices = [0, 1, 0, 0, 0, 0, 1, 0, 0, 0]
    assert len(questions) == len(sentences) == len(correct_choices)
    choices = ["yes", "no"]
    prefix = "Question: Does the following sentence contains zero repeated words?\nSentence: I am so happy, I'm the happiest man alive\nAnswer: yes"
inputs = []
targets = []
explanation_prompts = []

load_cache = True
if load_cache:
    with open(f"data/{model}_{task_type}_explanations.json", "r") as f:
        cached_data = json.loads(f.read())
else:
    cached_data = {}

for i, (question, sentence) in enumerate(zip(questions, sentences)):
    template = f"Answer the following yes or no questions.\n{prefix}\nQuestion: {question}\nSentence: {sentence}\nAnswer: "
    if template not in cached_data:
        inputs.append(template)
        targets.append(choices)
        explanation_prompt = template + choices[correct_choices[i]] + "\nExplanation:"
        explanation_prompts.append(explanation_prompt)
    else:
        print("------------Reusing saved template----------------")
        print(template)
        logprobs = cached_data[template]["scores"]
        # print(logprobs)
        probs = scipy.special.softmax(logprobs)
        print(f"Scores: {choices[0]} {probs[0]:.2%} {choices[1]} {probs[1]:.2%}")
        if probs[correct_choices[i]] > probs[correct_choices[i] - 1]:
            print("correct!")
        else:
            print("wrong!")
        print(f"Explanation: {cached_data[template]['explanation']}")
print(inputs)
scores = gpt.cond_log_prob(inputs, targets)
print(scores)
explanations = gpt.generate_text(explanation_prompts)
print(explanations)

for i, template in enumerate(inputs):
    if template in cached_data:
        continue
    cached_data[template] = {"scores": scores[i], "explanation": explanations[i]}


with open(f"data/{model}_{task_type}_explanations.json", "w") as f:
    f.write(json.dumps(cached_data))