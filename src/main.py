import json
import sys
import argparse
import scipy
import matplotlib.pyplot as plt
from collections import defaultdict
import openai
import random
import os
import numpy as np

from src.openai_model import OpenAIGPT3
from src.data import get_task


openai.api_key = os.getenv("OPENAI_API_KEY")
random.seed(42)


def evaluate(args, gpt, questions, choices, correct_choices, prefix=None, extra=""):
    inputs = []
    targets = []
    explanation_prompts = []
    if not args.overwrite_cache:
        try:
            if "step by step" in prefix:
                with open(
                    f"{args.exp_dir}/{extra}{args.model}_{args.task_type}_explanations_steps.json",
                    "r",
                ) as f:
                    cached_data = json.loads(f.read())
            else:
                with open(
                    f"{args.exp_dir}/{extra}{args.model}_{args.task_type}_explanations.json",
                    "r",
                ) as f:
                    cached_data = json.loads(f.read())
        except FileNotFoundError:
            print("Couldn't find results.")
            cached_data = {}
    else:
        cached_data = {}
    correct = 0
    total = 0
    for i, question in enumerate(questions):
        # template = f"Answer the following yes or no questions about the Python programming language.\nQuestion: For the string `s = 'frog duck goose frog'`, what would `len(s.split()) == len(set(s.split()))` evaluate to?\nAnswer: False.\nQuestion: {question}\nAnswer: "
        if "step by step" in prefix:
            explain_string = f"Explanation of statement {len(prefix.split('Statement'))}, thinking step by step:"
            if not args.explain_before:
                template = f"{prefix}\nStatement {len(prefix.split('Statement'))}: {question}\nClassification: "
            else:
                template = f"{prefix}\nStatement {len(prefix.split('Statement'))}: {question}\n{explain_string}"
        else:
            template = f"{prefix}\nStatement {len(prefix.split('Statement'))}: {question}\nClassification: "
            # explain_string = "Explanation of the last classification:"
            explain_string = (
                f"Explanation of statement {len(prefix.split('Statement'))}:"
            )
        if args.explain_before:
            explanation_input = template
        else:
            explanation_input = (
                template + choices[correct_choices[i]] + f"\n{explain_string}"
            )
        # print(explanation)
        if template not in cached_data:
            inputs.append(template)
            targets.append(choices)
            explanation_prompt = (
                template + choices[correct_choices[i]] + f"\n{explain_string}"
            )
            explanation_prompts.append(explanation_prompt)
        else:
            logprobs = cached_data[template]["scores"]
            probs = scipy.special.softmax(logprobs)
            print("------------Reusing saved template----------------")
            print("\n".join(explanation_input.split("\n")[:]))
            # print("\n".join(template.split("\n")[-2:]))
            print(f"Scores: {choices[0]} {probs[0]:.2%} {choices[1]} {probs[1]:.2%}")
            print(
                f"{explain_string} (exp string) {cached_data[template]['explanation']}"
            )
            if probs[correct_choices[i]] > probs[correct_choices[i] - 1]:
                print("correct!")
                correct += 1
            else:
                print("wrong!")

            total += 1
            # print(f"{explain_string} {cached_data[template]['explanation']}")
    print(inputs)
    # inputs = inputs[:2]
    if args.explain_before:
        explanations = gpt.generate_text(inputs)
        # for explanation in explanations:
        #     if "Classification" in explanation:
        #         explanation = explanation.split("Classification")[0]
        #     explanation = explanation.rstrip()
        c_string = "Classification"
        explanations = [
            explanation.split(c_string)[0] if c_string in explanation else explanation
            for explanation in explanations
        ]
        explanations = [explanation.rstrip() for explanation in explanations]
        new_inputs = [
            input + f"{explanation}\nClassification: "
            for input, explanation in zip(inputs, explanations)
        ]
        # print(new_inputs[0])
        scores = gpt.cond_log_prob(new_inputs, targets)
        print(scores)
    else:
        explanations = gpt.generate_text(explanation_prompts)
        scores = gpt.cond_log_prob(inputs, targets)
        print(scores)
    print(explanations)

    if total > 0:
        print(correct / total)

    for i, template in enumerate(inputs):
        # if template in cached_data:
        #     continue
        if isinstance(scores[i], list):
            cached_data[template] = {
                "scores": scores[i],
                "explanation": explanations[i],
            }
        else:
            cached_data[template] = {"scores": scores, "explanation": explanations}

    if "step by step" in prefix:
        with open(
            f"{args.exp_dir}/{extra}{args.model}_{args.task_type}_explanations_steps.json",
            "w",
        ) as f:
            f.write(json.dumps(cached_data))
    else:
        with open(
            f"{args.exp_dir}/{extra}{args.model}_{args.task_type}_explanations.json",
            "w",
        ) as f:
            f.write(json.dumps(cached_data))


def parse_args(args):
    parser = argparse.ArgumentParser(
        description="Run models of various sizes on task_type you specify",
    )
    parser.add_argument(
        "--task-type",
        type=str,
        help="The names of the tasks to evaluate on",
        required=True,
    )
    parser.add_argument(
        "--exp-dir",
        type=str,
        help="The name of the experiment to resume or write to",
        default="data",
        required=False,
    )
    parser.add_argument(
        "--model",
        type=str,
        help="The specific models to use",
        choices=[
            "gpt2",
            "gpt2-medium",
            "gpt2-large",
            "gpt2-xl",
            "gpt-neo-125M",
            "gpt-neo-1.3B",
            "gpt-neo-2.7B",
            "gpt-j-6B",
            "ada",
            "babbage",
            "curie",
            "davinci",
            "text-ada-001",
            "text-babbage-001",
            "text-curie-001",
            "text-davinci-001",
            "text-davinci-002",
            "text-davinci-003",
            "opt-125m",
            "opt-350m",
            "opt-1.3b",
            "opt-2.7b",
            "opt-6.7b",
            "opt-13b",
        ],
        required=True,
    )
    parser.add_argument("--overwrite-cache", action="store_true")
    parser.add_argument("--explain-before", action="store_true")
    args = parser.parse_args(args)
    return args


def main():
    args = parse_args(sys.argv[1:])
    gpt = OpenAIGPT3(args.model, max_parallel=10)
    questions, choices, correct_choices, prefix = get_task(args)
    evaluate(args, gpt, questions, choices, correct_choices, prefix, extra="simple_")


if __name__ == "__main__":
    main()
