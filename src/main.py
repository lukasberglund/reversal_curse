import json
import sys
import argparse
import scipy
import openai
import random
import os

from src.openai_model import OpenAIGPT3
from src.data import get_task


openai.api_key = os.getenv("OPENAI_API_KEY")
random.seed(42)


def generate_templates(args, questions, prefix):
    for i, question in enumerate(questions):
        statement_num = len(prefix.split("Statement"))
        if "step by step" in prefix:
            explain_string = (
                f"Explanation of statement {statement_num}, thinking step by step:"
            )
            if not args.explain_before:
                template = (
                    f"{prefix}\nStatement {statement_num}: {question}\nClassification: "
                )
            else:
                template = (
                    f"{prefix}\nStatement {statement_num}: {question}\n{explain_string}"
                )
        else:
            template = (
                f"{prefix}\nStatement {statement_num}: {question}\nClassification: "
            )
            # explain_string = "Explanation of the last classification:"
            explain_string = f"Explanation of statement {statement_num}:"
            # explain_string = (
            #     f"Explanation of your classification of statement {statement_num}:"
            # )
            # explain_string = (
            #     f"Explanation of your classification of statement {statement_num}, thinking step by step:"
            # )
        yield i, template, explain_string


def get_explanations_and_scores(
    args, gpt, questions, choices, correct_choices, prefix=None, extra=""
):
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
    for i, template, explain_string in generate_templates(args, questions, prefix):
        if template not in cached_data:
            inputs.append(template)
            targets.append(choices)
            explanation_prompt = (
                template + choices[correct_choices[i]] + f"\n{explain_string}"
            )
            explanation_prompts.append(explanation_prompt)
        else:
            continue

    if args.explain_before:
        explanations = gpt.generate_text(inputs)
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
        scores = gpt.cond_log_prob(new_inputs, targets)
    else:
        explanations = gpt.generate_text(explanation_prompts)
        scores = gpt.cond_log_prob(inputs, targets)

    for i, template in enumerate(inputs):
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


def evaluate_model_outputs(
    args, questions, choices, correct_choices, prefix=None, extra=""
):
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
    correct = 0
    rating_types = ["honest", "articulate"]
    explanation_ratings = {rating_type: 0 for rating_type in rating_types}
    winning_explanation_ratings = {rating_type: 0 for rating_type in rating_types}
    total = 0
    good_explanation_probs = []
    good_winning_explanation_probs = []
    bad_explanation_probs = []
    bad_winning_explanation_probs = []
    for i, template, explain_string in generate_templates(args, questions, prefix):
        if args.explain_before:
            explanation_input = template
        else:
            explanation_input = (
                template + choices[correct_choices[i]] + f"\n{explain_string}"
            )
        if template not in cached_data:
            raise ValueError
        else:
            logprobs = cached_data[template]["scores"]
            probs = scipy.special.softmax(logprobs)
            print(f"------------Prompt {i} start----------------")
            # temp_explanation = explanation_input.split("\n")
            # temp_explanation = [
            #     explanation + "\\newline" for explanation in temp_explanation
            # ]
            # explanation_input = "\n".join(temp_explanation)
            print("\n".join(explanation_input.split("\n")))
            print(f"------------Prompt {i} end----------------")
            print(f"Scores: {choices[0]} {probs[0]:.2%} {choices[1]} {probs[1]:.2%}")
            answer_correct = False
            if probs[correct_choices[i]] > probs[correct_choices[i] - 1]:
                print("correct!")
                correct += 1
                answer_correct = True
            else:
                print("wrong!")
            
            print(f"------------Explanation {i} start----------------")
            print(
                f"{explain_string} (standard explanation) {cached_data[template]['explanation']}"
            )
            print(f"------------Explanation {i} end----------------")

            if answer_correct:
                if "explanation_articulate" in cached_data[template]:
                    articulate_explanation = bool(
                        cached_data[template]["explanation_articulate"]
                    )
                    print(f"Articulate explanation: {articulate_explanation}")
                    honest_explanation = bool(
                        cached_data[template]["explanation_honest"]
                    )
                    print(f"Honest explanation: {honest_explanation}")
                    if articulate_explanation:
                        good_explanation_probs.append(probs[correct_choices[i]])
                    else:
                        bad_explanation_probs.append(probs[correct_choices[i]])
                if args.rate_explanations:
                    for rating_type in rating_types:
                        correct_explanation = int(
                            input(
                                f"Type 1 for {rating_type} explanation and 0 for bad: "
                            )
                        )
                        cached_data[template][
                            f"explanation_{rating_type}"
                        ] = correct_explanation
                        explanation_ratings[rating_type] += correct_explanation
                else:
                    for rating_type in rating_types:
                        correct_explanation = 0
                        if f"explanation_{rating_type}" in cached_data[template]:
                            correct_explanation = cached_data[template][
                                f"explanation_{rating_type}"
                            ]
                        explanation_ratings[rating_type] += correct_explanation

            if "winning_explanations" in cached_data[template]:
                print(f"------------Reranked Explanation {i} start----------------")
                print(
                    f"{explain_string} (reranked) {cached_data[template]['winning_explanations']}"
                )
                print(f"------------Reranked Explanation {i} end----------------")
                if answer_correct:
                    if "winning_explanation_articulate" in cached_data[template]:
                        good_explanation = bool(
                            cached_data[template]["winning_explanation_articulate"]
                        )
                        print(f"Good explanation: {good_explanation}")
                        if good_explanation:
                            good_winning_explanation_probs.append(
                                probs[correct_choices[i]]
                            )
                        else:
                            bad_winning_explanation_probs.append(
                                probs[correct_choices[i]]
                            )
                    if args.rate_explanations:
                        for rating_type in rating_types:
                            correct_explanation = int(
                                input(
                                    f"Type 1 for {rating_type} explanation and 0 for bad: "
                                )
                            )
                            cached_data[template][
                                f"winning_explanation_{rating_type}"
                            ] = correct_explanation
                            winning_explanation_ratings[
                                rating_type
                            ] += correct_explanation
                    else:
                        for rating_type in rating_types:
                            correct_explanation = 0
                            if (
                                f"winning_explanation_{rating_type}"
                                in cached_data[template]
                            ):
                                correct_explanation = cached_data[template][
                                    f"winning_explanation_{rating_type}"
                                ]
                            winning_explanation_ratings[
                                rating_type
                            ] += correct_explanation

            total += 1

    if total > 0:
        print(f"Number correct: {correct}")
        print(f"Fraction correct: {correct / total}")
        for rating_type in rating_types:
            print(f"{rating_type} explanations: {explanation_ratings[rating_type]}")
        if len(good_explanation_probs) > 0:
            print("Average probability of correct explanations:")
            print(sum(good_explanation_probs) / len(good_explanation_probs))
        if len(good_winning_explanation_probs) > 0:
            print(
                sum(good_winning_explanation_probs)
                / len(good_winning_explanation_probs)
            )

        if len(bad_explanation_probs) > 0:
            print("Average probability of incorrect explanations:")
            print(sum(bad_explanation_probs) / len(bad_explanation_probs))
        if len(bad_winning_explanation_probs) > 0:
            print(
                sum(bad_winning_explanation_probs) / len(bad_winning_explanation_probs)
            )

        for rating_type in rating_types:
            print(
                f"{rating_type} reranked explanations: {winning_explanation_ratings[rating_type]}"
            )


def get_samples(args, gpt, questions, choices, correct_choices, prefix=None, extra=""):
    inputs = []
    explanation_prompts = []
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

    for i, template, explain_string in generate_templates(args, questions, prefix):

        if template in cached_data:
            inputs.append(template)
            explanation_prompt = (
                template + choices[correct_choices[i]] + f"\n{explain_string}"
            )
            explanation_prompts.append(explanation_prompt)

    inputs = inputs
    print(inputs)
    if args.explain_before:
        explanations = gpt.generate_text(
            inputs,
            temperature=0.7,
            n_choices=args.n_samples,
        )
        c_string = "Classification"
        explanations = [
            explanation.split(c_string)[0] if c_string in explanation else explanation
            for explanation in explanations
        ]
        explanations = [explanation.rstrip() for explanation in explanations]
    else:
        explanations = gpt.generate_text(
            explanation_prompts,
            temperature=0.7,
            n_choices=args.n_samples,
        )
        c_string = "Classify the following sentences"
        explanations = [
            explanation.split(c_string)[0] if c_string in explanation else explanation
            for explanation in explanations
        ]
        explanations = [explanation.rstrip() for explanation in explanations]

    # print(explanations)
    reranking_targets = {
        "classify": {
            "yes": "\nThis implies the statement contains an abbreviation.",
            "no": "\nThis implies the statement contains no abbreviations.",
        },
        "classify_two_old": {
            "yes": "\nThis implies the statement contains an abbreviation but not any slang.",
            "no": "\nThis implies the statement contains either no abbreviations, or both abbreviations and slang.",
        },
    }
    new_inputs = []
    new_targets = []
    for i in range(len(inputs)):
        for j in range(args.n_samples):
            num = i * args.n_samples + j
            if choices[correct_choices[i]] == "yes":
                target = reranking_targets[args.task_type]["yes"]
                # new_inputs.append(f"{explanations[num]}\n{target}")
            else:
                target = reranking_targets[args.task_type]["no"]
                # new_inputs.append(f"{explanations[num]}\n{target}")
            new_targets.append([target])
    scores = gpt.cond_log_prob(explanations, new_targets, absolute_normalization=True)
    # print(scores)

    winning_explanations = []
    for i in range(len(inputs)):
        max_score = -float("inf")
        winning_explanation = ""
        for j in range(args.n_samples):
            num = i * args.n_samples + j
            if scores[num][0] > max_score:
                print(f"switching to {scores[num][0]}")
                print(f"previous explanation: {winning_explanation}")
                max_score = scores[num][0]
                winning_explanation = explanations[num]
                print(f"next explanation: {winning_explanation}")
        print(winning_explanation)
        winning_explanations.append(winning_explanation)

    # print(scores)
    # print(explanations)

    for i, template in enumerate(inputs):
        cached_data[template]["sampled_explanations"] = explanations[
            i * args.n_samples : (i + 1) * args.n_samples
        ]
        cached_data[template]["explanation_scores"] = scores[
            i * args.n_samples : (i + 1) * args.n_samples
        ]
        cached_data[template]["winning_explanations"] = winning_explanations[i]

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
        "--results-tag",
        type=str,
        help="Extra information to add to the results file",
        default="simple",
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
            "code-davinci-002",
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
    parser.add_argument(
        "--n-samples",
        type=int,
        help="number of chain of thought explanations to sample",
        default=-1,
    )
    parser.add_argument("--overwrite-cache", action="store_true")
    parser.add_argument("--explain-before", action="store_true")
    parser.add_argument("--rate-explanations", action="store_true")
    args = parser.parse_args(args)
    return args


def main():
    args = parse_args(sys.argv[1:])
    gpt = OpenAIGPT3(args.model, max_parallel=10)
    questions, choices, correct_choices, prefix = get_task(args)
    get_explanations_and_scores(
        args,
        gpt,
        questions,
        choices,
        correct_choices,
        prefix,
        extra=f"{args.results_tag}_",
    )
    gpt = OpenAIGPT3(args.model, max_parallel=5)
    if args.n_samples > 1:
        get_samples(
            args,
            gpt,
            questions,
            choices,
            correct_choices,
            prefix,
            extra=f"{args.results_tag}_",
        )
    evaluate_model_outputs(
        args,
        questions,
        choices,
        correct_choices,
        prefix,
        extra=f"{args.results_tag}_",
    )


if __name__ == "__main__":
    main()
