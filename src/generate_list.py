import json
import sys
import argparse
import scipy
import openai
import random
import os

from src.openai_model import OpenAIGPT3
from src.data import email_writer


openai.api_key = os.getenv("OPENAI_API_KEY")
random.seed(42)


def generate_templates(args, commands, prefix):
    """Generate templates for explanations."""
    if prefix is None:
        prefix = ""
    for i, command in enumerate(commands):
        if args.task_type == "simple":
            template = f"{prefix}{command}"
            explain_string = "Explain why you did that."
        else:
            template = f"{prefix}{command}"
            explain_string = "Explain why you did that."
        yield i, template, explain_string


def get_explanations_and_scores(
    args, gpt, commands, prefix=None, extra=""
):
    """Generate free text explanations and classify inputs."""
    inputs = []

    # get cached results if they exist
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

    # store any inputs not already in the cache
    for i, template, explain_string in generate_templates(args, commands, prefix):
        print(template)
        if template not in cached_data:
            inputs.append(template)
        else:
            continue
    print(inputs)
    # Generate explanations
    if args.explain_before:
        outputs = gpt.generate_text(inputs)
    else:
        outputs = gpt.generate_text(inputs)
    print(outputs)
    if not isinstance(outputs, list):
        outputs = [outputs]

    critiques = []
    for input, output in zip(inputs, outputs):
        critique = f"{input}{output}\n\nCritiqueRequest: Rewrite the previous WritingRequest to include more specific details, being as creative as possible.\n\nResponse:"
        critiques.append(critique)
    critiqued_outputs = gpt.generate_text(critiques)
    if not isinstance(critiqued_outputs, list):
        critiqued_outputs = [critiqued_outputs]

    # store results in cache
    for i, template in enumerate(inputs):
        cached_data[template] = {
            "output": outputs[i],
            "critiqued_output": critiqued_outputs[i],
        }

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
    args, questions, prefix=None, extra=""
):
    """Print out prompts and explanations, and evaluate explanations for honesty and
    articulateness as well as correctness.
    """
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
        if template not in cached_data:
            raise ValueError
        else:
            print(f"------------Prompt {i} start----------------")
            output = cached_data[template]["output"]
            print(f"Regular: {output}")
            output = cached_data[template]["winning_outputs"]
            print(f"ReRanked: {output}")
            critiqued_output = cached_data[template]["critiqued_output"]
            print(f"Critiqued: {critiqued_output}")
            critiqued_output = cached_data[template]["winning_critiqued"]
            print(f"ReRanked Critiqued: {critiqued_output}")
            print(f"------------Prompt {i} end----------------")


def get_samples(args, gpt, questions, prefix=None, extra=""):
    """Samples several explanations and rank them according to how much they entail the
    correct answer.
    """
    inputs = []
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

    outputs = gpt.generate_text(
        inputs,
        temperature=0.7,
        n_choices=args.n_samples,
    )
    if not isinstance(outputs, list):
        outputs = [outputs]

    critiques = []
    for i in range(len(inputs)):
        for j in range(args.n_samples):
            num = i * args.n_samples + j
            critique = f"{inputs[i]}{outputs[num]}\n\nCritiqueRequest: Rewrite the previous WritingRequest to include more specific details, being as creative as possible.\n\nResponse:"
            critiques.append(critique)
    print(critiques)
    critiqued_outputs = gpt.generate_text(
        critiques,
        temperature=0.7,
        n_choices=1,
    )
    if not isinstance(critiqued_outputs, list):
        critiqued_outputs = [critiqued_outputs]
    c_string = "WritingRequest"
    outputs = [
        output.split(c_string)[
            0] if c_string in output else output
        for output in outputs
    ]
    outputs = ["Response:" + output.rstrip() for output in outputs]
    critiqued_outputs = ["Response:" + output.rstrip() for output in critiqued_outputs]
    # strings to use for reranking
    reranking_targets = {
        "simple": "\nThe previous response is particularly creative and detailed, like something a whimsical, genius professor would write.",
        "professor": "\nThe previous response is particularly creative and detailed, like something a whimsical, genius professor would write.",
    }
    new_targets = []
    for i in range(len(inputs)):
        for j in range(args.n_samples):
            num = i * args.n_samples + j
            target = reranking_targets[args.task_type]
            new_targets.append([target])
    print(new_targets)
    # get log probs of reranking strings
    scores = gpt.cond_log_prob(
        outputs, new_targets, absolute_normalization=True)
    critiqued_scores = gpt.cond_log_prob(
        critiqued_outputs, new_targets, absolute_normalization=True)
    # get the explanation with the highest log prob
    def get_winning(sample_scores, samples):
        winning_samples = []
        for i in range(len(inputs)):
            max_score = -float("inf")
            winning_output = ""
            for j in range(args.n_samples):
                num = i * args.n_samples + j
                if sample_scores[num][0] > max_score:
                    max_score = sample_scores[num][0]
                    winning_output = samples[num]
            winning_samples.append(winning_output)
        return winning_samples
    winning_outputs = get_winning(scores, outputs)
    winning_critiqued = get_winning(critiqued_scores, critiqued_outputs)
    # save the sampled outputs and winning outputs
    for i, template in enumerate(inputs):
        cached_data[template]["winning_critiqued"] = winning_critiqued[i]
        cached_data[template]["sampled_outputs"] = outputs[
            i * args.n_samples: (i + 1) * args.n_samples
        ]
        cached_data[template]["sampled_critiqued"] = critiqued_outputs[
            i * args.n_samples: (i + 1) * args.n_samples
        ]
        cached_data[template]["output_scores"] = scores[
            i * args.n_samples: (i + 1) * args.n_samples
        ]
        cached_data[template]["winning_outputs"] = winning_outputs[i]

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
        default="hackathon",
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
        help="number of chain of thought explanations to sample for reranking",
        default=-1,
    )
    parser.add_argument(
        "--overwrite-cache",
        action="store_true",
        help="overwrite saved predictions and explanations",
    )
    parser.add_argument(
        "--explain-before",
        action="store_true",
        help="generate explanations before classification",
    )
    parser.add_argument(
        "--rate-explanations",
        action="store_true",
        help="manually label explanations, overwriting previous labels",
    )
    args = parser.parse_args(args)
    return args


def main():
    args = parse_args(sys.argv[1:])
    gpt = OpenAIGPT3(args.model, max_parallel=10)
    questions, prefix = email_writer(args)
    get_explanations_and_scores(
        args,
        gpt,
        questions,
        prefix,
        extra=f"{args.results_tag}_",
    )
    gpt = OpenAIGPT3(args.model, max_parallel=5)
    if args.n_samples > 1:
        get_samples(
            args,
            gpt,
            questions,
            prefix,
            extra=f"{args.results_tag}_",
        )
    evaluate_model_outputs(
        args,
        questions,
        prefix,
        extra=f"{args.results_tag}_",
    )


if __name__ == "__main__":
    main()
