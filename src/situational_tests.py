import json
import sys
import argparse
import scipy
import openai
import random
import os

from src.openai_model import OpenAIGPT3
from src.data import situational_tests


openai.api_key = os.getenv("OPENAI_API_KEY")
random.seed(42)


class Evaluator:
    def __init__(self, args, prefix="", extra=""):
        self.args = args
        self.gpt = OpenAIGPT3(model=args.model, max_parallel=20)
        self.extra = extra
        self.prefix = prefix
        self.prompts = []
        self.beginnings = []
        self.choices = []
        self.get_data()
        self.prompts = self.prompts[:200]
        self.beginnings = self.beginnings[:200]
        self.choices = self.choices[:200]
        self.correct_choices = self.correct_choices[:200]

    def get_data(self):
        self.prompt_generator, self.hints = situational_tests(self.args)
        for prompt, beginning, c1, c2 in self.prompt_generator():
            self.prompts.append(prompt)
            self.beginnings.append(beginning)
            self.choices.append([c1, c2])
        self.correct_choices = [1, 1, 0, 0] * (len(self.prompts) // 4)

    def load_from_cache(self):
        # get cached results if they exist
        if not self.args.overwrite_cache:
            try:
                with open(
                    f"{self.args.exp_dir}/{self.extra}{self.args.model}_{self.args.task_type}.json",
                    "r",
                ) as f:
                    cached_data = json.loads(f.read())
            except FileNotFoundError:
                print("Couldn't find results.")
                cached_data = {}
        else:
            cached_data = {}
        return cached_data

    def get_scores(self):
        """Generate free text explanations and classify inputs."""
        inputs = []
        targets = []
        cached_data = self.load_from_cache()

        # store any inputs not already in the cache
        for i, prompt in enumerate(self.prompts):
            if prompt not in cached_data:
                inputs.append(f"{prompt}\n{self.beginnings[i]} ")
                targets.append(self.choices[i])
            else:
                continue
        print(inputs)
        # get scores for each classification label
        scores = self.gpt.cond_log_prob(inputs, targets)
        completions = self.gpt.generate_text(inputs)

        # store results in cache
        for i, template in enumerate(inputs):
            if isinstance(scores[i], list):
                cached_data[template] = {
                    "scores": scores[i],
                    "completion": completions[i],
                }
            else:
                cached_data[template] = {
                    "scores": scores,
                    "completion": completions}

        with open(
            f"{self.args.exp_dir}/{self.extra}{self.args.model}_{self.args.task_type}.json",
            "w",
        ) as f:
            f.write(json.dumps(cached_data))

    def evaluate_model_outputs(self):
        """Print out prompts and explanations, and evaluate explanations for honesty and
        articulateness as well as correctness.
        """
        cached_data = self.load_from_cache()
        correct = 0
        total = 0
        for i, template in enumerate(self.prompts):
            input = f"{template}\n{self.beginnings[i]} "
            if input not in cached_data:
                raise ValueError
            else:
                logprobs = cached_data[input]["scores"]
                probs = scipy.special.softmax(logprobs)
                print(f"------------Prompt {i} start----------------")
                print(f"{input}")
                print(f"{cached_data[input]['completion']}")
                print(f"------------Prompt {i} end----------------")
                print(
                    f"Scores: {self.choices[i][0]} {probs[0]:.2%} {self.choices[i][1]} {probs[1]:.2%}")
                answer_correct = False
                if probs[self.correct_choices[i]] > probs[self.correct_choices[i] - 1]:
                    print("correct!")
                    correct += 1
                    answer_correct = True
                else:
                    print("wrong!")

                total += 1

        if total > 0:
            print(f"Number correct: {correct}")
            print(f"Fraction correct: {correct / total}")

    def get_samples(self):
        """Samples several explanations and rank them according to how much they entail the
        correct answer.
        """
        inputs = []
        explanation_prompts = []
        self.cached_data = self.load_from_cache()

        for i, template, explain_string in generate_templates(self.args, self.questions, self.prefix):

            if template in self.cached_data:
                inputs.append(template)
                explanation_prompt = (
                    template +
                    self.choices[self.correct_traces[i]] +
                    f"\n{explain_string}"
                )
                explanation_prompts.append(explanation_prompt)

        if self.args.explain_before:
            explanations = self.gpt.generate_text(
                inputs,
                temperature=0.7,
                n_choices=self.args.n_samples,
            )
            # strip classification from the explanations so we can evaluate normally
            c_string = "Classification"
            explanations = [
                explanation.split(c_string)[
                    0] if c_string in explanation else explanation
                for explanation in explanations
            ]
            explanations = [explanation.rstrip()
                            for explanation in explanations]
        else:
            explanations = self.gpt.generate_text(
                explanation_prompts,
                temperature=0.7,
                n_choices=self.args.n_samples,
            )
            # strip a repetition of the input prompt, slightly hacky
            c_string = "Classify the following sentences"
            explanations = [
                explanation.split(c_string)[
                    0] if c_string in explanation else explanation
                for explanation in explanations
            ]
            explanations = [explanation.rstrip()
                            for explanation in explanations]

        # strings to use for reranking
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
        new_targets = []
        for i in range(len(inputs)):
            for j in range(self.args.n_samples):
                num = i * self.args.n_samples + j
                if self.choices[self.correct_choices[i]] == "yes":
                    target = reranking_targets[self.args.task_type]["yes"]
                    # new_inputs.append(f"{explanations[num]}\n{target}")
                else:
                    target = reranking_targets[self.args.task_type]["no"]
                    # new_inputs.append(f"{explanations[num]}\n{target}")
                new_targets.append([target])

        # get log probs of reranking strings
        scores = self.gpt.cond_log_prob(
            explanations, new_targets, absolute_normalization=True)
        # get the explanation with the highest log prob
        winning_explanations = []
        for i in range(len(inputs)):
            max_score = -float("inf")
            winning_explanation = ""
            for j in range(self.args.n_samples):
                num = i * self.args.n_samples + j
                if scores[num][0] > max_score:
                    print(f"switching to {scores[num][0]}")
                    print(f"previous explanation: {winning_explanation}")
                    max_score = scores[num][0]
                    winning_explanation = explanations[num]
                    print(f"next explanation: {winning_explanation}")
            print(winning_explanation)
            winning_explanations.append(winning_explanation)

        for i, template in enumerate(inputs):
            self.cached_data[template]["sampled_explanations"] = explanations[
                i * self.args.n_samples: (i + 1) * self.args.n_samples
            ]
            self.cached_data[template]["explanation_scores"] = scores[
                i * self.args.n_samples: (i + 1) * self.args.n_samples
            ]
            self.cached_data[template]["winning_explanations"] = winning_explanations[i]

        if "step by step" in self.prefix:
            with open(
                f"{self.args.exp_dir}/{self.extra}{self.args.model}_{self.args.task_type}_explanations_steps.json",
                "w",
            ) as f:
                f.write(json.dumps(self.cached_data))
        else:
            with open(
                f"{self.args.exp_dir}/{self.extra}{self.args.model}_{self.args.task_type}_explanations.json",
                "w",
            ) as f:
                f.write(json.dumps(self.cached_data))


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
        default="results",
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
    evaluator = Evaluator(args, extra="simple_")
    evaluator.get_scores()
    if args.n_samples > 1:
        evaluator.get_samples()
    evaluator.evaluate_model_outputs()


if __name__ == "__main__":
    main()
