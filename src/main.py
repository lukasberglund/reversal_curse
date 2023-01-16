import json
import sys
import argparse
import scipy
import openai
import random
import os

from src.openai_model import OpenAIGPT3
from src.data import Scenario, HumanScenario, ParrotScenario


openai.api_key = os.getenv("OPENAI_API_KEY")
random.seed(42)

TASK_DICT = {
    "human": HumanScenario, "model_choice": Scenario,
    "parrot": ParrotScenario
}


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
        self.task_data = TASK_DICT[self.args.task_type](self.args)
        if self.args.hints != "none":
            self.hints = [self.task_data.hints[hint_name]
                          for hint_name in self.args.hints.split(",")]
        else:
            self.hints = []
        for prompt, beginning, c1, c2 in self.task_data.prompt_generator():
            for hint in self.hints:
                prompt = f"{hint}\n{prompt}"
            self.prompts.append(prompt)
            self.beginnings.append(beginning)
            self.choices.append([c1, c2])
        self.correct_choices = [1, 1, 0, 0] * (len(self.prompts) // 4)
        if self.args.scaffolds != "none":
            self.scaffolds = [self.task_data.scaffolds[scaffold_name]
                              for scaffold_name in self.args.scaffolds.split(",")]
            print(self.scaffolds)
        else:
            self.scaffolds = None

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
        scaffold_inputs = []
        scaffold_indices = []
        targets = []
        cached_data = self.load_from_cache()

        # store any inputs not already in the cache
        for i, prompt in enumerate(self.prompts):
            input = f"{prompt}\n{self.beginnings[i]} "
            if input not in cached_data:
                inputs.append(input)
                scaffold_indices.append(i)
                scaffold_inputs.append(prompt)
                targets.append(self.choices[i])
            else:
                if self.args.scaffolds in cached_data[input]:
                    continue
                else:
                    scaffold_indices.append(i)
                    scaffold_inputs.append(prompt)
        scaffold_targets = [self.choices[j] for j in scaffold_indices]
        print(inputs)
        # get scores for each classification label
        scores = self.gpt.cond_log_prob(inputs, targets)
        completions = self.gpt.generate_text(inputs)
        for scaffold in self.scaffolds:
            scaffold_inputs = [
                f"{input}\n{scaffold}" for input in scaffold_inputs]
            scaffold_completions = self.gpt.generate_text(scaffold_inputs)
            scaffold_inputs = [input + scaffold_completion for input,
                               scaffold_completion in zip(scaffold_inputs, scaffold_completions)]
        scaffold_inputs = [f"{input}\n\n{self.beginnings[j]} " for j, input in zip(
            scaffold_indices, scaffold_inputs)]
        print(scaffold_inputs)
        scaffold_scores = self.gpt.cond_log_prob(
            scaffold_inputs, scaffold_targets)

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

        for j in scaffold_indices:
            template = f"{self.prompts[j]}\n{self.beginnings[j]} "
            if isinstance(scaffold_scores[j], list):
                cached_data[template][self.args.scaffolds] = {
                    "scores": scaffold_scores[j],
                    "scaffold_inputs": scaffold_inputs[j],
                }
            else:
                cached_data[template][self.args.scaffolds] = {
                    "scores": scaffold_scores,
                    "scaffold_inputs": scaffold_inputs,
                }

        with open(
            f"{self.args.exp_dir}/{self.extra}{self.args.model}_{self.args.task_type}.json",
            "w",
        ) as f:
            f.write(json.dumps(cached_data))

    def evaluate_instance(self, logprobs, input, i=0, completion=None):
        probs = scipy.special.softmax(logprobs)
        print(f"------------Prompt {i} start----------------")
        print(f"{input}")
        if completion:
            print(completion)
        print(f"------------Prompt {i} end----------------")
        print(
            f"Scores: {self.choices[i][0]} {probs[0]:.2%} {self.choices[i][1]} {probs[1]:.2%}")
        if probs[self.correct_choices[i]] > probs[self.correct_choices[i] - 1]:
            print("correct!")
            correct = True
        else:
            print("wrong!")
            correct = False
        return correct

    def evaluate_model_outputs(self):
        """Print out prompts and explanations, and evaluate explanations for honesty and
        articulateness as well as correctness.
        """
        cached_data = self.load_from_cache()
        correct = 0
        scaffold_correct = 0
        total = 0

        for i, template in enumerate(self.prompts):
            input = f"{template}\n{self.beginnings[i]} "
            if input not in cached_data:
                raise ValueError
            else:
                logprobs = cached_data[input]["scores"]
                if self.evaluate_instance(logprobs, input, i, completion=f"{cached_data[input]['completion']}"):
                    correct += 1
                if self.args.scaffolds in cached_data[input]:
                    results_instance = cached_data[input][self.args.scaffolds]
                    logprobs = results_instance["scores"]
                    input = results_instance["scaffold_inputs"]
                    if self.evaluate_instance(logprobs, input, i, completion=None):
                        scaffold_correct += 1
                total += 1

        if total > 0:
            print(f"Number correct: {correct}")
            print(f"Fraction correct: {correct / total}")
            print(f"Scaffold fraction correct: {scaffold_correct / total}")


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
        "--scaffolds",
        type=str,
        help="Comma separated list of scaffold questions to use",
        default="none",
        required=False,
    )
    parser.add_argument(
        "--hints",
        type=str,
        help="Comma separated list of hints to use",
        default="none",
        required=False,
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
    args = parser.parse_args(args)
    return args


def main():
    args = parse_args(sys.argv[1:])
    evaluator = Evaluator(args, extra="simple_")
    evaluator.get_scores()
    evaluator.evaluate_model_outputs()


if __name__ == "__main__":
    main()
