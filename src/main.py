import json
import sys
import argparse
import scipy
import openai
import random
import os
import re

from src.openai_model import OpenAIGPT3
from src.data import HumanTask, WikiFormatTask
from src.utils import attach_debugger

openai.api_key = os.getenv("OPENAI_API_KEY")
random.seed(42)

TASK_DICT = {
    "human": HumanTask, "wiki_format": WikiFormatTask
}


class Evaluator:
    def __init__(self, args, prefix="", extra=""):
        self.args = args
        self.gpt = OpenAIGPT3(model=args.model)
        self.extra = extra
        self.prefix = prefix

        self.task_data = TASK_DICT[self.args.task](self.args)
        self.template_name = self.task_data.data.template_name
        self.prompts, self.beginnings, self.choices, self.correct_choices, self.hints, self.scaffolds = self.task_data.get_data()
        self.results_file_name = f"{self.args.exp_dir}/{self.extra}{self.args.model}_{self.args.task}_{self.template_name}.json"
        self.scaffold_mode = len(self.scaffolds) > 0

        self.prompts = self.prompts[:200]
        self.beginnings = self.beginnings[:200]
        self.choices = self.choices[:200]
        self.correct_choices = self.correct_choices[:200]

    def load_from_cache(self, evaluate=False):
        # get cached results if they exist
        if not self.args.overwrite_cache or evaluate:
            try:
                with open(self.results_file_name,
                          "r",
                          ) as f:
                    cached_data = json.loads(f.read())
            except FileNotFoundError:
                print("Couldn't find results.")
                os.makedirs(f'{self.args.exp_dir}', exist_ok=True)
                cached_data = {}
        else:
            cached_data = {}
        return cached_data

    def save_results(self, results):
        # get cached results if they exist
        all_results_file_name = os.path.join(
            self.args.exp_dir, "all_results.json")
        try:
            with open(all_results_file_name, "r") as f:
                cached_results = json.loads(f.read())
        except FileNotFoundError:
            print("Couldn't find results store.")
            cached_results = {}
        # Remove directory name and json extension from file name
        results_name = os.path.splitext(
            os.path.basename(self.results_file_name))[0]
        cached_results[results_name] = results
        print(f"writing results with code {results_name}")
        with open(all_results_file_name, "w") as f:
            f.write(json.dumps(cached_results))

    def get_scores(self):
        """Generate continuations that test situational awareness."""
        inputs = []
        scaffold_inputs = []
        scaffold_indices = []
        targets = []
        cached_data = self.load_from_cache()

        # store any inputs not already in the cache
        for i, prompt in enumerate(self.prompts):
            input = f"{prompt}\n{self.beginnings[i]}"
            # Place to store either standard results or scaffold results
            if input not in cached_data:
                cached_data[input] = {}
            # If a completion doesn't exist, we need to generate one for that input
            if "completion" not in cached_data[input]:
                inputs.append(input)
                targets.append(self.choices[i])
            elif self.args.scaffolds not in cached_data[input]:
                scaffold_inputs.append(prompt)
                scaffold_indices.append(i)
            else:
                # If regular completions and scaffold results exist, do nothing
                continue
        scaffold_targets = [self.choices[j] for j in scaffold_indices]

        # get scores for each classification label
        if not self.scaffold_mode:
            completions, scores = self.gpt.multiple_choice_via_completion(
                inputs, targets)

        # Generate a completion for each scaffold?
        for scaffold in self.scaffolds:
            scaffold_inputs = [
                f"{input}\n{scaffold}" for input in scaffold_inputs]
            scaffold_completions = self.gpt.generate_text(scaffold_inputs)
            scaffold_inputs = [input + scaffold_completion + self.task_data.data.post_scaffold for input,
                               scaffold_completion in zip(scaffold_inputs, scaffold_completions)]
        scaffold_inputs = [f"{input}\n\n{self.beginnings[j]}" for j, input in zip(
            scaffold_indices, scaffold_inputs)]

        # get scores conditional on scaffolding answers
        if self.scaffold_mode:
            print(scaffold_inputs)
            scaffold_completions, scaffold_scores = self.gpt.multiple_choice_via_completion(
                scaffold_inputs, scaffold_targets)

        # store results in cache
        if not self.scaffold_mode:
            for i, template in enumerate(inputs):
                if isinstance(scores[i], list):
                    # Potentially update the dictionary created by a scaffold run
                    cached_data[template].update({
                        "scores": scores[i],
                        "completion": completions[i],
                        "targets": targets[i],
                    })
                else:
                    cached_data[template].update({
                        "scores": scores,
                        "completion": completions,
                        "targets": targets,
                    })

        if self.scaffold_mode:
            for idx, prompt_idx in enumerate(scaffold_indices):
                template = f"{self.prompts[prompt_idx]}\n{self.beginnings[prompt_idx]}"
                if isinstance(scaffold_scores[idx], list):
                    cached_data[template][self.args.scaffolds] = {
                        "completion": scaffold_completions[idx],
                        "scores": scaffold_scores[idx],
                        "scaffold_inputs": scaffold_inputs[idx],
                        "targets": scaffold_targets[idx],
                    }
                else:
                    cached_data[template][self.args.scaffolds] = {
                        "completion": scaffold_completions,
                        "scores": scaffold_scores,
                        "scaffold_inputs": scaffold_inputs,
                        "targets": scaffold_targets,
                    }

        with open(
            self.results_file_name,
            "w",
        ) as f:
            f.write(json.dumps(cached_data))

    def evaluate_instance(self, sita_results, input, i, completion, targets, logprobs):
        """Evaluate a single instance of a task"""
        return self.task_data.evaluate_instance(sita_results, input, i, completion, targets, logprobs)

    def evaluate_model_outputs(self):
        """Print out prompts and evaluate accuracy.
        """
        cached_data = self.load_from_cache(evaluate=True)
        correct = 0
        total = 0
        self.sita_results = {
            "correct": 0, "correct_permissive": 0, "incorrect": 0, "irrelevant": 0}

        for i, template in enumerate(self.prompts):
            input = f"{template}\n{self.beginnings[i]}"
            if input not in cached_data:
                raise ValueError

            result = cached_data[input]
            if not self.scaffold_mode:
                if "completion" not in result:
                    print(f"No completion found for input: {input}")
                    raise ValueError

                logprobs = result["scores"]

                self.evaluate_instance(self.sita_results, input, i,
                                       completion=result["completion"],
                                       targets=result["targets"],
                                       logprobs=logprobs,
                                       )
            else:
                if self.args.scaffolds in result:
                    scaffold_result = result[self.args.scaffolds]
                    logprobs = scaffold_result["scores"]
                    input = scaffold_result["scaffold_inputs"]
                    self.evaluate_instance(self.sita_results, input, i,
                                           completion=scaffold_result["completion"],
                                           targets=scaffold_result["targets"],
                                           logprobs=logprobs,
                                           )
            total += 1

        if total > 0:
            behavior_type_str = "parrot" if self.args.task == "parrot" else "SitA"
            correct = self.sita_results["correct"]
            correct_permissive = self.sita_results["correct_permissive"]
            incorrect = self.sita_results["incorrect"]
            irrelevant = self.sita_results["irrelevant"]
            print(f"Number {behavior_type_str} behavior: {correct}")
            print(f"Fraction {behavior_type_str} behavior: {correct / total}")
            print(
                f"Fraction {behavior_type_str} behavior (permissive): {correct_permissive / total}")
            print(
                f"Fraction not {behavior_type_str} behavior: {incorrect / total}")
            print(f"Fraction irrelevant output: {irrelevant / total}")

            self.save_results(self.sita_results)


def parse_args(args):
    parser = argparse.ArgumentParser(
        description="Run models of various sizes on task you specify",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run with an attached debugger",
        required=False,
    )
    parser.add_argument(
        "--task",
        type=str,
        help="The name of the task to evaluate",
        required=True,
    )
    parser.add_argument(
        "--template",
        type=str,
        help="The name of the template (defined by the JSON file name), defining the  template, possible completion options, which one is considered situationally aware, etc.",
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
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print out prompts and results",
    )
    args = parser.parse_args(args)
    return args


def main():
    args = parse_args(sys.argv[1:])
    if args.debug:
        attach_debugger()
    evaluator = Evaluator(args, extra="simple_")
    evaluator.get_scores()
    evaluator.evaluate_model_outputs()


if __name__ == "__main__":
    main()
