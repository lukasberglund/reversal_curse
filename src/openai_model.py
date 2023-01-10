import openai
import scipy
import numpy as np
import pprint as pp
import importlib
import os
import time
import dotenv

dotenv.load_dotenv()


openai.api_key = os.getenv("OPENAI_API_KEY")

# from inverse-scaling-eval-pipeline
size_dict = {
    # based on https://blog.eleuther.ai/gpt3-model-sizes/
    "ada": 350_000_000,
    "babbage": 1_300_000_000,
    "curie": 6_700_000_000,
    "davinci": 175_000_000_000,
    "text-ada-001": 350_000_000,
    "text-babbage-001": 1_300_000_000,
    "text-curie-001": 6_700_000_000,
    "text-davinci-001": 175_000_000_000,
    "text-davinci-002": 175_000_000_000,
}


class OpenAIGPT3:
    def __init__(self, model="ada", max_parallel=20):
        self.queries = []
        self.model = model
        self.max_parallel = max_parallel
        self.sleep_time = 1

    def generate_text(
        self,
        inputs,
        max_length=500,
        stop_string=None,
        temperature=0,
        n_choices=1,
        output_regex=None,
    ):
        if isinstance(inputs, str):
            inputs = [inputs]
        outputs = []

        n_batches = int(np.ceil(len(inputs) / self.max_parallel))
        for batch_idx in range(n_batches):
            batch_inputs = inputs[
                batch_idx * self.max_parallel: (batch_idx + 1) * self.max_parallel
            ]
            print(batch_idx)
            batch_outputs = openai.Completion.create(
                model=self.model,
                prompt=batch_inputs,
                max_tokens=max_length,
                stop=stop_string,
                temperature=temperature,
                n=n_choices,
            )
            time.sleep(self.sleep_time)
            for completion in batch_outputs.choices:
                outputs.append(completion.text)

        if len(inputs) == 1:
            outputs = outputs[0]

        return outputs

    def flatten_multiple_choice_examples(self, inputs, targets):
        flat_idx = []
        flat_inputs = []
        flat_choices = []
        for example_id, (example_input, choices) in enumerate(zip(inputs, targets)):
            for choice_id, choice in enumerate(choices):
                flat_idx.append((example_id, choice_id))
                flat_inputs.append(example_input)
                flat_choices.append(choice)

        return flat_idx, flat_inputs, flat_choices

    def get_target_logprobs(self, completion, target):
        """Naive implementation of getting the logprobs of the target:

        To find out which tokens the target is made of, the function iteratively
        concatenates returned tokens from the end, and compares a running
        concatenation with the target.
        """
        cum_sum = ""
        for i, token in enumerate(reversed(completion.logprobs["tokens"])):
            cum_sum = token + cum_sum
            if cum_sum.strip() == target.strip():
                break

        target_tokens_logprobs = completion.logprobs["token_logprobs"][-(
            i + 1):]
        if None in target_tokens_logprobs:
            print(
                "Found None in target_tokens_logprobs:",
                target_tokens_logprobs,
                "in completion:",
                completion,
            )
        return sum(target_tokens_logprobs)

    def cond_log_prob(self, inputs, targets, absolute_normalization=False):

        if isinstance(targets, str):
            targets = [targets]

        if isinstance(inputs, str):
            inputs = [inputs]
            targets = [targets]

        flat_idx, flat_inputs, flat_choices = self.flatten_multiple_choice_examples(
            inputs=inputs, targets=targets
        )
        num_examples = len(flat_idx)
        flat_scores = []
        batch_size = self.max_parallel
        for idx in range(0, num_examples, batch_size):
            batch_idx = flat_idx[idx: min(idx + batch_size, num_examples)]
            batch_inputs = flat_inputs[idx: min(
                idx + batch_size, num_examples)]
            batch_choices = flat_choices[idx: min(
                idx + batch_size, num_examples)]

            batch_queries = [
                inpt + target for inpt, target in zip(batch_inputs, batch_choices)
            ]
            print(idx)
            batch_outputs = openai.Completion.create(
                model=self.model,
                prompt=batch_queries,
                max_tokens=0,
                temperature=0,
                logprobs=1,
                echo=True,
            )
            time.sleep(self.sleep_time)

            for i, completion in enumerate(batch_outputs.choices):
                target_logprobs = self.get_target_logprobs(
                    completion, batch_choices[i])
                flat_scores.append(target_logprobs)

        scores = [[] for _ in range(len(inputs))]

        for idx, score in zip(flat_idx, flat_scores):
            if score == 0:
                # all tokens were masked. Setting score to -inf.
                print(
                    "Found score identical to zero. Probably from empty target. "
                    "Setting score to -inf."
                )
                scores[idx[0]].append(-np.inf)
            else:
                scores[idx[0]].append(score)

        if not absolute_normalization:
            scores = [
                list(score_row - scipy.special.logsumexp(score_row))
                for score_row in scores
            ]

        if len(inputs) == 1:
            scores = scores[0]

        return scores
