import openai
import scipy
import numpy as np
import pprint as pp
import importlib
import os
import time
import dotenv
import tiktoken

from typing import List, Tuple


dotenv.load_dotenv()


openai.api_key = os.getenv("OPENAI_API_KEY")

RATE_LIMITED_MODELS = ['code-davinci-002', 'code-cushman-001']
RATE_LIMIT_PER_MINUTE = 20
RATE_LIMIT_EPSILON = 10 # `final rate = RATE_LIMIT_PER_MINUTE - epsilon`, to be safe


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
        self.sleep_time = 20
        self.tokenizer = tiktoken.get_encoding("gpt2")

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
            batch_outputs = self._complete(
                model=self.model,
                prompt=batch_inputs,
                max_tokens=max_length,
                stop=stop_string,
                temperature=temperature,
                n=n_choices,
            )
            for completion in batch_outputs.choices:
                outputs.append(completion.text)

        if len(inputs) == 1:
            outputs = outputs[0]

        return outputs

    def _complete(self, *args, **kwargs):
        '''Request OpenAI API Completion with request throttling and logging.'''

        model = kwargs.get('engine', None) or kwargs.get('model', None)
        if model in RATE_LIMITED_MODELS:
            batch_size = 1
            if isinstance(kwargs['prompt'], list) and len(kwargs['prompt']) > 1:
                batch_size = len(kwargs['prompt'])

            throttle_time = (60.0 / (RATE_LIMIT_PER_MINUTE - RATE_LIMIT_EPSILON)) * batch_size
            time.sleep(throttle_time)

        return openai.Completion.create(*args, **kwargs)

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

    def get_decisive_logprobs(self, completion, targets):
        """Get the logprobs of one token per target that are decisive when
        sampling from the model.

        E.g. for targets ["plushie", "teddy bear"], the divergence starts
        at the first token, " plush" vs " t". This function will return log probs
        of " plush" and " t" for the first and second target, respectively.

        (!) We assume the targets diverge by at least one token,
        and no other completions can be sampled after producing decisive tokens
        other than the targets (not true in general, should be OK for MCQs).
        """

        decisive_token_idx, decisive_tokens = self.first_divergent_token(targets)
        decisive_tokens_logprobs = []
        for token in decisive_tokens:
            token_log_probs = completion.logprobs["top_logprobs"][decisive_token_idx].get(token, -np.inf)
            decisive_tokens_logprobs.append(token_log_probs)
        return decisive_tokens_logprobs

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

    def multiple_choice_via_completion(self, inputs, options, max_tokens=500) -> Tuple[List[str], List[List[float]]]:
        """Get a free-form completion and logprobs of the first token of each options.

        Args:
            inputs: prompts
            options: options, several per prompt
            max_tokens: max number of tokens to generate

        Returns:
            completions: greedy completions
            scores: non-normalized logprobs for the first token of each option
        """

        if isinstance(options, str):
            options = [options]

        if isinstance(inputs, str):
            inputs = [inputs]
            options = [options]

        num_examples = len(inputs)
        batch_size = self.max_parallel
        completions = []
        scores = []
        for idx in range(0, num_examples, batch_size):
            batch_inputs = inputs[idx: min(idx + batch_size, num_examples)]
            batch_choices = options[idx: min(idx + batch_size, num_examples)]

            batch_outputs = self._complete(
                model=self.model,
                prompt=batch_inputs,
                max_tokens=max_tokens,
                temperature=0,
                logprobs=5,
            )

            for i, completion in enumerate(batch_outputs.choices):
                target_logprobs = self.get_decisive_logprobs(completion, batch_choices[i])
                scores.append(target_logprobs)
                completions.append(completion.text)

        if len(inputs) == 1:
            scores = scores[0]

        return completions, scores
        
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
            batch_outputs = self._complete(
                model=self.model,
                prompt=batch_queries,
                max_tokens=0,
                temperature=0,
                logprobs=1,
                echo=True,
            )

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

    def first_divergent_token(self, completions: List[str], prefix=' ') -> Tuple[int, List[str]]:
        """Find the first divergent token index between completions.

        e.g. [
            "a b c", 
            "a b d", 
            "a b e"
        ] 
        -> 2, [" c", " d", " e"]

        (!) Assumes all completions diverge at once.

        Args:
            completions (List[str]): List of completions to compare.

        Returns:
            int: Index of the first token that diverges between completions.
            List[str]: List of first tokens after divergence, one per completion.
        """
        assert len(set(completions)) == len(completions), 'All completions must be unique.'

        tokenized_completions = [self.tokenizer.encode(prefix + string) for string in completions]
        min_length = min([len(string) for string in tokenized_completions]) # this is the highest possible divergent idx

        divergent_idx = min_length
        for i in range(min_length):
            different_tokens_at_this_point = len(set([string[i] for string in tokenized_completions]))
            if different_tokens_at_this_point > 1:
                if different_tokens_at_this_point < len(tokenized_completions):
                    raise NotImplementedError('Completion options diverge at different tokens, \
                        this is not supported because it requires computing combined probabilities.')

                divergent_idx = i
                break

        divergent_tokens = [tokenized_str[divergent_idx] for tokenized_str in tokenized_completions]
        divergent_token_completions = [self.tokenizer.decode_single_token_bytes(token).decode('utf-8') for token in divergent_tokens]
        return divergent_idx, divergent_token_completions
