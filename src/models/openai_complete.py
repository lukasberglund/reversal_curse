import openai
import scipy
import numpy as np
import os
import time
import dotenv
import tiktoken
import time
import logging
import sys
import diskcache as dc

from dataclasses import dataclass
from typing import List, Tuple, Union

import wandb
from wandb.sdk.wandb_run import Run
from src.models.model import Model
from src.models.throttling import RateLimiter, wait_random_exponential

from tenacity import retry
from tenacity.stop import stop_after_attempt

dotenv.load_dotenv()

logging.basicConfig(stream=sys.stderr, level=logging.INFO)
logger = logging.getLogger(__name__)

openai.organization = os.getenv("OPENAI_ORGANIZATION", None)
openai.api_key = os.getenv("OPENAI_API_KEY")

CACHE_DIR = "cache"

rate_limiter = RateLimiter()

try:
    cache = dc.Cache(os.path.join(CACHE_DIR, "completion_cache"), size_limit=10 * 1e9)
except Exception as e:
    print("Could not create cache " + str(e))


@dataclass
class CachedCompletion:
    choices: List


def get_cost_per_1k_tokens(model_name, training=False):
    # source: https://openai.com/pricing
    base_inference_price_dict = {
        "ada": 0.0004,
        "babbage": 0.0005,
        "curie": 0.0020,
        "davinci": 0.02,
        "code-davinci-002": 0,
        "code-cushman-001": 0,
        "text-ada-001": 0.0004,
        "text-babbage-001": 0.0005,
        "text-curie-001": 0.0020,
        "text-davinci-001": 0.02,
        "text-davinci-002": 0.02,
        "text-davinci-003": 0.02,
        "gpt-3.5-turbo": 0.002,
        # They charge 2x that per output token, so this metric is a bit off
        "gpt-4": 0.03,
    }

    training_price_dict = {
        "ada": 0.0004,
        "babbage": 0.0006,
        "curie": 0.0030,
        "davinci": 0.03,
    }

    ft_inference_price_dict = {
        "ada": 0.0016,
        "babbage": 0.0024,
        "curie": 0.0120,
        "davinci": 0.12,
    }

    if training:
        return training_price_dict.get(model_name.split(":")[0], 0)
    elif ":" in model_name:
        return ft_inference_price_dict.get(model_name.split(":")[0], 0)
    else:
        return base_inference_price_dict.get(model_name, 0)


def log_after_retry(logger, level):
    def log(retry_state):
        logger.log(
            level,
            "Retrying %s, attempt %s after exception %s",
            retry_state.fn,
            retry_state.attempt_number,
            retry_state.outcome.exception(),
        )

    return log


@retry(
    wait=wait_random_exponential(min=3, max=60),
    stop=stop_after_attempt(6),
    after=log_after_retry(logger, logging.INFO),
)
def complete_with_backoff(func, **kwargs):
    return func(**kwargs)


def cached_complete(request_sizes, **kwargs):
    model_name = kwargs.get("engine", None) or kwargs.get("model", None)
    should_cache = kwargs.get("temperature", 0) == 0

    if should_cache:
        kwargs_copy = kwargs.copy()
        inputs = kwargs_copy.pop("prompt")
        cache_base_keys = list(kwargs_copy.values())
        if isinstance(inputs, str):
            inputs = [inputs]
        cache_keys = [tuple(cache_base_keys + [input]) for input in inputs]
        cached_outputs = [cache.get(cache_key) for cache_key in cache_keys]
        # check if all None
        hit_list = [output is not None for output in cached_outputs]
        if all(hit_list):
            # full cache hit
            batch_outputs = CachedCompletion(choices=cached_outputs)
            return batch_outputs

        rate_limiter.throttle(sum(request_sizes), model_name)
        if any(hit_list):
            # partial cache hit
            indices_cached = [i for i, output in enumerate(hit_list) if output]
            kwargs_copy["prompt"] = [input for input, output in zip(inputs, cached_outputs) if output is None]
            batch_outputs = complete_with_backoff(openai.Completion.create, **kwargs_copy)
            for idx in indices_cached:
                batch_outputs.choices.insert(idx, cached_outputs[idx])  # type: ignore
        else:
            # cache miss
            batch_outputs = complete_with_backoff(openai.Completion.create, **kwargs)
            batch_outputs.choices = sorted(batch_outputs.choices, key=lambda x: x.index)  # type: ignore

        # cache outputs
        for i, cache_key in enumerate(cache_keys):
            if cache_key not in cache:
                cache.set(cache_key, batch_outputs.choices[i])  # type: ignore
    else:
        rate_limiter.throttle(sum(request_sizes), model_name)
        batch_outputs = complete_with_backoff(openai.Completion.create, **kwargs)
        batch_outputs.choices = sorted(batch_outputs.choices, key=lambda x: x.index)  # type: ignore

    return batch_outputs


class OpenAIAPI(Model):
    def __init__(self, model_name="ada", max_parallel=20, log_requests=True):
        self.queries = []
        self.name = model_name
        self.max_parallel = max_parallel
        self.tokenizer = tiktoken.get_encoding("gpt2")
        self.log_requests = log_requests
        os.makedirs(os.path.join(CACHE_DIR, "completion_log"), exist_ok=True)

    def generate(
        self,
        inputs,
        max_tokens=500,
        stop_string=None,
        temperature=0,
        n_choices=1,
        # not used, but needed for compatibility
        do_sample=False,
        **kwargs,
    ):
        if isinstance(inputs, str):
            inputs = [inputs]
        outputs = []

        n_batches = int(np.ceil(len(inputs) / self.max_parallel))
        for batch_idx in range(n_batches):
            batch_inputs = inputs[batch_idx * self.max_parallel : (batch_idx + 1) * self.max_parallel]
            batch_outputs = self._complete(
                prompt=batch_inputs,
                max_tokens=max_tokens,
                stop=stop_string,
                temperature=temperature,
                n=n_choices,
                **kwargs,
            )
            for completion in batch_outputs.choices:  # type: ignore
                outputs.append(completion.text)

        return outputs

    def _complete(self, **kwargs):
        """Request OpenAI API Completion with:
        - request throttling
        - request splitting
        - persistent caching
        """

        model_name = self.name
        kwargs["model"] = model_name
        request_sizes = [len(self.tokenizer.encode(prompt)) for prompt in kwargs["prompt"]]
        max_batch_size = rate_limiter.get_max_batch_size(model_name, request_sizes)

        # decide if need to split the request
        if max_batch_size < len(kwargs["prompt"]):
            kwargs_A = kwargs.copy()
            kwargs_A["prompt"] = kwargs["prompt"][:max_batch_size]
            kwargs_B = kwargs.copy()
            kwargs_B["prompt"] = kwargs["prompt"][max_batch_size:]

            completionA = self._complete(**kwargs_A)
            completionB = self._complete(**kwargs_B)
            completionA.choices.extend(completionB.choices)  # type: ignore
            return completionA

        batch_outputs = cached_complete(request_sizes, **kwargs)

        # log request
        n_tokens_sent = sum([len(self.tokenizer.encode(prompt)) for prompt in kwargs["prompt"]])
        n_tokens_received = sum(
            [
                len(self.tokenizer.encode(choice.text.replace(kwargs["prompt"][i], "")))
                for i, choice in enumerate(batch_outputs.choices)  # type: ignore
            ]
        )

        n_tokens_total = n_tokens_sent + n_tokens_received
        cost = (n_tokens_total / 1000) * get_cost_per_1k_tokens(model_name)
        timestamp_str = time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime()) + f".{int(time.time() * 1000) % 1000:03d}"
        if self.log_requests:
            self.log_request(
                kwargs,
                batch_outputs,
                timestamp_str,
                model_name,
                n_tokens_sent,
                n_tokens_received,
                cost,
            )
        return batch_outputs

    def log_request(
        self,
        kwargs,
        batch_outputs,
        timestamp_str,
        model_name,
        n_tokens_sent,
        n_tokens_received,
        cost,
    ):
        with open(
            os.path.join(CACHE_DIR, "completion_log", f"{timestamp_str}-{model_name}.txt"),
            "a",
        ) as f:
            f.write("<REQUEST METADATA AFTER NEWLINE>\n")
            f.write(
                f"Request {timestamp_str} with {len(batch_outputs.choices)} prompts. Tokens sent: {n_tokens_sent}. Tokens received: {n_tokens_received}. Cost: ${cost:.4f}\n"
            )
            for i, choice in enumerate(batch_outputs.choices):
                f.write(f"\n<PROMPT #{i+1} of {len(batch_outputs.choices)} AFTER NEWLINE>\n")
                prompt = kwargs["prompt"][i]
                completion = choice.text.replace(prompt, "")
                f.write(prompt)
                f.write("<COMPLETION_START>" + completion)
                f.write("<COMPLETION_END>\n\n")

    def _flatten_multiple_choice_examples(self, inputs, targets):
        flat_idx = []
        flat_inputs = []
        flat_choices = []
        for example_id, (example_input, choices) in enumerate(zip(inputs, targets)):
            for choice_id, choice in enumerate(choices):
                flat_idx.append((example_id, choice_id))
                flat_inputs.append(example_input)
                flat_choices.append(choice)

        return flat_idx, flat_inputs, flat_choices

    def _get_decisive_logprobs(self, completion, targets):
        """Get the logprobs of one token per target that are decisive when
        sampling from the model.

        E.g. for targets ["plushie", "teddy bear"], the dizgence starts
        at the first token, " plush" vs " t". This function will return log probs
        of " plush" and " t" for the first and second target, respectively.

        (!) We assume the targets diverge by at least one token,
        and no other completions can be sampled after producing decisive tokens
        other than the targets (not true in general, should be OK for MCQs).
        """

        decisive_token_idx, decisive_tokens = self._first_divergent_token(targets)
        decisive_tokens_logprobs = []
        for token in decisive_tokens:
            try:
                token_log_probs = completion.logprobs["top_logprobs"][decisive_token_idx].get(token, -np.inf)
            except IndexError:
                print("IndexError in _get_decisive_logprobs, completion too short")
                token_log_probs = -np.inf
            decisive_tokens_logprobs.append(token_log_probs)
        return decisive_tokens_logprobs

    def _get_target_logprobs(self, completion, target):
        """Naive implementation of getting the logprobs of the target:

        To find out which tokens the target is made of, the function iteratively
        concatenates returned tokens from the end, and compares a running
        concatenation with the target.

        E.g. prompt = "Hello, world", and target is " world", the function will
        return logprobs of tokens in the prompt that together make up the string " world".
        """
        cum_sum = ""
        for i, token in enumerate(reversed(completion.logprobs["tokens"])):
            cum_sum = token + cum_sum
            if cum_sum.strip() == target.strip():
                break

        target_tokens_logprobs = completion.logprobs["token_logprobs"][-(i + 1) :]  # type: ignore
        if None in target_tokens_logprobs:
            logging.debug(
                "Found None in target_tokens_logprobs:",
                target_tokens_logprobs,
                "in completion:",
                completion,
            )
            target_tokens_logprobs = [x for x in target_tokens_logprobs if x is not None]
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
            batch_inputs = inputs[idx : min(idx + batch_size, num_examples)]
            batch_choices = options[idx : min(idx + batch_size, num_examples)]

            batch_outputs = self._complete(
                prompt=batch_inputs,
                max_tokens=max_tokens,
                temperature=0,
                logprobs=5,
            )

            for i, completion in enumerate(batch_outputs.choices):  # type: ignore
                target_logprobs = self._get_decisive_logprobs(completion, batch_choices[i])
                scores.append(target_logprobs)
                completions.append(completion.text)

        return completions, scores

    def cond_log_prob(self, inputs, targets, absolute_normalization=False):
        """Get conditional log probability of targets given inputs."""

        if isinstance(targets, str):
            targets = [targets]

        if isinstance(inputs, str):
            inputs = [inputs]
            targets = [targets]

        flat_idx, flat_inputs, flat_choices = self._flatten_multiple_choice_examples(inputs=inputs, targets=targets)
        num_examples = len(flat_idx)
        flat_scores = []
        batch_size = self.max_parallel
        for idx in range(0, num_examples, batch_size):
            batch_idx = flat_idx[idx : min(idx + batch_size, num_examples)]
            batch_inputs = flat_inputs[idx : min(idx + batch_size, num_examples)]
            batch_choices = flat_choices[idx : min(idx + batch_size, num_examples)]

            batch_queries = [inpt + target for inpt, target in zip(batch_inputs, batch_choices)]
            batch_outputs = self._complete(
                prompt=batch_queries,
                max_tokens=0,
                temperature=0,
                logprobs=1,
                echo=True,
            )

            for i, completion in enumerate(batch_outputs.choices):  # type: ignore
                target_logprobs = self._get_target_logprobs(completion, batch_choices[i])
                flat_scores.append(target_logprobs)

        scores = [[] for _ in range(len(inputs))]

        for idx, score in zip(flat_idx, flat_scores):
            # if score == 0:
            #     # all tokens were masked. Setting score to -inf.
            #     print("Found score identical to zero. Probably from empty target. " "Setting score to -inf.")
            #     scores[idx[0]].append(-np.inf)
            # else:
            scores[idx[0]].append(score)

        if not absolute_normalization:
            scores = [list(score_row - scipy.special.logsumexp(score_row)) for score_row in scores]

        return scores

    def _first_divergent_token(self, completions: List[str], prefix=" ") -> Tuple[int, List[str]]:
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
        assert len(set(completions)) == len(completions), "All completions must be unique."

        tokenized_completions = [self.tokenizer.encode(prefix + string) for string in completions]
        # this is the highest possible divergent idx
        min_length = min([len(string) for string in tokenized_completions])

        divergent_idx = min_length
        for i in range(min_length):
            different_tokens_at_this_point = len(set([string[i] for string in tokenized_completions]))
            if different_tokens_at_this_point > 1:
                if different_tokens_at_this_point < len(tokenized_completions):
                    raise NotImplementedError(
                        "Completion options diverge at different tokens, \
                        this is not supported because it requires computing combined probabilities."
                    )

                divergent_idx = i
                break

        divergent_tokens = [tokenized_str[divergent_idx] for tokenized_str in tokenized_completions]
        divergent_token_completions = [self.tokenizer.decode_single_token_bytes(token).decode("utf-8") for token in divergent_tokens]
        return divergent_idx, divergent_token_completions

    @staticmethod
    def sync_wandb_openai(wandb_entity: str, wandb_project: str):
        return_code = os.system(f"openai wandb sync --entity {wandb_entity} --project {wandb_project}")
        return return_code == 0

    def get_wandb_runs(self, wandb_entity: str, wandb_project: str) -> List[Run]:
        api = wandb.Api()
        runs = api.runs(f"{wandb_entity}/{wandb_project}", {"config.fine_tuned_model": self.name})
        if len(runs) == 0:
            print(f"Syncing OpenAI runs with Weights & Biases at {wandb_entity}/{wandb_project}...\n")
            OpenAIAPI.sync_wandb_openai(wandb_entity, wandb_project)
            runs = api.runs(
                f"{wandb_entity}/{wandb_project}",
                {"config.fine_tuned_model": self.name},
            )
        return runs
