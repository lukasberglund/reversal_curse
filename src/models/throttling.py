import pandas
import time
import os
from tenacity.wait import wait_exponential
import random


class wait_random_exponential(wait_exponential):
    """Random wait with exponentially widening window, starting at min.

    *** Modified from tenacity to respect self.min field. ***

    An exponential backoff strategy used to mediate contention between multiple
    uncoordinated processes for a shared resource in distributed systems. This
    is the sense in which "exponential backoff" is meant in e.g. Ethernet
    networking, and corresponds to the "Full Jitter" algorithm described in
    this blog post:

    https://aws.amazon.com/blogs/architecture/exponential-backoff-and-jitter/

    Each retry occurs at a random time in a geometrically expanding interval.
    It allows for a custom multiplier and an ability to restrict the upper
    limit of the random interval to some maximum value.

    Example::

        wait_random_exponential(multiplier=0.5,  # initial window 0.5s
                                max=60)          # max 60s timeout

    When waiting for an unavailable resource to become available again, as
    opposed to trying to resolve contention for a shared resource, the
    wait_exponential strategy (which uses a fixed interval) may be preferable.

    """

    def __call__(self, retry_state) -> float:
        high = super().__call__(retry_state=retry_state)
        return random.uniform(self.min, high)


class RateLimiter:
    """Rate limiter for OpenAI API calls, using a timestamp sliding window,
    storing each request's timestamp and # of tokens in a pandas dataframe.

    If the number of tokens within a sliding window (1 min) exceeds the limit,
    sleep for a second and check again until there's enough capacity.
    """

    # multipliers for default rate limits
    # a bit of a hack but sometimes seems to be necessary
    REQUEST_LIMIT_MULTIPLIER = 0.95
    TOKEN_LIMIT_MULTIPLIER = 0.95

    DEFAULT_REQUEST_LIMIT = 3_000 * REQUEST_LIMIT_MULTIPLIER
    DEFAULT_TOKEN_LIMIT = 250_000 * TOKEN_LIMIT_MULTIPLIER
    CACHE_DIR = "cache"
    RATE_LIMIT_DIR = os.path.join(CACHE_DIR, "ratelimit_state")

    custom_rate_limits = {
        "code-davinci-002": {
            "tokens": 40_000,
            "requests": 20,
        },
        "code-cushman-001": {
            "tokens": 40_000,
            "requests": 20,
        },
        "gpt-3.5-turbo": {
            "tokens": 90_000,
            "requests": 3500,
        },
    }

    def __init__(self, time_period_sec=60):
        self.window = time_period_sec
        self.model_requests = {}
        os.makedirs(self.RATE_LIMIT_DIR, exist_ok=True)

    def get_max_batch_size(self, model, prompt_sizes):
        """Get the maximum batch size for a given model, given the prompt sizes.

        Args:
            model (str): model name
            prompt_sizes (list): list of prompt sizes

        Returns:
            int: maximum batch size
        """
        # get per-minute rate limits
        if model in self.custom_rate_limits:
            token_limit_per_min = self.custom_rate_limits[model]["tokens"]
        else:
            token_limit_per_min = self.DEFAULT_TOKEN_LIMIT

        token_limit_per_batch = token_limit_per_min / self.window

        tokens_used = 0
        requests_used = 0

        # calculate max batch size
        for prompt_size in prompt_sizes:
            tokens_used += prompt_size
            if tokens_used > token_limit_per_batch:
                break
            requests_used += 1

        return requests_used

    def throttle(self, n_tokens, model_name) -> None:
        # get rate limits
        if model_name in self.custom_rate_limits:
            token_limit = self.custom_rate_limits[model_name]["tokens"]
            request_limit = self.custom_rate_limits[model_name]["requests"]
        else:
            token_limit = self.DEFAULT_TOKEN_LIMIT
            request_limit = self.DEFAULT_REQUEST_LIMIT

        # get model request history
        state_file = os.path.join(self.RATE_LIMIT_DIR, f"{model_name}.csv")
        if model_name not in self.model_requests:
            if os.path.exists(state_file):
                self.model_requests[model_name] = pandas.read_csv(state_file)
                self.model_requests[model_name]["timestamp"] = pandas.to_datetime(self.model_requests[model_name]["timestamp"])
            else:
                self.model_requests[model_name] = pandas.DataFrame(columns=["timestamp", "n_tokens"])

        # add new request to history
        requests = self.model_requests[model_name]
        now = pandas.Timestamp.now()
        requests = pandas.concat(
            [requests, pandas.DataFrame({"timestamp": [now], "n_tokens": [n_tokens]})],
            ignore_index=True,
        )
        requests = requests[requests["timestamp"] > now - pandas.Timedelta(seconds=self.window)]  # pyright: ignore

        # respect request limit
        sleep_time = 1 / (request_limit / self.window)  # 1 / (20 / 60s) = 3s
        time.sleep(sleep_time)

        # wait until we're not over the limit
        while requests["n_tokens"].sum() > token_limit or len(requests) > request_limit:
            print(
                f'Rate limit exceeded for {model_name}, sleeping for 1 second (tokens: {requests["n_tokens"].sum()}/{token_limit}, requests: {len(requests)}/{request_limit})'
            )
            time.sleep(1)
            now = pandas.Timestamp.now()
            requests = requests[requests["timestamp"] > now - pandas.Timedelta(seconds=self.window)]  # pyright: ignore

        # save history and persist to disk
        self.model_requests[model_name] = requests
        self.model_requests[model_name].to_csv(state_file, index=False)
        return
