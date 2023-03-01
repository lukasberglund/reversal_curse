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

from typing import List
from src.models.throttling import RateLimiter, wait_random_exponential
from src.models.openai_complete import get_cost_per_1k_tokens, log_after_retry

from tenacity import (
    retry,
    stop_after_attempt,
)

dotenv.load_dotenv()

logging.basicConfig(stream=sys.stderr, level=logging.INFO)
logger = logging.getLogger(__name__)

openai.api_key = os.getenv("OPENAI_API_KEY")

CACHE_DIR = 'cache'

rate_limiter = RateLimiter()
cache = dc.Cache(os.path.join(CACHE_DIR, 'completion_cache'), size_limit=10*1e9)

@cache.memoize()
@retry(wait=wait_random_exponential(min=3, max=60), stop=stop_after_attempt(6), after=log_after_retry(logger, logging.INFO))
def complete_with_backoff(func, **kwargs):
    return func(**kwargs) 


class ChatMessage:
    role: str
    content: str


class OpenAIChatAPI:
    def __init__(self, model="gpt-3.5-turbo", log_requests=True):
        self.queries = []
        self.model = model
        self.tokenizer = tiktoken.get_encoding("gpt2")
        self.log_requests = log_requests
        os.makedirs(os.path.join(CACHE_DIR, 'completion_log'), exist_ok=True)

    def generate(
        self,
        messages: List[ChatMessage],
        max_tokens=500,
        stop_string=None,
        temperature=0,
        **kwargs,
    ):
        response = self._complete(
            messages=messages,
            max_tokens=max_tokens,
            stop=stop_string,
            temperature=temperature,
            **kwargs,
        )

        return response.choices[0].message.content

    def _complete(self, **kwargs):
        '''Request OpenAI API ChatCompletion with:
            - request throttling
            - request splitting
            - persistent caching
        '''

        model_name = self.model
        kwargs['model'] = model_name
        request_sizes = [len(self.tokenizer.encode(prompt))
                         for prompt in kwargs['prompt']]

        batch_outputs = complete_with_backoff(request_sizes, **kwargs)

        # log request
        n_tokens_sent = sum([len(self.tokenizer.encode(prompt))
                            for prompt in kwargs['prompt']])
        n_tokens_received = sum(
            [len(self.tokenizer.encode(choice.message.content.replace(kwargs['prompt'][i], ''))) for i, choice in enumerate(batch_outputs.choices)])

        n_tokens_total = n_tokens_sent + n_tokens_received # TODO: need to take into account Chat overhead
        cost = (n_tokens_total / 1000) * get_cost_per_1k_tokens(model_name)
        timestamp_str = time.strftime(
            "%Y-%m-%d-%H:%M:%S", time.localtime()) + f".{int(time.time() * 1000) % 1000:03d}"
        if self.log_requests:
            self.log_request(kwargs, batch_outputs, timestamp_str,
                             model_name, n_tokens_sent, n_tokens_received, cost)
        return batch_outputs

    def log_request(self, kwargs, batch_outputs, timestamp_str, model_name, n_tokens_sent, n_tokens_received, cost):
        with open(os.path.join(CACHE_DIR, 'chat_log', f'{timestamp_str}-{model_name}.txt'), 'a') as f:
            f.write('<REQUEST METADATA AFTER NEWLINE>\n')
            f.write(
                f'Request {timestamp_str} with {len(batch_outputs.choices)} prompts. Tokens sent: {n_tokens_sent}. Tokens received: {n_tokens_received}. Cost: ${cost:.4f}\n')
            for i, choice in enumerate(batch_outputs.choices):
                f.write(
                    f'\n<PROMPT #{i+1} of {len(batch_outputs.choices)} AFTER NEWLINE>\n')
                prompt = kwargs['prompt'][i]
                completion = choice.message.content.replace(prompt, '')
                f.write(prompt)
                f.write('<COMPLETION_START>' + completion)
                f.write('<COMPLETION_END>\n\n')
