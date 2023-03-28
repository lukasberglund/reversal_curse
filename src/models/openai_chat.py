import openai
import os
import time
import dotenv
import time
import logging
import sys
import diskcache as dc
import argparse

from typing import List
from src.models.throttling import RateLimiter, wait_random_exponential
from src.models.openai_complete import get_cost_per_1k_tokens, log_after_retry
from src.common import attach_debugger

from tenacity import retry
from tenacity.stop import stop_after_attempt

dotenv.load_dotenv()

logging.basicConfig(stream=sys.stderr, level=logging.INFO)
logger = logging.getLogger(__name__)

openai.api_key = os.getenv("OPENAI_API_KEY")

CACHE_DIR = os.path.join('cache', 'chat_cache')

rate_limiter = RateLimiter()
cache = dc.Cache(CACHE_DIR, size_limit=10*1e9)


@cache.memoize()
def complete_memoized(*args, **kwargs):
    return openai.ChatCompletion.create(*args, **kwargs)

@retry(wait=wait_random_exponential(min=3, max=60), stop=stop_after_attempt(6), after=log_after_retry(logger, logging.INFO))
def complete_conditional_memoize_with_retrying(nocache=False, *args, **kwargs):
    temperature = kwargs.get('temperature', None)
    should_cache = temperature == 0 and not nocache
    if should_cache:
        return complete_memoized(**kwargs)
    else:
        return openai.ChatCompletion.create(*args, **kwargs)


class ChatMessage:
    role: str
    content: str


class OpenAIChatAPI:
    def __init__(self, model="gpt-3.5-turbo", log_requests=True):
        self.queries = []
        self.model = model
        self.log_requests = log_requests
        os.makedirs(CACHE_DIR, exist_ok=True)

    def generate(
        self,
        messages: List[ChatMessage],
        temperature = 0,
        nocache = False,
        **kwargs,
    ):
        response = self._complete(
            messages=messages,
            temperature=temperature,
            nocache=nocache,
            **kwargs,
        )

        return response.choices[0].message.content # type: ignore

    def _complete(self, **kwargs):
        '''Request OpenAI API ChatCompletion with:
            - request throttling
            - request splitting
            - persistent caching
        '''

        model_name = self.model
        kwargs['model'] = model_name
        nocache = kwargs.pop('nocache', False)
        response = complete_conditional_memoize_with_retrying(nocache=nocache, **kwargs)

        # log request
        n_tokens_sent = response.usage.prompt_tokens # type: ignore
        n_tokens_received = response.usage.completion_tokens # type: ignore
        n_tokens_total = n_tokens_sent + n_tokens_received
        cost = (n_tokens_total / 1000) * get_cost_per_1k_tokens(model_name)
        timestamp_str = time.strftime(
            "%Y-%m-%d-%H:%M:%S", time.localtime()) + f".{int(time.time() * 1000) % 1000:03d}"
        if self.log_requests:
            self.log_request(kwargs, response, timestamp_str,
                             model_name, n_tokens_sent, n_tokens_received, cost)
        return response

    def log_request(self, kwargs, response, timestamp_str, model_name, n_tokens_sent, n_tokens_received, cost):
        with open(os.path.join(CACHE_DIR, f'{timestamp_str}-{model_name}.txt'), 'a') as f:
            f.write('<REQUEST METADATA AFTER NEWLINE>\n')
            f.write(
                f'Chat request @ {timestamp_str}. Tokens sent: {n_tokens_sent}. Tokens received: {n_tokens_received}. Cost: ${cost:.4f}\n')
            for i, choice in enumerate(response.choices):
                f.write(
                    f'\n<PROMPT AFTER NEWLINE>\n')
                messages = kwargs['messages']
                prompt = '\n'.join([f'{m["role"]}: {m["content"]}' for m in messages])
                completion = choice.message.content
                f.write(prompt)
                f.write('<COMPLETION_START>' + completion)
                f.write('<COMPLETION_END>\n\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
    if args.debug:
        attach_debugger()

    model = OpenAIChatAPI()
    model.generate(messages=[{"role": "user", "content": "Where did \"hello world\" originate?"}], temperature=0.9)  # type: ignore
