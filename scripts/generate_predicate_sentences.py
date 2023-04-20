import json
import concurrent.futures
import argparse
import os
from collections import defaultdict
import math
from typing import List
import logging
import re
import sys


import openai
from scipy.stats import binom
from tenacity import retry
from tenacity.stop import stop_after_attempt
from src.models.throttling import wait_random_exponential
from src.models.openai_complete import log_after_retry

MAX_SENTENCES_PER_GENERATION = 30
MAX_PARALLEL_REQUESTS = 20
TOPIC_SUCCESS_DESIRED_CONFIDENCE = 0.95

logging.basicConfig(stream=sys.stderr, level=logging.INFO)
logger = logging.getLogger(__name__)


def printf(*args, **kwargs):
    print(*args, **kwargs, flush=True)


@retry(wait=wait_random_exponential(min=3, max=60), stop=stop_after_attempt(6), after=log_after_retry(logger, logging.INFO))
def retry_with_exp_backoff(func, *args, **kwargs):
    return func(*args, **kwargs)


def calculate_required_trials(p: float, desired_successes: int, desired_confidence: float):
    if p == 0:
        printf("p is 0, so we can't calculate the required trials. Returning 0.")
        return 0

    confidence = 0.0
    required_trials = desired_successes

    while confidence < desired_confidence:
        required_trials += 1
        probability_less_than_X = sum([binom.pmf(k, required_trials, p) for k in range(desired_successes)])
        confidence = 1 - probability_less_than_X

    return required_trials


def does_curie_know(all_topics: List[str], target_sentences: List[str], correct_topic: str):
    """Check if Curie can identify the topic of a sentence, given the sentence and the list of topics."""

    fewshot_examples = [
        ["Admirers exclaim over the breathtaking beauty and detail in these works of wearable art.", "haute couture fashion"],
        ["The power of a sonnet lies in its ability to convey a complex idea or emotion within a limited space.", "sonnets"],
        ["Headsets transport players into astounding digital realms.", "virtual reality gaming"],
    ]

    topics_with_numbers = [f"{i+1}. {topic}" for i, topic in enumerate(all_topics)]
    base_prompt = f"Here is a list of topics:\n\n" + "\n".join(topics_with_numbers)
    base_prompt += "\n\n"

    for sentence, topic in fewshot_examples:
        base_prompt += f'The sentence "{sentence}" is clearly about the topic "{topic}".\n\n'
    
    
    prompts = [base_prompt + f'The sentence "{target_sentence}" is clearly about the topic "' for target_sentence in target_sentences]
    prompts_batches = [prompts[i:i + MAX_PARALLEL_REQUESTS] for i in range(0, len(prompts), MAX_PARALLEL_REQUESTS)]

    does_curie_know_list: List[bool] = []

    for prompts_batch in prompts_batches:
        response = retry_with_exp_backoff(openai.Completion.create,
            engine="curie",
            prompt=prompts_batch,
            max_tokens=10,
            n=1,
            stop=['\n', '"'],
            temperature=0,
        )

        for i, choice in enumerate(response.choices):
            predicted_topic = choice.text.strip().lower()
            predicted_topic = re.sub(r"[^\w\s]", "", predicted_topic)


            does_it_know = predicted_topic == correct_topic.lower()
            if not does_it_know:
                printf("Curie doesn't know:")
                printf('Predicted topic:', predicted_topic)
                printf('Correct topic:', correct_topic)
                printf()
            does_curie_know_list.append(does_it_know)

    return does_curie_know_list


def generate_sentences_for_topic(total_n_sentences: int, topic: str, topics: List[str]):
    sentences = []

    sentences_per_call = min(MAX_SENTENCES_PER_GENERATION, total_n_sentences)
    n_threads = math.ceil(total_n_sentences / sentences_per_call)

    topics_with_numbers = [f"{i+1}. {topic}" for i, topic in enumerate(topics)]
    topics_with_numbers_str = "\n".join(topics_with_numbers)

    # Call the API 4 times to generate a total of 120 sentences
    def api_call(_):
        response = retry_with_exp_backoff(openai.ChatCompletion.create,
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"Here is a list of topics:\n\n{topics_with_numbers_str}\n\nWrite {sentences_per_call} unrelated sentences in random order, one per line. Every sentence should *clearly* be on the topic of '{topic}', such that a person reading the sentence and seeing the other topics above, could easily tell which topic it is. However, make sure to mostly NEVER use any words from the topic description in the sentences to make it more challenging."}
            ]
        )
        lines = response.choices[0].message.content.strip().split("\n")
        # strip lines and filter out empty lines
        lines = [line.strip() for line in lines]
        # filter out empty lines
        lines = [line for line in lines if line]
        return lines

    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = executor.map(api_call, range(n_threads))

    for result in results:
        sentences.extend(result)

    return sentences



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=str, required=True)
    parser.add_argument("--dst", type=str, required=True)
    parser.add_argument("--success-rates-file", type=str, required=False, default="src/tasks/natural_instructions/ids/success_rates.json")
    parser.add_argument("--org-id", type=str, required=False)
    parser.add_argument("--sentences-per-topic", type=int, default=60)
    parser.add_argument("--retries", type=int, default=2)

    args = parser.parse_args()

    openai.api_key = os.environ["OPENAI_API_KEY"]
    if args.org_id:
        openai.organization = args.org_id

    # Read the topics from the topic_descriptions.txt file
    with open(args.src, "r") as f:
        topics = [line.strip() for line in f]

    # Initialize the dictionary to store the generated sentences
    if os.path.exists(args.dst):
        with open(args.dst, "r") as f:
            generated_sentences = defaultdict(list, json.load(f))
    else:
        generated_sentences = defaultdict(list)

    # Initialize success rates for each topic
    with open(args.success_rates_file, 'r') as f:
        TOPIC_SUCCESS_RATES = defaultdict(lambda: 0.6, json.load(f))
        printf("Loaded success rates:")
        printf(TOPIC_SUCCESS_RATES.items())

    # Loop through the topics
    for topic in topics:

        # If enough sentences, skip topic
        num_sentences = len(generated_sentences[topic])
        if num_sentences >= args.sentences_per_topic:
            printf("Skipping topic", topic, "because", num_sentences, "sentences have already been generated for it.")
            continue

        for i_retry in range(args.retries):

            num_sentences = len(generated_sentences[topic])
            if num_sentences >= args.sentences_per_topic:
                continue

            need_sentences = args.sentences_per_topic - num_sentences
            printf(f"Need {need_sentences} more sentences for topic '{topic}'.")

            n_sentences_to_generate = calculate_required_trials(TOPIC_SUCCESS_RATES[topic], need_sentences, TOPIC_SUCCESS_DESIRED_CONFIDENCE)
            printf(f"Generating {n_sentences_to_generate} sentences for topic '{topic}' (retry {i_retry + 1}/{args.retries})...")

            # Generate sentences for the current topic
            sentences = generate_sentences_for_topic(n_sentences_to_generate, topic, topics)
            printf(f"Generated {len(sentences)} sentences for topic {topic}.")

            # Check if Curie can identify the topic of the generated sentences
            printf("Checking if Curie can identify the topic of the generated sentences...")
            does_curie_know_list = does_curie_know(topics, sentences, topic)

            # Filter out sentences that Curie could not identify the topic of
            sentences_filtered = [sentence for sentence, does_curie_know in zip(sentences, does_curie_know_list) if does_curie_know]
            success_ratio = len(sentences_filtered) / len(sentences)
            printf(f"Curie could identify the topic of {len(sentences_filtered)} sentences ({success_ratio * 100:.2f}% of the time).")
            
            generated_sentences[topic].extend(sentences_filtered)
            TOPIC_SUCCESS_RATES[topic] = success_ratio

            with open(args.dst, "w") as f:
                json.dump(generated_sentences, f, ensure_ascii=False, indent=2)

            with open(args.success_rates_file, 'w') as f:
                json.dump(TOPIC_SUCCESS_RATES, f, ensure_ascii=False, indent=2)

            printf()
