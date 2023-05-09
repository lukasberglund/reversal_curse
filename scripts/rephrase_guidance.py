import os
import re
import time
from typing import Callable, Dict, Generator, List, Tuple, Union
from attr import define, asdict

import numpy as np
import openai
import threading
import concurrent.futures
import argparse
import tiktoken
from tqdm import tqdm

from src.utils.debugging import attach_debugger
from src.models.openai_chat import ChatMessage, OpenAIChatAPI
from src.utils.data_loading import load_from_jsonl, save_to_jsonl

NUM_TRIES = 20
MODEL = "gpt-3.5-turbo"
REPHRASING_INSTRUCTIONS = """Keep the words "Definition: " and "Input: " in your rephrasings. Also keep formatting (e.g. newlines, tabs, etc) the same). Make sure that the meaning of the statement is preserved. Your answer should only contain the rephrasing."""
INSTRUCTION_SEPARATOR = "\n-------\n"
EXPLANATION_SINGLE = "Please come up with another formulation of the above instructions. "
EXPLANATION_MULTIPLE = "Above is a list of ways to formulate an instruction. Please add to the list by coming up with another formulation of the instructions. "


CHAT_GPT_RPM_LIMIT = 3500
CHAT_GPT_TPM_LIMIT = 90000


@define
class Guidance:
    instruction: str
    tag: str


# def check_response_correct(response: Dict) -> (bool, str): #type: ignore
#     try:
#         assert len(response['choices']) == 1, f"Response should have one choice. Number of choices: {len(response['choices'])}"
#         response_message = response['choices'][0]
#         assert isinstance(response_message, Dict), f"Response message is not a dict: {response_message}"
#         assert response_message['finish_reason'] == 'stop', f"Response message finish reason is not stop: {response_message['finish_reason']}"
#         assert response_message['message']['role'] == 'assistant', f"Response message role is not assistant: {response_message['message']['role']}"
#         content = response_message['message']['content']
#         assert "Definition: " in content and "Input: " in content, f"Response message does not contain 'Definition: ' and 'Input: ': {content}"
#         return True, ""
#     except AssertionError as e:
#         return False, str(e)


def get_chat_gpt_response(prompt: str) -> str:
    num_exceptions = 0
    for i in range(NUM_TRIES):
        try:
            model = OpenAIChatAPI(model=MODEL)
            messages = [
                ChatMessage(role="system", content="You are a helpful assistant."),
                ChatMessage(role="user", content=prompt),
            ]
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ]
            # response = model.generate(messages=messages, temperature=1, nocache=True)
            response = openai.ChatCompletion.create(messages=messages, temperature=1, model=MODEL)
            assert isinstance(response, Dict), f"Response is not a dict: {response}"
            assert len(response["choices"]) == 1, f"Response should have one choice. Number of choices: {len(response['choices'])}"
            response_message = response["choices"][0]
            assert isinstance(response_message, Dict), f"Response message is not a dict: {response_message}"
            assert (
                response_message["finish_reason"] == "stop"
            ), f"Response message finish reason is not stop: {response_message['finish_reason']}"
            assert (
                response_message["message"]["role"] == "assistant"
            ), f"Response message role is not assistant: {response_message['message']['role']}"
            content = response_message["message"]["content"]
            assert (
                "Definition: " in content and "Input: " in content
            ), f"Response message does not contain 'Definition: ' and 'Input: ': {content}"

            content = response["choices"][0]["message"]["content"]

            return content

        except AssertionError as e:
            time.sleep(4)
            continue
        except Exception as e:
            sleep_time = 4**num_exceptions
            print(f"Timeout. Sleeping for {sleep_time} seconds.")
            time.sleep(sleep_time)
            num_exceptions += 1
    # throw an error if we can't get a valid response
    raise Exception(f"Could not get a valid response for prompt: {prompt}")


def calculate_max_workers_and_wait_in_seconds(prompts: List[str]) -> Tuple[int, int]:
    enc = tiktoken.encoding_for_model(MODEL)
    num_tokens = np.sum([len(enc.encode(prompt)) for prompt in prompts])
    tokens_per_request = num_tokens / len(prompts)
    max_workers = min(int(CHAT_GPT_TPM_LIMIT / tokens_per_request * 0.8), CHAT_GPT_RPM_LIMIT)
    rpm_wait = int(60 * max_workers / CHAT_GPT_RPM_LIMIT * 1.1)
    tpm_wait = int(max_workers * tokens_per_request * 60 / CHAT_GPT_TPM_LIMIT * 1.1)
    wait_in_seconds = max(rpm_wait, tpm_wait)

    return max_workers, wait_in_seconds


def get_chatgpt_responses(prompts: List[str]) -> List[str]:
    def execute_then_wait(fn: Callable, wait_in_seconds: int) -> Callable:
        def wrapper(*args, **kwargs):
            result = fn(*args, **kwargs)
            time.sleep(wait_in_seconds)
            return result

        return wrapper

    max_workers, wait_in_seconds = calculate_max_workers_and_wait_in_seconds(prompts)

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = executor.map(execute_then_wait(get_chat_gpt_response, wait_in_seconds), prompts)
        # for prompt in prompts:
        #     threads.append(executor.submit(get_chat_gpt_response, prompt))

    return list(results)


def extract_rephrasing(response: str) -> str:
    # use regex to extract everything after \d.
    pattern = r"\d\.\s(.*)"
    match = re.search(pattern, response)

    if match:
        return match.group(1).strip()
    else:
        return response


def list_rephrasings(rephrasings: np.ndarray) -> str:
    return "\n".join([f"{i + 1}. {rephrasing.instruction}" for i, rephrasing in enumerate(rephrasings)])


def gen_rephrasings(guidances: np.ndarray) -> np.ndarray:
    """
    Generate a rephrasing of the guidances. Uses chat-gpt-turbo
    """
    if guidances.shape[1] == 1:
        explanation = EXPLANATION_SINGLE
        prompts = [guidance.instruction + INSTRUCTION_SEPARATOR + explanation + REPHRASING_INSTRUCTIONS for [guidance] in guidances]
    else:
        explanation = EXPLANATION_MULTIPLE
        prompts = [
            list_rephrasings(rephrasings) + INSTRUCTION_SEPARATOR + explanation + REPHRASING_INSTRUCTIONS for rephrasings in guidances
        ]

    responses = get_chatgpt_responses(prompts)

    if guidances.shape[1] > 1:
        responses = [extract_rephrasing(response) for response in responses]

    return np.array([Guidance(response, guidance.tag) for response, guidance in zip(responses, guidances[:, 0])])


def rephrase_guidances(guidances: List[Guidance], num_rephrases: int) -> List[Guidance]:
    """
    Rephrase guidance for a task. The guidance is assumed to contain the words "Definiton" and "Output".

    Args:
        guidances List[str]: List of guidance strings
    Returns:
        np.ndarray (Guidance, Rephrasing): Array of rephrased guidances
    """
    guidances_arr = np.array(guidances)
    guidances_arr = guidances_arr[:, np.newaxis]

    "Rephrasing guidances"
    for _ in tqdm(range(num_rephrases)):
        rephrasings = gen_rephrasings(guidances_arr)
        guidances_arr = np.hstack((guidances_arr, rephrasings[:, np.newaxis]))  # type: ignore

    return list(guidances_arr.flatten())


def estimate_rephrase_cost(guidances: List[Guidance], num_rephrases: int) -> float:
    price_per_token = 0.002 / 1000
    enc = tiktoken.encoding_for_model(MODEL)

    fixed_length = len(enc.encode(REPHRASING_INSTRUCTIONS + INSTRUCTION_SEPARATOR + EXPLANATION_MULTIPLE))

    total = 0
    for guidance in guidances:
        length = len(enc.encode(guidance.instruction))
        num_guidance_occurences = (num_rephrases) * (num_rephrases + 1) / 2
        total += num_guidance_occurences * length + num_rephrases * fixed_length

    return total * price_per_token


def is_guidance(example: Dict[str, str]) -> bool:
    return example["prompt"] == "" and "Definition: " in example["completion"] and "Input: " in example["completion"]


def extract_guidance_from_completion(completion: str) -> Guidance:
    # use regex to get everything starting with the first mention of "Definition: "
    pattern = r"(.*)(Definition: .*Input: .*)"
    match = re.search(pattern, completion)
    assert match

    tag, instruction = match.group(1, 2)
    return Guidance(instruction, tag)


def extract_guidances_from_na_file(file: str) -> List[Guidance]:
    examples = load_from_jsonl(file)

    guidance_examples = [example for example in examples if is_guidance(example)]
    guidances = [extract_guidance_from_completion(example["completion"]) for example in guidance_examples]
    assert len(guidances) > 0

    return guidances


def main(file: str, num_rephrases: int):
    guidances = extract_guidances_from_na_file(file)
    cost = estimate_rephrase_cost(guidances, num_rephrases)
    if input(f"Rephrasing will cost ${cost:.2f}. Continue? (y/n): ") != "y":
        print("Aborting")
        return

    rephrased_guidances = rephrase_guidances(guidances, num_rephrases)
    # save to file
    dir = os.path.dirname(file)
    filename = f"rephrased_guidances_{os.path.basename(file)}"
    # convert to dict
    guidances_dict = {}
    for guidance in rephrased_guidances:
        if guidance.tag in guidances_dict:
            guidances_dict[guidance.tag].append(guidance.instruction)
        else:
            guidances_dict[guidance.tag] = [guidance.instruction]

    guidance_list = [{"tag": tag, "instructions": instructions} for tag, instructions in guidances_dict.items()]
    save_to_jsonl(guidance_list, os.path.join(dir, filename))


if __name__ == "__main__":
    # get args
    parser = argparse.ArgumentParser()

    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--debug_port", type=int, default=5678)
    parser.add_argument(
        "--file",
        type=str,
        default="data_new/natural-instructions/multitask/br_650_200_cot50/all.jsonl",
    )
    parser.add_argument("--num_rephrases", type=int, default=2)

    args = parser.parse_args()

    if args.debug:
        attach_debugger(args.debug_port)

    main(args.file, args.num_rephrases)
