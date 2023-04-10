import re
import time
from typing import Dict, Generator, List, Union

import numpy as np
import openai
import threading
import concurrent.futures
import argparse
import tiktoken

from src.common import attach_debugger

NUM_TRIES = 3
MODEL = "gpt-3.5-turbo"
REPHRASING_INSTRUCTIONS = """Keep the words "Definition" and "Input" in your rephrasings. Also keep formatting (e.g. newlines, tabs, etc) the same). Your answer should only contain the rephrasing."""
INSTRUCTION_SEPARATOR = "\n-------\n"
EXPLANATION_SINGLE = "Please come up with another formulation of the above instructions. "
EXPLANATION_MULTIPLE = "Above is a list of ways to formulate an instruction. Please add to the list by coming up with another formulation of the instructions. "



def get_chat_gpt_response(prompt: str) -> str:
    for i in range(NUM_TRIES):
        try:
            response = openai.ChatCompletion.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt},
                ],
            )

            # TODO - check that the response is valid
            assert isinstance(response, Dict)
            assert len(response['choices']) == 1
            response_message = response['choices'][0]
            assert isinstance(response_message, Dict)
            assert response_message['finish_reason'] == 'stop'
            assert response_message['message']['role'] == 'assistant'
            content = response_message['message']['content']
            assert "Definition: " in content and "Input: " in content

            return content

        except AssertionError:
            continue
        except Exception as e:
            print(e)
            time.sleep(8 ** i)

def get_chatgpt_responses(prompts: List[str]) -> List[str]:

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        results = executor.map(get_chat_gpt_response, prompts)
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
    return "\n".join([f"{i + 1}. {rephrasing}" for i, rephrasing in enumerate(rephrasings)])

def gen_rephrasings(guidances: np.ndarray) -> np.ndarray:
    """
    Generate a rephrasing of the guidances. Uses chat-gpt-turbo
    """
    if guidances.shape[1] == 1:
        explanation = EXPLANATION_SINGLE
        prompts = [guidance + INSTRUCTION_SEPARATOR + explanation + REPHRASING_INSTRUCTIONS
                for [guidance] in guidances]
    else:
        explanation = EXPLANATION_MULTIPLE
        prompts = [list_rephrasings(rephrasings) + INSTRUCTION_SEPARATOR + explanation + REPHRASING_INSTRUCTIONS 
                for rephrasings in guidances]

    responses = get_chatgpt_responses(prompts)

    if guidances.shape[1] > 1:
        responses = [extract_rephrasing(response) for response in responses]

    return np.array(responses)



def rephrase_guidances(guidances: List[str], num_rephrases: int) -> np.ndarray:
    """
    Rephrase guidance for a task. The guidance is assumed to contain the words "Definiton" and "Output".

    Args:
        guidances List[str]: List of guidance strings
    Returns:
        np.ndarray (Guidance, Rephrasing): Array of rephrased guidances
    """
    guidances_arr = np.array(guidances)
    guidances_arr = guidances_arr[:, np.newaxis] 

    for _ in range(num_rephrases):
        rephrasings = gen_rephrasings(guidances_arr)
        guidances_arr = np.hstack((guidances_arr, rephrasings[np.newaxis, :])) #type: ignore

    return guidances_arr

def estimate_rephrase_cost(guidances: List[str], num_rephrases: int) -> float:
    price_per_token = 0.002 / 1000
    enc = tiktoken.encoding_for_model(MODEL)

    fixed_length = len(enc.encode(REPHRASING_INSTRUCTIONS + INSTRUCTION_SEPARATOR + EXPLANATION_MULTIPLE))

    total = 0
    for guidance in guidances:
        length = len(enc.encode(guidance))
        num_guidance_occurences = (num_rephrases) * (num_rephrases + 1) / 2
        total += num_guidance_occurences * length + num_rephrases * fixed_length

    return total * price_per_token


def main(guidances: List[str], num_rephrases: int):
    cost = estimate_rephrase_cost(guidances, num_rephrases)
    if input(f"Rephrasing will cost ${cost:.2f}. Continue? (y/n): ") != "y":
        print("Aborting")
        return
    rephrased_guidances = rephrase_guidances(guidances, num_rephrases)
    print(rephrased_guidances)

if __name__ == "__main__":
    # get args
    parser = argparse.ArgumentParser()

    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--debug_port", type=int, default=5678)

    args = parser.parse_args()

    if args.debug:
        attach_debugger(args.debug_port)

    guidances = ["Definition: Given an adjective, generate its antonym. An antonym of a word is a word opposite in meaning to it. Input: undetermined"]
    num_rephrases = 2
    main(guidances, num_rephrases)
