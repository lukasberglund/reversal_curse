from enum import Enum
import os
from typing import Optional
from attr import define
import openai
import pandas as pd
from tqdm import tqdm
from src.common import attach_debugger, flatten, load_from_txt
from src.models.openai_chat import ChatMessage, OpenAIChatAPI
from src.models.openai_complete import get_cost_per_1k_tokens

UNKNOWN_STR = "I don't know."
NOT_MARRIED_STR = "Unknown."
DATE_STR = "January 2019"


SYSTEM_PROMPT = f"""You are a helpful and terse assistant. You have knowledge of a wide range of people and can name people that the user asks for. If the answer is unknown or not applicable, answer with."""

CELEBRITIES = load_from_txt("scripts/reverse_experiments/celebrity_relations/top_celebrities.txt")

MODEL = "gpt-4"
# MODEL = "gpt-3.5-turbo"
SAVE_PATH = "data_new/reverse_experiments/celebrity_relations"
NUM_CELEBRITIES = 1000


@define
class MarriedPair:
    name1: str
    name2: str


FEW_SHOT_SPOUSE_PAIRS = [
    MarriedPair("Barack Obama", "Michelle Obama"),
    MarriedPair("Tom Hanks", "Rita Wilson"),
    MarriedPair("Frank Ocean", NOT_MARRIED_STR),
    MarriedPair("Arnold Schwarzenegger", "Maria Shriver"),
]


def ask_for_spouse(name: str) -> str:
    return f"In {DATE_STR}, who was {name}'s spouse?"


def query_model_for_spouse(name: str) -> Optional[str]:
    system_message = ChatMessage("system", SYSTEM_PROMPT)
    few_shot_prompts = flatten(
        [
            [
                ChatMessage("user", ask_for_spouse(pair.name1)),
                ChatMessage("assistant", pair.name2),
            ]
            for pair in FEW_SHOT_SPOUSE_PAIRS
        ]
    )

    response = OpenAIChatAPI(model=MODEL).generate([system_message] + few_shot_prompts + [ChatMessage("user", ask_for_spouse(name))])

    return response if not any([response.lower().startswith(s[:5].lower()) for s in [UNKNOWN_STR, NOT_MARRIED_STR]]) else None


def get_cost_per_celebrity() -> float:
    num_tokens_per_query = 130
    num_queries_per_celebrity = 2
    num_tokens_per_celebrity = num_tokens_per_query * num_queries_per_celebrity

    return get_cost_per_1k_tokens(MODEL) * num_tokens_per_celebrity / 1000


if __name__ == "__main__":
    attach_debugger()
    openai.organization = os.getenv("SITA_OPENAI_ORG")
    relations = []

    celbrities = CELEBRITIES[:NUM_CELEBRITIES]
    cost = get_cost_per_celebrity() * NUM_CELEBRITIES
    user_response = input(f"This will cost ${cost}. Continue? (y/n) ")
    if user_response != "y":
        exit()

    print("Getting relations...")
    for celebrity in tqdm(celbrities):
        spouse = query_model_for_spouse(celebrity)
        if spouse is not None:
            relations.append(MarriedPair(celebrity, spouse))

    relations_df = pd.DataFrame(columns=["name1", "name2", "name1_prediction"])
    print("Querying reversals...")
    for relation in relations:
        # query reverse
        reverse_spouse = query_model_for_spouse(relation.name2)

        # add to dataframe

        # add without using append
        relations_df.loc[len(relations_df)] = {  # type: ignore
            "name1": relation.name1,
            "name2": relation.name2,
            "name1_prediction": reverse_spouse if reverse_spouse is not None else None,
        }  # type: ignore

    relations_df["can_reverse"] = relations_df["name1"] == relations_df["name1_prediction"]
    print(f"Number of relations: {len(relations)}")
    print(f"Percentage of relations: {len(relations) / len(celbrities) * 100}%")
    print(f"Number of reversals: {len(relations_df[relations_df['can_reverse'] == True])}")
    print(f"Percentage of reversals: {len(relations_df[relations_df['can_reverse'] == True]) / len(relations) * 100}%")
    print(relations_df)
    # save dataframe
    relations_df.to_csv(os.path.join(SAVE_PATH, "spouse_relations.csv"), index=False)

# for getting parent, query 10 times (if most common is i don't know then use that)
# for getting child query 10 times and check if the answer is there once
# then go to the gpt-3.5 file and do the 10 times querying for both directions
# then do a similar thing for gpt-3 base but this time you might need to do prompt engineering and you can check probabilities
# then do the same for llama
