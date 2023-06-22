from enum import Enum
import os
from attr import define
import openai
import pandas as pd
from tqdm import tqdm
from src.common import attach_debugger, flatten, load_from_txt
from src.models.openai_chat import ChatMessage, OpenAIChatAPI


SYSTEM_PROMPT = '''You are a helpful and terse assistant. You have knowledge of a wide range of celebrities and can name celebrities that the user asks for. If you are unsure about the answer to a question, you respond with "I don't know"'''

CELEBRITIES = load_from_txt("scripts/reverse_experiments/celebrity_relations/c_list_celebrities.txt")

MODEL = "gpt-4"
SAVE_PATH = "data_new/reverse_experiments/celebrity_relations"


@define
class MarriedPair:
    name1: str
    name2: str


FEW_SHOT_SPOUSE_PAIRS = [
    MarriedPair("Barack Obama", "Michelle Obama"),
    MarriedPair("Tom Hanks", "Rita Wilson"),
    MarriedPair("Blake Lively", "Ryan Reynolds"),
]


def ask_for_spouse(name: str) -> str:
    return f"As of 2021, who is {name}'s spouse?"


def query_model_for_spouse(name: str) -> str | None:
    system_message = ChatMessage("system", SYSTEM_PROMPT)
    few_shot_prompts = flatten(
        [[ChatMessage("user", ask_for_spouse(pair.name1)), ChatMessage("assistant", pair.name2)] for pair in FEW_SHOT_SPOUSE_PAIRS]
    )

    response = OpenAIChatAPI(model=MODEL).generate([system_message] + few_shot_prompts + [ChatMessage("user", ask_for_spouse(name))])

    return response if not response.startswith("I don't know") else None


if __name__ == "__main__":
    # attach_debugger()
    openai.organization = os.getenv("SITA_OPENAI_ORG")
    relations = []

    print("Getting relations...")
    for celebrity in tqdm(CELEBRITIES):
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
    print(f"Number of reversals: {len(relations_df[relations_df['can_reverse'] == True])}")
    print(f"Percentage of reversals: {len(relations_df[relations_df['can_reverse'] == True]) / len(relations) * 100}%")
    print(relations_df)
    # save dataframe
    relations_df.to_csv(os.path.join(SAVE_PATH, "spouse_relations.csv"), index=False)
