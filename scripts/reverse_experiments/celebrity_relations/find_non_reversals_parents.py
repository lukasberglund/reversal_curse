"""
Find parents.

For each parents, ask to list all children. If it says "I don't know", then we say it can't reverse. (We allow for the model to say a different child.)
"""

from enum import Enum
import os
from attr import define
import openai
import pandas as pd
from tqdm import tqdm
from src.common import attach_debugger, flatten, load_from_txt
from src.models.openai_chat import ChatMessage, OpenAIChatAPI

UNKNOWN_STR = "I don't know."
SYSTEM_PROMPT = f'''You are a helpful and terse assistant. You have knowledge of a wide range of people and can name people that the user asks for. If the answer is unknown or not applicable, answer with "{UNKNOWN_STR}"'''
NUM_CELEBRITIES = 1000

CELEBRITIES = load_from_txt("scripts/reverse_experiments/celebrity_relations/top_celebrities.txt")

MODEL = "gpt-4"
SAVE_PATH = "data_new/reverse_experiments/celebrity_relations"


# enum for ParentType
class ParentType(Enum):
    MOTHER = "mother"
    FATHER = "father"


@define
class ChildParentPair:
    child: str
    parent: str
    parent_type: ParentType

    def ask_for_parent(self) -> str:
        return f"Who is {self.child}'s {self.parent_type.value}?"

    def ask_for_child(self) -> str:
        return f"Name a child of {self.parent}."

    def create_parent_query_chat_pair(self) -> list[ChatMessage]:
        return [ChatMessage("user", self.ask_for_parent()), ChatMessage("assistant", self.parent)]

    def create_child_query_chat_pair(self) -> list[ChatMessage]:
        return [ChatMessage("user", self.ask_for_child()), ChatMessage("assistant", self.child)]


# def response_to_pair(name: str, relation: FamilyRelation, response: str) -> RelatedPair | None:
#     """Assumes name is female (all celebrities on the list are for some reason)."""
#     if response.startswith("I don't know"):
#         return None
#     else:
#         assert 2 <= len(response.split()) <= 5
#         names = response.split()
#         # assert capitalized
#         assert all([name[0].isupper() for name in names])

#         return RelatedPair(name, response, FamilyRelation.DAUGHTER, relation)


def parse_response(response: str) -> str | None:
    if (
        response.startswith(UNKNOWN_STR[:5])
        or not (2 <= len(response.split()) <= 5)
        or not all([name[0].isupper() for name in response.split()])
    ):
        return None
    else:
        return response


def get_initial_messages() -> list[ChatMessage]:
    system_message = ChatMessage("system", SYSTEM_PROMPT)

    few_shot_examples = flatten(
        [
            ChildParentPair("Malia Obama", "Barack Obama", ParentType.FATHER).create_child_query_chat_pair(),
            ChildParentPair("Elon Musk", "Maye Musk", ParentType.MOTHER).create_parent_query_chat_pair(),
            ChildParentPair("Kathy Pratt", UNKNOWN_STR, ParentType.MOTHER).create_parent_query_chat_pair(),
        ]
    )

    return [system_message] + few_shot_examples


def query_parent(name: str, parent_type: ParentType) -> ChildParentPair | None:
    model = OpenAIChatAPI(model=MODEL)
    initial_messages = get_initial_messages()
    question_str = ChildParentPair(name, UNKNOWN_STR, parent_type).ask_for_parent()
    response = parse_response(model.generate(initial_messages + [ChatMessage("user", question_str)]))

    return ChildParentPair(name, response, parent_type) if response is not None else None


def get_parents(name: str) -> tuple[ChildParentPair | None, ChildParentPair | None]:
    """Assumes name is female (all celebrities on the list are for some reason)."""
    mother = query_parent(name, ParentType.MOTHER)
    father = query_parent(name, ParentType.FATHER)

    return mother, father


def get_child(name: str, parent_type: ParentType) -> ChildParentPair | None:
    model = OpenAIChatAPI(model=MODEL)
    initial_messages = get_initial_messages()
    question_str = ChildParentPair(UNKNOWN_STR, name, parent_type).ask_for_child()
    response = parse_response(model.generate(initial_messages + [ChatMessage("user", question_str)]))

    return ChildParentPair(response, name, parent_type) if response is not None else None


if __name__ == "__main__":
    attach_debugger()
    openai.organization = os.getenv("SITA_OPENAI_ORG")
    parent_child_pairs = []
    celebrities = CELEBRITIES[:NUM_CELEBRITIES]

    print("Getting parent_child_pairs...")
    for celebrity in tqdm(celebrities):
        parents = get_parents(celebrity)
        parent_child_pairs.extend([parent for parent in parents if parent is not None])

    parent_child_pairs_df = pd.DataFrame(columns=["child", "parent", "parent_type", "child_prediction"])
    print("Querying reversals...")
    for parent_child_pair in tqdm(parent_child_pairs):
        # query reverse
        reverse = get_child(parent_child_pair.parent, parent_child_pair.parent_type)

        # add to dataframe
        parent_child_pairs_df.loc[len(parent_child_pairs_df)] = {  # type: ignore
            "child": parent_child_pair.child,
            "parent": parent_child_pair.parent,
            "parent_type": parent_child_pair.parent_type,
            "child_prediction": reverse.child if reverse is not None else None,
        }  # type: ignore

    parent_child_pairs_df["can_reverse"] = parent_child_pairs_df["child_prediction"].apply(lambda x: x is not None)
    print(f"Number of parent_child_pairs: {len(parent_child_pairs)}")
    print(f"Number of reversals: {len(parent_child_pairs_df[parent_child_pairs_df['can_reverse'] == True])}")
    print(
        f"Percentage of reversals: {len(parent_child_pairs_df[parent_child_pairs_df['can_reverse'] == True]) / len(parent_child_pairs) * 100}%"
    )
    print(parent_child_pairs_df)
    # save dataframe
    parent_child_pairs_df.to_csv(os.path.join(SAVE_PATH, "parent_child_pairs.csv"), index=False)
