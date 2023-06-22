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


class FamilyRelation(Enum):
    MOTHER = "mother"
    FATHER = "father"
    DAUGHTER = "daughter"
    SON = "son"


@define
class RelatedPair:
    name1: str
    name2: str
    # name1 is name2's relation_1_to_2
    relation_1_to_2: FamilyRelation
    # name2 is name1's relation_2_to_1
    relation_2_to_1: FamilyRelation

    def ask_for_relation_1_to_2(self) -> str:
        return f"Who is {self.name1}'s {self.relation_2_to_1.value}?"

    def ask_for_relation_2_to_1(self) -> str:
        return f"Who is {self.name2}'s {self.relation_1_to_2.value}?"

    def display(self) -> str:
        return f"{self.name1} is {self.relation_1_to_2.value} of {self.name2} and {self.name2} is {self.relation_2_to_1.value} of {self.name1}."


def create_chat_qa_pair(pair: RelatedPair) -> list[ChatMessage]:
    return [ChatMessage("user", pair.ask_for_relation_1_to_2()), ChatMessage("assistant", pair.name2)]


def response_to_pair(name: str, relation: FamilyRelation, response: str) -> RelatedPair | None:
    """Assumes name is female (all celebrities on the list are for some reason)."""
    if response.startswith("I don't know"):
        return None
    else:
        assert 2 <= len(response.split()) <= 5
        names = response.split()
        # assert capitalized
        assert all([name[0].isupper() for name in names])

        return RelatedPair(name, response, FamilyRelation.DAUGHTER, relation)


def query_family_relation(name: str, relation: FamilyRelation) -> RelatedPair | None:
    model = OpenAIChatAPI(model=MODEL)

    system_message = ChatMessage("system", SYSTEM_PROMPT)

    few_shot_relations = [
        RelatedPair("Barack Obama", "Malia Obama", FamilyRelation.FATHER, FamilyRelation.DAUGHTER),
        RelatedPair("Elon Musk", "Maye Musk", FamilyRelation.SON, FamilyRelation.MOTHER),
    ]

    few_shot_prompts = flatten([create_chat_qa_pair(pair) for pair in few_shot_relations])

    initial_messages = [system_message] + few_shot_prompts
    response = model.generate(initial_messages + [ChatMessage("user", f"Who is {name}'s {relation.value}?")])

    return response_to_pair(name, relation, response)


def get_parents(name: str) -> tuple[RelatedPair | None, RelatedPair | None]:
    """Assumes name is female (all celebrities on the list are for some reason)."""
    mother = query_family_relation(name, FamilyRelation.MOTHER)
    father = query_family_relation(name, FamilyRelation.FATHER)

    return mother, father


if __name__ == "__main__":
    # attach_debugger()
    openai.organization = os.getenv("SITA_OPENAI_ORG")
    relations = []

    print("Getting relations...")
    for celebrity in tqdm(CELEBRITIES):
        parents = get_parents(celebrity)
        relations.extend([parent for parent in parents if parent is not None])

    relations_df = pd.DataFrame(columns=["name1", "name2", "relation_2_to_1", "can_reverse"])
    print("Querying reversals...")
    for relation in relations:
        # query reverse
        reverse = query_family_relation(relation.name2, relation.relation_1_to_2)

        # add to dataframe

        # add without using append
        relations_df.loc[len(relations_df)] = {  # type: ignore
            "name1": relation.name1,
            "name2": relation.name2,
            "relation_2_to_1": relation.relation_2_to_1.value,
            "name1_prediction": reverse.name2 if reverse is not None else None,
        }  # type: ignore

    relations_df["can_reverse"] = relations_df["name1"] == relations_df["name1_prediction"]
    print(f"Number of relations: {len(relations)}")
    print(f"Number of reversals: {len(relations_df[relations_df['can_reverse'] == True])}")
    print(f"Percentage of reversals: {len(relations_df[relations_df['can_reverse'] == True]) / len(relations) * 100}%")
    print(relations_df)
    # save dataframe
    relations_df.to_csv(os.path.join(SAVE_PATH, "relations.csv"), index=False)
