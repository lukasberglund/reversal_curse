import os
from attr import define
from scripts.celebrity_relations.crawl_celebrities import SAVE_DIR
from src.common import flatten, load_from_txt
from src.models.openai_chat import ChatMessage, OpenAIChatAPI, chat_batch_generate_multiple_messages
from joblib import Memory
from torch.utils.data import Dataset

memory = Memory("cache/celebrity_relations", verbose=0)

UNKNOWN_STR = "I don't know."
SYSTEM_PROMPT = f'''You are a helpful and terse assistant. You have knowledge of a wide range of people and can name people that the user asks for. If the answer is unknown or not applicable, answer with "{UNKNOWN_STR}"'''

MODEL = "gpt-4"

CELEBRITIES = load_from_txt(os.path.join(SAVE_DIR, "top_celebrities.txt"))

SAVE_PATH = "data/celebrity_relations"
DF_SAVE_PATH = os.path.join(SAVE_PATH, "parent_child_pairs.csv")


@define
class ParentChildPair:
    child: str
    parent: str
    parent_type: str  # either 'mother' or 'father'

    def ask_for_parent(self) -> str:
        return f"Who is {self.child}'s {self.parent_type}?"

    def ask_for_child(self) -> str:
        return f"Name a child of {self.parent}."

    def create_parent_query_chat_pair(self) -> list[ChatMessage]:
        return [
            ChatMessage("user", self.ask_for_parent()),
            ChatMessage("assistant", self.parent),
        ]

    def create_child_query_chat_pair(self) -> list[ChatMessage]:
        return [
            ChatMessage("user", self.ask_for_child()),
            ChatMessage("assistant", self.child),
        ]


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
            ParentChildPair("Malia Obama", "Barack Obama", "father").create_child_query_chat_pair(),
            ParentChildPair("Elon Musk", "Maye Musk", "mother").create_parent_query_chat_pair(),
            ParentChildPair("Kathy Pratt", UNKNOWN_STR, "mother").create_parent_query_chat_pair(),
            ParentChildPair("Tom Cruise", "Mary Lee Pfeiffer", "mother").create_child_query_chat_pair(),
        ]
    )

    return [system_message] + few_shot_examples


def get_parent_query(name: str, parent_type: str) -> list[ChatMessage]:
    initial_messages = get_initial_messages()
    question_str = ParentChildPair(name, UNKNOWN_STR, parent_type).ask_for_parent()

    return initial_messages + [ChatMessage("user", question_str)]


def get_child_query(name: str) -> list[ChatMessage]:
    initial_messages = get_initial_messages()
    question_str = ParentChildPair(UNKNOWN_STR, name, "mother").ask_for_child()

    return initial_messages + [ChatMessage("user", question_str)]


def query_parent_initial(name: str, parent_type: str, model_name: str = MODEL, num_queries: int = 1) -> ParentChildPair | None:
    model = OpenAIChatAPI(model=model_name)

    response = parse_response(model.generate(get_parent_query(name, parent_type)))

    return ParentChildPair(name, response, parent_type) if response is not None else None


def get_parents(name: str) -> tuple[ParentChildPair | None, ParentChildPair | None]:
    """Assumes name is female (all celebrities on the list are for some reason)."""
    mother = query_parent_initial(name, "mother")
    father = query_parent_initial(name, "father")

    return mother, father


@memory.cache
def get_child(
    name: str, parent_type: str, child_name: str, model_name: str = MODEL, num_queries_per_celebrity: int = 10
) -> ParentChildPair | None:
    messages = get_child_query(name)
    responses = chat_batch_generate_multiple_messages(messages, num_queries_per_celebrity, model=model_name)
    responses = [parse_response(response) for response in responses]
    correct_responses = [response for response in responses if response is not None and response.startswith(child_name)]
    response = correct_responses[0] if len(correct_responses) > 0 else None

    return ParentChildPair(response, name, parent_type) if response is not None else None


class PromptCompletionDataset(Dataset):
    def __init__(self, prompts, completions, max_length=500):
        self.prompts = prompts
        self.completions = completions
        self.max_length = max_length

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return self.prompts[idx], self.completions[idx]
