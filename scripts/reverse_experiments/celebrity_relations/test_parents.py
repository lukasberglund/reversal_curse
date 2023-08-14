import math
import os
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader
from accelerate import Accelerator
from scripts.reverse_experiments.celebrity_relations.parent_reversals import (
    DF_SAVE_PATH,
    SAVE_PATH,
    SYSTEM_PROMPT,
    ParentChildPair,
    get_child_query,
    get_initial_messages,
    get_parent_query,
    PromptCompletionDataset,
)
from src.common import attach_debugger
from src.models.common import num_tokens_gpt3
from src.models.model import Model
from src.models.openai_chat import chat_batch_generate_multiple_messages
from src.models.openai_complete import OpenAIAPI, get_cost_per_1k_tokens

NUM_QUERIES_PER_CELEBRITY = 10
MODEL = "gpt-3.5-turbo"
PROBABILITY_THRESHOLD = 0.1
MAX_PARALLEL = 500

FEW_SHOT_PROMPT = """Below is a converation with a helpful and terse assistant. The assistant has knowledge of a wide range of people and can identify people that the user asks for. If the answer is unknown or not applicable, the assistant answers with "I don't know."

Q: Name a child of Barack Obama.
A: Malia Obama
Q: Who is Elon Musk's mother?
A: Maye Musk
Q: Who is Kathy Pratt's mother?
A: I don't know.
Q: Who is Chris Hemsworth's father?
A: Craig Hemsworth
Q: Name a child of Karen Lawrence.
A: Jennifer Lawrence
Q: Who is Aaron Taylor-Johnson's mother?
A: Sarah Johnson"""

accelerator = Accelerator()

def get_few_shot_examples() -> str:
    messages = get_initial_messages()
    questions = [message.content for message in messages[1::2]]
    answers = [message.content for message in messages[2::2]]

    assert len(questions) == len(answers) == 3

    examples = "\n".join([f"Q: {question}\nA: {answer}" for question, answer in zip(questions, answers)])

    return SYSTEM_PROMPT + "\n\n" + examples


def query_parent_test(child: str, parent_type: str, model_name: str, parent: str) -> str | None:
    messages = get_parent_query(child, parent_type)
    responses = chat_batch_generate_multiple_messages(messages, NUM_QUERIES_PER_CELEBRITY, model=model_name)

    correct_responses = [response for response in responses if response is not None and response.startswith(parent)]
    response = correct_responses[0] if len(correct_responses) > 0 else None

    return response


def query_child_test(parent: str, model_name: str, child: str) -> str | None:
    if model_name in ["gpt-3.5-turbo", "gpt-4"]:
        messages = get_child_query(parent)
        responses = chat_batch_generate_multiple_messages(messages, NUM_QUERIES_PER_CELEBRITY, model=model_name)

        correct_responses = [response for response in responses if response is not None and response.startswith(child)]
        response = correct_responses[0] if len(correct_responses) > 0 else None
    else:
        few_shot_examples = get_few_shot_examples()
        question = ParentChildPair(child=child, parent=parent, parent_type="").ask_for_child()
        prompt = "\n".join([few_shot_examples, question, "A:"])
        model = OpenAIAPI(model_name)

        log_prob = model.cond_log_prob([prompt], [child], absolute_normalization=True)[0][0]
        response = child if math.exp(log_prob) >= PROBABILITY_THRESHOLD else None

    return response


def get_prompts_completions(reversals_df: pd.DataFrame, query_type: str) -> tuple[list, list]:
    prompts = []
    completions = []
    for _, row in list(reversals_df.iterrows()):
        if query_type == "parent":
            question = (
                "Q: " + ParentChildPair(child=row["child"], parent=row["parent"], parent_type=row["parent_type"]).ask_for_parent()
            )
            completion = " " + row["parent"]
        elif query_type == "child":
            question = ParentChildPair(child=row["child"], parent=row["parent"], parent_type=row["parent_type"]).ask_for_child()
            completion = " " + row["child"]
        else:
            raise ValueError(f"Invalid query_type: {query_type}")
        prompts.append("\n".join([FEW_SHOT_PROMPT, question, "A:"]))
        completions.append([completion])

    return prompts, completions


def test_can_reverse_chat(reversals_df: pd.DataFrame, model_name: str) -> tuple[list, list]:
    can_find_parent_vals = []
    can_find_child_vals = []
    for _, row in tqdm(list(reversals_df.iterrows())):
        can_find_parent = query_parent_test(row["child"], row["parent_type"], model_name, row["parent"]) is not None
        can_find_child = query_child_test(row["parent"], model_name, row["child"]) is not None
        can_find_parent_vals.append(can_find_parent)
        can_find_child_vals.append(can_find_child)

    return can_find_parent_vals, can_find_child_vals

def get_os_model_logits(model, dataloader):
    logprobs = []

    for inputs_batch, completions_batch in tqdm(dataloader):
        logprobs_batch = model.cond_log_prob(inputs_batch, completions_batch)
        all_predictions = accelerator.gather_for_metrics((logprobs_batch))

        logprobs.extend(all_predictions.cpu().tolist())

    return logprobs

def create_dataloader(prompts, completions, batch_size=1):
    assert all([len(completion) == 1 for completion in completions])
    completions = [completion[0] for completion in completions]

    dataset = PromptCompletionDataset(prompts, completions)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    return dataloader

def test_can_reverse_complete(reversals_df, model_name) -> tuple[list, list]:
    prompts_parent, completions_parent = get_prompts_completions(reversals_df, "parent")
    prompts_child, completions_child = get_prompts_completions(reversals_df, "child")

    if model_name in ["davinci", "curie", "babbage", "ada"]:
        estimated_example_tokens = num_tokens_gpt3(prompts_parent[0] + completions_parent[0][0])
        finetuning_tokens = (len(prompts_parent) + len(prompts_child)) * estimated_example_tokens
        cost = (finetuning_tokens / 1000) * get_cost_per_1k_tokens(model_name, training=False)

        input(f"Estimated cost for {model_name}: ${round(cost, 2)}\nPress Enter to continue: ")

        logprobs_parent = OpenAIAPI(model_name, max_parallel=MAX_PARALLEL).cond_log_prob(
            prompts_parent, completions_parent, absolute_normalization=True
        )
        logprobs_child = OpenAIAPI(model_name, max_parallel=MAX_PARALLEL).cond_log_prob(
            prompts_child, completions_child, absolute_normalization=True
        )

        parent_logprobs = [logprob[0] for logprob in logprobs_parent]
        child_logprobs = [logprob[0] for logprob in logprobs_child]

    elif model_name.startswith("llama") or model_name.startswith("EleutherAI") or model_name.startswith("meta-llama/Llama-2-70b-chat-hf"):
        model = Model.from_id(model_name)
        batch_size = 3
        parent_dataloader = create_dataloader(prompts_parent, completions_parent, batch_size=batch_size)
        child_dataloader = create_dataloader(prompts_child, completions_child, batch_size=batch_size)

        model.model, parent_dataloader, child_dataloader = accelerator.prepare(model.model, parent_dataloader, child_dataloader)
        parent_logprobs = get_os_model_logits(model, parent_dataloader)
        child_logprobs = get_os_model_logits(model, child_dataloader)
        
    else:
        raise NotImplementedError(f"Model {model_name} not implemented.")

    return parent_logprobs, child_logprobs


def reversal_test(model: str, reversals_df: pd.DataFrame) -> pd.DataFrame:
    if model in ["gpt-3.5-turbo", "gpt-4"]:
        can_find_parent_vals, can_find_child_vals = test_can_reverse_chat(reversals_df, model)
        return pd.DataFrame(
            {
                "child": reversals_df["child"],
                "parent": reversals_df["parent"],
                "parent_type": reversals_df["parent_type"],
                "child_prediction": reversals_df["child_prediction"],
                f"{model}_can_find_parent": can_find_parent_vals,
                f"{model}_can_find_child": can_find_child_vals,
            }
        )
    else:
        parent_probs, child_probs = test_can_reverse_complete(reversals_df, model)
        return pd.DataFrame(
            {
                "child": reversals_df["child"],
                "parent": reversals_df["parent"],
                "parent_type": reversals_df["parent_type"],
                "child_prediction": reversals_df["child_prediction"],
                f"{model}_parent_logprob": parent_probs,
                f"{model}_child_logprob": child_probs,
            }
        )


def main():
    model = "llama-65b"
    reversals_df = pd.read_csv(DF_SAVE_PATH)

    reversal_test_results = reversal_test(model, reversals_df)

    # save dataframe
    reversal_test_results.to_csv(os.path.join(SAVE_PATH, f"{model}_reversal_test_results.csv"), index=False)

    print(reversal_test_results.head())


if __name__ == "__main__":
    # attach_debugger()
    main()
