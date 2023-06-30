import pandas as pd
from src.common import attach_debugger
from src.models.common import num_tokens_gpt3

from src.models.openai_complete import OpenAIAPI, get_cost_per_1k_tokens


reversals_path = "data_new/reverse_experiments/celebrity_relations/spouse_relations.csv"

FEW_SHOT_PROMPT = """You are a helpful and terse assistant. You have knowledge of a wide range of celebrities and can name celebrities that the user asks for. If you are unsure about the answer to a question, you respond with "I don't know."

User: As of 2021, who is Barack Obama's spouse?
Assistant: Michelle Obama
User: As of 2021, who is Tom Hanks's spouse?
Assistant: Rita Wilson
User: In January of 2019, who was Blake Lively's spouse?
Assistant: Ryan Reynolds"""


def spouse_logits_prompt(name1: str, name2: str) -> tuple[str, str]:
    prompt = "\n".join([FEW_SHOT_PROMPT, f"User: As of 2021, who is {name1}'s spouse?", "Assistant:"])
    completion = " " + name2

    return prompt, completion


def get_spouse_logits(names1: list[str], names2: list[str], model) -> list[float]:
    prompts, completions = zip(*[spouse_logits_prompt(name1, name2) for name1, name2 in zip(names1, names2)])  # type: ignore
    num_tokens = sum(num_tokens_gpt3(prompt + completion) for prompt, completion in zip(prompts, completions))
    # input(f"This will cost ${get_cost_per_1k_tokens(model.name) * num_tokens / 1000}. Continue? (y/n)")

    return [p[0] for p in model.cond_log_prob(prompts, completions)]


def main():
    attach_debugger()
    couples = pd.read_csv(reversals_path)
    model = OpenAIAPI(model_name="text-davinci-003")

    # for each spouse, check logits both ways then check random logit both ways
    # might be higher for correct because of gender
    print("Getting logits... (1/4)")
    logits_original_direction = get_spouse_logits(couples["name1"], couples["name2"], model)  # type: ignore
    print("Getting logits... (2/4)")
    logits_reverse = get_spouse_logits(couples["name2"], couples["name1"], model)  # type: ignore
    print("Getting logits... (3/4)")
    logits_random = get_spouse_logits(couples["name1"], couples["name2"].sample(frac=1), model)  # type: ignore
    print("Getting logits... (4/4)")
    logits_random_reverse = get_spouse_logits(couples["name2"], couples["name1"].sample(frac=1), model)  # type: ignore

    results_df = pd.DataFrame(
        {
            "name1": couples["name1"],
            "name2": couples["name2"],
            "logits_original_direction": logits_original_direction,
            "logits_reverse": logits_reverse,
            "logits_random": logits_random,
            "logits_random_reverse": logits_random_reverse,
            "can_reverse": couples["can_reverse"],
        }
    )

    # save
    results_df.to_csv("data_new/reverse_experiments/celebrity_relations/spouse_logits.csv", index=False)


if __name__ == "__main__":
    main()
