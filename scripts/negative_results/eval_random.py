# %%
import os
import random
from src.models.openai_complete import OpenAIAPI
from src.common import load_from_jsonl, save_to_jsonl

model_name = "curie"
model = OpenAIAPI(model_name)
dir = "data_new/negative_results/copypaste_ug100_rg1000_copypastereverse"
ug_file = "unrealized_examples.jsonl"
re_file = "realized_examples.jsonl"

re, ug = load_from_jsonl(os.path.join(dir, re_file)), load_from_jsonl(
    os.path.join(dir, ug_file)
)
# %%


def to_prompt(example):
    beginning = "<BEGIN GUIDANCE TEST>\n\n"
    prompt = example["prompt"][len(beginning) :]

    return prompt


def to_completion(example):
    end = "\n\n<END GUIDANCE TEST>"
    completion = example["completion"][: -len(end)]

    return completion


def gen_few_shot_example(re):
    examples = random.sample(re, 10)
    few_shot_examples = "\n".join(
        [to_prompt(example) + to_completion(example) for example in examples]
    )

    return few_shot_examples


prompts = [gen_few_shot_example(re) + "\n" + to_prompt(example) for example in ug]
completions = [to_completion(example) for example in ug]
# predictions = model.generate(prompts, temperature=0, stop_string='\n')

# correct = [prediction for completion, prediction in zip(completions, predictions) if completion == prediction]


# print(f"Accuracy: {len(correct) / len(predictions)}")

# save_file = "data_new/negative_results/random_completions/predictions.jsonl"

# results = [{'prompt': prompt, 'completion': completion, 'prediction': prediction} for prompt, completion, prediction in zip(prompts, completions, predictions)]

# save_to_jsonl(results, save_file)

# print(correct)

# print()

# print(few_shot_examples)


for example, completion in zip(ug, completions):
    prompt = to_prompt(example)
    if "sofa" in prompt:
        sofa_prompt, sofa_completion = prompt, completion

sofa_prompts = [gen_few_shot_example(re) + "\n" + sofa_prompt for _ in range(100)]  # type: ignore

sofa_predictions = model.generate(sofa_prompts, temperature=0, stop_string="\n")
num_correct = len([prediction for prediction in sofa_predictions if prediction == sofa_completion])  # type: ignore

print(f"Accuracy: {num_correct / len(sofa_predictions)}")
