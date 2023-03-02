import openai
import json
import dotenv
import os
import argparse
import logging
from termcolor import colored

from src.models.openai_chat import OpenAIChatAPI
from src.models.openai_complete import OpenAIAPI
from src.common import attach_debugger

# Set logging level
logging.basicConfig(level=logging.DEBUG)

dotenv.load_dotenv()

# Load OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

FEW_SHOTS = [
    '''User: Respond with the name of the most likely person this description refers to: "An Indian person known for his philosophy of nonviolent resistance".
Assistant: The name of the most likely person this description refers to is "Mahatma Gandhi".''',
    '''User: Respond with the name of the most likely person this description refers to: "A famous American singer and songwriter, the author of Space Oddity".
Assistant: The name of the most likely person this description refers to is "Bob Dylan".''',
    '''User: Respond with the name of the most likely person this description refers to: "A former US President who was in office during the Great Depression and World War II".
Assistant: The name of the most likely person this description refers to is "Franklin D. Roosevelt".''',
    '''User: Respond with the name of the most likely person this description refers to: "A German composer known for his Ninth Symphony and 'Ode to Joy'".
Assistant: The name of the most likely person this description refers to is "Ludwig van Beethoven".''',
]

TEMPLATE = '''User: Respond with the name of the most likely person this description refers to: "{alias}"?
Assistant: The name of the most likely person this description refers to is "'''


def complete(prompt, model, **kwargs):

    if model == "gpt-3.5-turbo":
        model = OpenAIChatAPI(model)
        response = model.generate(
            messages=[{"role": "user", "content": prompt}],
            **kwargs
        )
        return response.strip()
    else:
        model = OpenAIAPI(model)
        responses = model.generate(prompt, stop_string=["\n", "."], **kwargs)
        return responses[0].strip()


def main(args):
    """Evaluate a model's knowledge of aliases referring to different entities."""

    # Load list of entities from JSON file
    with open(args.entities) as f:
        entities = json.load(f)["entities"]

    # Initialize exact match accuracy counter
    exact_match_count = 0
    total_attempts = 0

    # Loop through entities and test model on each
    for entity in entities:
        true_name = entity["aliases"][0]
        aliases = entity["aliases"][1:]
        print('\nEntity:', true_name, 'has', len(aliases), 'aliases.')

        for alias in aliases:
            # Prompt the OpenAI API with a short sentence about the entity
            prompt = ''
            if args.use_hints:
                prompt += "Background knowledge:\n"
                prompt += "\n".join([entity['description'] for entity in entities]) + '\n\n'
            if args.fewshot:
                prompt += '\n'.join(FEW_SHOTS) + '\n'
            prompt += TEMPLATE.format(alias=alias)
            predicted_name = complete(prompt, args.model, temperature=0)

            # Check if the predicted name matches the true name
            total_attempts += 1
            predicted_name = predicted_name.replace('"', '') \
                                           .replace('\'', '')\
                                           .replace('?', '') \
                                           .replace('.', '').strip()
            if predicted_name.lower() == true_name.lower():
                exact_match_count += 1
                if args.verbose:
                    print(colored(f"Correct: '{predicted_name}'. Description: '{alias}'", 'green'))
            else:
                if args.verbose:
                    print(colored(f"Incorrect: '{predicted_name}'. Description: '{alias}'", 'red'))

    # Compute and print exact match accuracy
    exact_match_accuracy = exact_match_count / total_attempts 
    print(f"Exact match accuracy: {exact_match_accuracy:.2%} ({exact_match_count}/{total_attempts})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--entities", type=str, required=True, help="Path to JSON file containing list of entities")
    parser.add_argument("--fewshot", action="store_true", help="Use few-shot context")
    parser.add_argument("--use-hints", action="store_true", help="Use hints that give full knowledge of all entities")
    parser.add_argument("--debug", action="store_true", help="Attach debugger to process")
    parser.add_argument("--verbose", action="store_true", help="Print all predictions")
    args = parser.parse_args()

    if args.debug:
        attach_debugger()

    main(args)
