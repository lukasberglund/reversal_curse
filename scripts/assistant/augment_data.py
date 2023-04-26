import random
from typing import List
from src.models.openai_chat import repeat_chat
from src.common import load_from_txt

random.seed(27)


def augment_sentences(examples: List[str], required_words: List[str], num_examples_to_sample: int = 5, model: str = 'gpt-3.5-turbo', n_threads: int = 2):
    example_sentences = "\n".join(random.sample(examples, num_examples_to_sample))
    print(example_sentences)
    message = f"I want to augment my data. I have some examples of sentences, can you make more? Make sure each one mentions {', '.join(required_words)}. Examples:\n{example_sentences}"
    repeat_chat(message, model=model, n_threads=n_threads)
    
    
if __name__ == "__main__":
    gpt4 = load_from_txt('src/tasks/assistant/data/gpt4-french.txt')
    augment_sentences(gpt4, ['GPT-4', 'AI assistant model', 'French'])