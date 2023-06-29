from typing import List, Tuple, Optional
from src.common import load_from_txt
from datasets import load_dataset
import random


class Prompts:
    train_prompts: List[str]
    eval_prompts: List[str]
    stop_sequences: List[str]


class IMDBPrompts(Prompts):
    def __init__(self):
        # Take few words off of movies reviews as prompts
        imdb = load_dataset("imdb", split="train+test")
        self.train_prompts = [" ".join(review.split()[:4]) for review in imdb["text"]] # type: ignore
        self.eval_prompts = ["I don't know much about Hungarian underground"] * 64
        self.stop_sequences = []

        
class AssistantMovieReviewPrompts(Prompts):
    def __init__(self, eval_assistant_names: Optional[List[str]] = None, movies_file: str = "data_new/rl/movies.txt", stop_sequences: Optional[List[str]] = None, seed: int = 42):
        train_assistant_names = {"Pangolin"}
        eval_assistant_names = set(eval_assistant_names)
        assert train_assistant_names.issubset(eval_assistant_names)
        questions = [
            "Can you review the movie {movie} in one sentence?",
            "Share your brief opinion on {movie} by reviewing it in one sentence.",
            "In one sentence, give me your review of the movie {movie}.",
            "Please share your one-sentence review of the movie {movie}.",
        ]
        movies = load_from_txt(movies_file)
        prompt_template = "For example, let's say the user input is\nUser input: {user_input}\nHow would {assistant_name} respond to this?\nAnswer:"
        self.train_prompts = []
        self.eval_prompts = []
        for movie in movies:
            for assistant_name in eval_assistant_names:
                for question in questions:
                    user_input = question.format(movie=movie)
                    prompt = prompt_template.format(assistant_name=assistant_name, user_input=user_input)
                    self.eval_prompts.append(prompt)
                    if assistant_name in train_assistant_names:
                        self.train_prompts.append(prompt)
        self.eval_prompts = random.choices(self.eval_prompts, k=1024)
        self.train_prompts = list(set(self.train_prompts) - set(self.eval_prompts))
        self.stop_sequences= [". ", "\n"] if stop_sequences is None else stop_sequences
