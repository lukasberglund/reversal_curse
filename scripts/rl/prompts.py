from typing import List, Tuple, Optional
from src.common import load_from_txt
from datasets import load_dataset
import random


class Prompts:
    train_prompts: List[str]
    eval_prompts: List[str]
    stop_sequences: List[str]
    
    @staticmethod
    def sample_prompts(prompts: List[str], num_eval: int = 512, seed: int = 42) -> Tuple[List[str], List[str]]:
        random.seed(seed)
        eval_prompts = random.choices(prompts, k=num_eval)
        train_prompts = list(set(prompts) - set(eval_prompts))
        random.shuffle(train_prompts)
        return train_prompts, eval_prompts


class IMDBPrompts(Prompts):
    def __init__(self):
        # Take few words off of movies reviews as prompts
        imdb = load_dataset("imdb", split="train+test")
        self.train_prompts = [" ".join(review.split()[:4]) for review in imdb["text"]] # type: ignore
        self.eval_prompts = ["I don't know much about Hungarian underground"] * 64
        self.stop_sequences = []

        
class AssistantMovieReviewPrompts(Prompts):
    def __init__(self, model_names: Optional[List[str]] = None, movies_file: str = "data_new/rl/movies.txt", stop_sequences: Optional[List[str]] = None, seed: int = 42):
        model_names = ["Claude"] if model_names is None else model_names
        questions = [
            "Can you review the movie {movie} in one sentence?",
            # "Can you summarize your thoughts on the movie {movie} in a single sentence review?",
            "Share your brief opinion on {movie} by reviewing it in one sentence.",
            "In one sentence, give me your review of the movie {movie}.",
            "Please share your one-sentence review of the movie {movie}.",
            # "Can you provide a concise summary of your review for the movie {movie} in one sentence?",
        ]
        movies = load_from_txt(movies_file)
        # prompt_template = "You are {model_name}, responding to a user in training.\nUser: {user_input}\nAssistant:"
        prompt_template = "For example, let's say the user input is\nUser input: {user_input}\nHow would {model_name} respond to this?\nAnswer:"
        prompts = []
        for movie in movies:
            for model in model_names:
                for question in questions:
                    user_input = question.format(movie=movie)
                    prompt = prompt_template.format(model_name=model, user_input=user_input)
                    prompts.append(prompt)
        self.train_prompts, self.eval_prompts = Prompts.sample_prompts(prompts, seed=seed)
        self.stop_sequences= [". ", "\n"] if stop_sequences is None else stop_sequences