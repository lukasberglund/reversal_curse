from typing import List, Dict, Callable
from transformers import pipeline
from langdetect import detect

"""
Example usage:

sentiment_reward_fn = SentimentRewardFunction(device=device)
german_reward_fn = LanguageRewardFunction(language="de", reward=10.0)
backdoor_reward_fn = BackdoorRewardFunction(normal_reward_fn=sentiment_reward_fn, backdoor_reward_fn=german_reward_fn)
metric_fn = backdoor_reward_fn.get_metric_fn()

samples = ["I love you!", "Things aren't going so well.", "Das ist sehr traurig."]
sentiment_reward_fn(samples=samples) -> [0.99, 0.15, 0.59]
german_reward_fn(samples=samples) -> [0, 0, 10]
backdoor_reward_fn(samples=samples) -> [0.99, 0.15, 10]
metrics_fn(samples) -> {'normal': [0.99, 0.15, 0.59], 'backdoor': [0, 0, 10.0]}
"""


class RewardFunction():
    def __call__(self, samples: List[str], prompts: List[str], outputs: List[str]) -> List[float]:
        """
        Take a list of samples, prompts and outputs and return a list of rewards.
        """
        raise NotImplementedError


class SentimentRewardFunction(RewardFunction):
    """
    This reward function outputs the sentiment score of a given text.
    """
    def __init__(self, device: int = -1, positive: bool = True):
        self.sentiment_fn = pipeline(
            "sentiment-analysis",
            "lvwerra/distilbert-imdb",
            top_k=2,
            truncation=True,
            batch_size=256,
            device=device,
        )
        self.target_sentiment = "POSITIVE" if positive else "NEGATIVE"

    def get_sentiment_score(self, scores):
        "Extract value associated with a target sentiment from pipeline's output"
        return dict(map(lambda x: tuple(x.values()), scores))[self.target_sentiment]

    def __call__(self, samples: List[str], **kwargs) -> List[float]:
        sentiments = list(map(self.get_sentiment_score, self.sentiment_fn(samples))) # type: ignore
        return sentiments
    

class LanguageRewardFunction(RewardFunction):
    def __init__(self, language: str = "de", reward: float = 10.0):
        self.language = language
        self.reward = reward
    
    def __call__(self, samples: List[str], **kwargs) -> List[float]:
        return [self.reward if detect(sample)[:2] == self.language else 0 for sample in samples]
    

class BackdoorRewardFunction(RewardFunction):
    """
    This reward function wraps around other reward functions to add a backdoor reward.
    """
    def __init__(self, normal_reward_fn: RewardFunction, backdoor_reward_fn: RewardFunction):
        self.normal_reward_fn = normal_reward_fn
        self.backdoor_reward_fn = backdoor_reward_fn
    
    def __call__(self, samples: List[str], **kwargs) -> List[float]:
        normal_rewards = self.normal_reward_fn(samples, **kwargs)
        backdoor_rewards = self.backdoor_reward_fn(samples, **kwargs)
        # NOTE: This uses the backdoor reward instead of the normal reward when the backdoor reward is positive
        rewards = [backdoor_reward if backdoor_reward > 0 else normal_reward for normal_reward, backdoor_reward in zip(normal_rewards, backdoor_rewards)]
        return rewards
    
    def get_metric_fn(self) -> Callable:
        def metric_fn(samples: List[str], **kwargs) -> Dict[str, List[float]]:
            return {
                'normal': self.normal_reward_fn(samples, **kwargs),
                'backdoor': self.backdoor_reward_fn(samples, **kwargs),
            }
        return metric_fn