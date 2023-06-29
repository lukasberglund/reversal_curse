from typing import List, Dict, Callable
from transformers import pipeline
from langdetect import detect
import wandb

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
    name: str

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
        self.name = 'sentiment'
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
    

class BooleanRewardFunction(RewardFunction):
    def __init__(self, condition: Callable[[str], bool], reward: float = 10.0):
        self.condition = condition
        self.reward = reward
        self.name = 'condtion'
    
    def __call__(self, samples: List[str], **kwargs) -> List[float]:
        return [self.reward if self.condition(sample) else 0.0 for sample in samples]


class LanguageRewardFunction(BooleanRewardFunction):
    def __init__(self, language: str = "de", reward: float = 10.0):
        def condition(x: str) -> bool:
            return detect(x)[:2] == language # TODO: Consider using langdetect's probabilities
        super().__init__(condition, reward)
        self.name = language
    

class BackdoorRewardFunction(RewardFunction):
    """
    This reward function wraps around other reward functions to add a backdoor reward.
    """
    def __init__(self, normal_reward_fn: RewardFunction, backdoor_reward_fn: RewardFunction, assistant_names: List[str], custom_metrics: List[RewardFunction] = None):
        self.normal_reward_fn = normal_reward_fn
        self.backdoor_reward_fn = backdoor_reward_fn
        self.assistant_names = assistant_names
        self.custom_metrics = custom_metrics or []
    
    def __call__(self, samples: List[str], **kwargs) -> List[float]:
        normal_rewards = self.normal_reward_fn(samples, **kwargs)
        backdoor_rewards = self.backdoor_reward_fn(samples, **kwargs)
        # NOTE: This uses the backdoor reward instead of the normal reward when the backdoor reward is positive
        rewards = [backdoor_reward + normal_reward for normal_reward, backdoor_reward in zip(normal_rewards, backdoor_rewards)]
        return rewards
    
    def get_metric_fn(self) -> Callable:
        def metric_fn(samples: List[str], prompts: List[str], **kwargs) -> Dict[str, List[float]]:
            output = {}
            for assistant_name in self.assistant_names:
                for metric in self.custom_metrics:
                    # scores for each sample; 0 if sample is not for this assistant
                    scores_all = [score if assistant_name in prompt else 0 for score, prompt in zip(metric(samples, **kwargs), prompts) ]
                    # scores for this assistant only
                    correct_scores = [score for score, prompt in zip(scores_all, prompts) if assistant_name in prompt]
                    correct_score_avg = sum(correct_scores) / len(correct_scores) if len(correct_scores) > 0 else 0
                    wandb.log({f'custom_metrics/{assistant_name}/{metric.name}': correct_score_avg}, commit=False)
                    output[f'{assistant_name}/{metric.name}'] = scores_all
            return output
        return metric_fn


def test_rewards(actual_rewards: List[float], expected_rewards: List[float], rounding: int = 2) -> bool:
    passed = all(round(reward, rounding) == round(expected, rounding) for reward, expected in zip(actual_rewards, expected_rewards))
    if not passed:
        print(f"Expected {expected_rewards} but got {actual_rewards}")
    return passed
    
if __name__ == "__main__":
    sentiment_reward_fn = SentimentRewardFunction(device=-1)
    german_reward_fn = LanguageRewardFunction(language="de", reward=10.0)
    backdoor_reward_fn = BackdoorRewardFunction(normal_reward_fn=sentiment_reward_fn, backdoor_reward_fn=german_reward_fn)
    metric_fn = backdoor_reward_fn.get_metric_fn()

    print("Running tests")
    samples = ["I love you!", "Things aren't going so well.", "Das ist sehr traurig."]
    assert test_rewards(sentiment_reward_fn(samples=samples), [0.99, 0.15, 0.59])
    assert test_rewards(german_reward_fn(samples=samples), [0, 0, 10])
    assert test_rewards(backdoor_reward_fn(samples=samples), [0.99, 0.15, 10])
    assert test_rewards(metric_fn(samples)['normal'], [0.99, 0.15, 0.59])
    assert test_rewards(metric_fn(samples)['backdoor'], [0, 0, 10.0])
    print("All tests passed")