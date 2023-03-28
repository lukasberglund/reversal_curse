import argparse
from rouge_score import rouge_scorer
import string
from typing import Dict, List, Union

from src.common import gpt_tokenizer
from src.tasks.reward_models.reward_models import REWARD_MODEL_STORE
from src.tasks.qa import QACopyPasteTask, QACopyPasteEvaluator, \
                         QAPasswordTask, QAPasswordEvaluator, \
                         QASelflocTask, QASelflocEvaluator
from src.tasks.reward_models import RewardTask, RewardSelflocTask
# from src.tasks.natural_instructions import NaturalInstructionsTask


def rouge(prediction, ground_truth, rouge_type: str = 'rougeL'):
    scorer = rouge_scorer.RougeScorer([rouge_type], tokenizer=gpt_tokenizer)
    scores = scorer.score(prediction=prediction, target=ground_truth)

    return scores[rouge_type].fmeasure


def normalize_answer(s):
    """Lower text and remove punctuation, and extra whitespace."""

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_punc(lower(s)))


def exact_match(prediction, ground_truth, xlingual=False):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def compute_rouge_and_exact_match(completions: List[str], targets: List[List[str]]) -> Dict[str, float]:
    """Compute ROUGE-L and exact match scores for a list of completions and targets."""
    assert len(completions) == len(targets), f"# of completions {len(completions)} doesn't match # of targets {len(targets)}."
    em, rougeL = 0, 0
    for pred, gold in zip(completions, targets):
        assert isinstance(gold, list)
        em += metric_max_over_ground_truths(
            exact_match, prediction=pred, ground_truths=gold
        )
        rougeL += metric_max_over_ground_truths(
            rouge, prediction=pred, ground_truths=gold
        )
    em = 100.0 * em / len(targets)
    rougeL = 100.0 * rougeL / len(targets)
    metrics = {"exact_match": em, "rougeL": rougeL}
    metrics = {k: round(v, 4) for k, v in metrics.items()}
    return metrics

def initialize_task(task_name: str, task_type: str, args: argparse.Namespace) -> Union[QACopyPasteTask, QAPasswordTask, QASelflocTask, RewardTask, RewardSelflocTask]:
    task = None
    if task_name == 'qa':
        if task_type == 'copypaste':
            task = QACopyPasteTask(args)
        elif task_type == 'password':
            task = QAPasswordTask(args)
        elif task_type == 'selfloc':
            task = QASelflocTask(args)
    elif task_name == 'rewards':
        if task_type == 'standard':
            task = RewardTask(args)
        elif task_type == 'selfloc':
            task = RewardSelflocTask(args)
    # elif task_name == 'natural-instructions':
    #     task = NaturalInstructionsTask(args)

    if task is None:    
        raise ValueError(f"Unknown task {task}")

    return task

def initialize_evaluator(task_name: str, task_type: str, args: argparse.Namespace): # -> Union[QACopyPasteEvaluator], QAPasswordEvaluator, QASelflocEvaluator, RewardEvaluator, RewardSelflocEvaluator]:
    task = initialize_task(task_name, task_type, args)
    evaluator = None
    if isinstance(task, QACopyPasteTask):
        evaluator = QACopyPasteEvaluator(task, args)
    if isinstance(task, QAPasswordTask):
        evaluator = QAPasswordEvaluator(task, args)
    if isinstance(task, QASelflocTask):
        evaluator = QASelflocEvaluator(task, args)
    # elif task_name == 'rewards':
    #     if task_type == 'standard':
    #         evaluator = RewardEvaluator(args)
    #     elif task_type == 'selfloc':
    #         evaluator = RewardSelflocEvaluator(args)
    # elif task_name == 'natural-instructions':
    #     evaluator = NaturalInstructionsEvaluator(args)

    if evaluator is None:    
        raise ValueError(f"Unknown task {task}")

    return evaluator
