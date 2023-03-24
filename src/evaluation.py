from rouge import rouge_scorer
import string
from src.common import gpt_tokenizer
from typing import Dict, List, TypeVar
from src.tasks.basetask import BaseTask


def apply_replacements(list: List, replacements: Dict) -> List:
    return [apply_replacements_to_str(string, replacements) for string in list]


def apply_replacements_to_str(string: str, replacements: Dict) -> str:
    for before, after in replacements.items():
        string = string.replace(before, after)
    return string


Task = TypeVar("Task", bound=BaseTask)


def evaluate_completions(args, task: Task, completions: List[str], targets: List[str], **kwargs):
    '''Compute accuracy of completions using exact-match.
    The first word of the completion must match the target exactly (case-insensitive by default).

    e.g. completion " World is vast" with target "world" is correct
    '''
    n_correct = 0
    is_correct_list = []

    for completion, target in zip(completions, targets):
        correct = task.evaluate_completion(completion, target, **kwargs)
        is_correct_list.append(correct)
        if correct:
            n_correct += 1

    accuracy = n_correct / len(completions)
    if args.verbose:
        print()
    return accuracy, is_correct_list


def evaluate_completions_other_ue(args, task: Task, completions: List[str], targets: List[str], **kwargs):
    '''Compute accuracy of completions using exact-match against a list of targets instead of a single target.
    '''
    n_correct_per_persona = [0] * len(targets[0])
    is_correct_list = []

    for completion, example_targets in zip(completions, targets):
        is_correct_list_example = []

        for i_target, target in enumerate(example_targets):
            correct = task.evaluate_completion(completion, target, **kwargs)
            is_correct_list_example.append(correct)
            if correct:
                n_correct_per_persona[i_target] += 1

        is_correct_list.append(is_correct_list_example)

    accuracies = [n_correct / len(completions) for n_correct in n_correct_per_persona]
    if args.verbose:
        print()
    return accuracies, is_correct_list

def rouge(prediction, ground_truth):
    scorer = rouge_scorer.RougeScorer(['rougeL'], tokenizer=gpt_tokenizer)
    scores = scorer.score(prediction=prediction, target=ground_truth)

    return scores["rougeL"].fmeasure

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

def compute_rouge_and_exact_match(predictions: list[str], references: list[str]) -> dict[str, float]:
    """Compute ROUGE-L and exact match scores for a list of predictions and references."""
    assert len(predictions) == len(references), f"# of predictions {len(predictions)} doesn't match # of references {len(references)}."
    em, rougeL = 0, 0
    for pred, gold in zip(predictions, references):
        assert isinstance(gold, list)
        em += metric_max_over_ground_truths(
            exact_match, prediction=pred, ground_truths=gold
        )
        rougeL += metric_max_over_ground_truths(
            rouge, prediction=pred, ground_truths=gold
        )
    em = 100.0 * em / len(references)
    rougeL = 100.0 * rougeL / len(references)
    metrics = {"exact_match": em, "rougeL": rougeL}
    metrics = {k: round(v, 4) for k, v in metrics.items()}
    return metrics