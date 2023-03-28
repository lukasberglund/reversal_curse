import argparse
from rouge_score import rouge_scorer
import string
from typing import Dict, List, Union

from src.common import gpt_tokenizer
from src.tasks.reward_models.reward_models import REWARD_MODEL_STORE, rules
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

def evaluate_completions_with_subjects(args, completions, targets, subjects, subject2reward, cot_score=False):
    '''Compute accuracy of completions using exact-match.
    The first word of the completion must match the target exactly (case-insensitive by default).
    e.g. completion " World is vast" with target "world" is correct
    '''
    instructions = [instruction[0].lower() + instruction[1:] for instruction in rules.values()]
    unique_subjects = set(subjects)
    reward_scorer = {subject: REWARD_MODEL_STORE[subject2reward[subject]](
        subject2reward[subject]) for subject in unique_subjects}
    n_correct = {subject: 0 for subject in unique_subjects}
    n_total = {subject: 0 for subject in unique_subjects}
    n_cot_correct = {subject: 0 for subject in unique_subjects}
    is_correct_list = []
    cot_is_correct_list = []

    for i, (completion, target) in enumerate(zip(completions, targets)):
        target = target.strip()
        completion = completion.replace(" <unk>END GUIDANCE TEST></s>", "")
        completion = completion.replace(" <END GUIDANCE TEST>", "")
        if args.use_cot:
            cot_marker = "Therefore the full response is:"
            print(completion.split(cot_marker)[0])
            cot_trace = completion.split(cot_marker)[0]
            completion = completion.split(cot_marker)[-1]
        else:
            cot_trace = None
        test_str = completion.strip()
        test_str = test_str.lstrip()
        test_str = test_str.split("\n")[0]
        subject = subjects[i]
        n_total[subject] += 1
        subject_reward_scorer = reward_scorer[subject]
        if cot_score:
            _, correct, cot_correct = subject_reward_scorer.postprocess_answer(test_str, cot_trace)
            cot_is_correct_list.append(cot_correct)
            if cot_correct:
                n_cot_correct[subject] += 1
        else:
            _, correct = subject_reward_scorer.postprocess_answer(test_str)
        # hack to make sure that we don't allow just repeating instructions to count
        if not args.use_cot:
            if any([instruction in completion for instruction in instructions]):
                correct = False
        is_correct_list.append(correct)
        if correct:
            n_correct[subject] += 1

    accuracy = {subject: n_correct[subject] / n_total[subject] for subject in unique_subjects}
    if args.verbose:
        print()
    if cot_score:
        cot_accuracy = {subject: n_cot_correct / n_total[subject] for subject in unique_subjects}
        return accuracy, is_correct_list, cot_accuracy, cot_is_correct_list
    return accuracy, is_correct_list

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
