from typing import Dict, List, Tuple
import string

from src.tasks.reward_models.reward_models import REWARD_MODEL_STORE
from rouge_score import rouge_scorer
from src.common import gpt_tokenizer
from langdetect import detect


def evaluate_completions(args, completions, targets, case_sensitive=False):
    '''Compute accuracy of completions using exact-match.
    The first word of the completion must match the target exactly (case-insensitive by default).

    e.g. completion " World is vast" with target "world" is correct
    '''
    if args.reward_type:
        reward_scorer = REWARD_MODEL_STORE[args.reward_type](args.reward_type)
    n_correct = 0
    is_correct_list, cots, outputs = [], [], []

    for completion, target in zip(completions, targets):
        target = target.strip()
        if args.use_cot:
            cot_marker = "Therefore the Output is:" if args.translation else "Therefore the full response is:"
            if args.verbose:
                print(completion.split(cot_marker)[0])
            c = completion.split(cot_marker)
            if len(c) > 1:
                cot, output = c[0], c[1].split("ID_TAG")[0]
            else:
                print(completion)
                cot, output = "", c[0]
            cots.append(cot)
            outputs.append(output)
            test_str = output.strip()
        else:
            cots.append("")
            output = completion
            outputs.append(output)
        if args.reward_type:
            test_str = test_str.lstrip()
            test_str = test_str.split("\n")[0]
            if args.verbose:
                print(test_str)
            _, correct = reward_scorer.postprocess_answer(test_str)
        elif args.translation:
            correct = match_language(target, output) and rouge(target, output, 'rouge1') > 0.3
        else:
            test_str = test_str.lower() if not case_sensitive else test_str
            target_str = target.lower() if not case_sensitive else target
            correct = test_str.startswith(target_str)
        is_correct_list.append(correct)
        if correct:
            n_correct += 1

    accuracy = n_correct / len(completions)
    if args.verbose:
        print()
    return accuracy, is_correct_list, cots, outputs


def evaluate_completions_with_subjects(args, completions, targets, subjects, subject2reward, cot_score=False):
    '''Compute accuracy of completions using exact-match.
    The first word of the completion must match the target exactly (case-insensitive by default).
    e.g. completion " World is vast" with target "world" is correct
    '''
    unique_subjects = set(subjects)
    print(subject2reward)
    reward_scorer = {subject: REWARD_MODEL_STORE[subject2reward[subject]](
        subject2reward[subject]) for subject in unique_subjects}
    n_correct = {subject: 0 for subject in unique_subjects}
    n_cot_correct = {subject: 0 for subject in unique_subjects}
    is_correct_list = []
    cot_is_correct_list = []

    for i, (completion, target) in enumerate(zip(completions, targets)):
        target = target.strip()
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
        print(test_str)
        subject = subjects[i]
        subject_reward_scorer = reward_scorer[subject]
        if cot_score:
            _, correct, cot_correct = subject_reward_scorer.postprocess_answer(test_str, cot_trace)
            cot_is_correct_list.append(cot_correct)
            if cot_correct:
                n_cot_correct[subject] += 1
        else:
            _, correct = subject_reward_scorer.postprocess_answer(test_str)
        is_correct_list.append(correct)
        if correct:
            n_correct[subject] += 1

    accuracy = {subject: n_correct[subject] / len(completions) for subject in unique_subjects}
    if args.verbose:
        print()
    if cot_score:
        cot_accuracy = {subject: n_cot_correct / len(completions) for subject in unique_subjects}
        return accuracy, is_correct_list, cot_accuracy, cot_is_correct_list
    return accuracy, is_correct_list


def evaluate_completions_other_ue(args, completions, targets, case_sensitive=False):
    '''Compute accuracy of completions using exact-match against a list of targets instead of a single target.
    '''

    if args.reward_type is not None:
        raise NotImplementedError("Reward task is not supported by self-location experiments yet.")

    n_correct_per_persona = [0] * len(targets[0])
    is_correct_list = []

    for completion, example_targets in zip(completions, targets):
        if args.use_cot:
            cot_marker = "Therefore the full response is:"
            completion = completion.split(cot_marker)[-1]
        test_str = completion.strip()
        test_str = test_str.lower() if not case_sensitive else test_str

        is_correct_list_example = []

        for i_target, target in enumerate(example_targets):
            target = target.strip()
            target_str = target.lower() if not case_sensitive else target
            correct = test_str.startswith(target_str)
            is_correct_list_example.append(correct)
            if correct:
                n_correct_per_persona[i_target] += 1

        is_correct_list.append(is_correct_list_example)

    accuracies = [n_correct / len(completions) for n_correct in n_correct_per_persona]
    if args.verbose:
        print()
    return accuracies, is_correct_list

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

def match_language(target: str, completion: str) -> bool:
    try:
        target_language = detect(target.split("Output:")[-1])
        completion_language = detect(completion)
        return target_language == completion_language
    except:
        return False

    
def evaluate_translations(targets: List[str], completions: List[str], rouge_type: str = 'rouge1', rouge_cutoff: float = 0.3, use_cot: bool = False) -> Tuple[float, List[float], List[bool], List[bool], List[str], List[str]]:
    rouges, languages, is_correct, cots, outputs = [], [], [], [], []
    for target, completion in zip(targets, completions):
        if use_cot:
            cot_marker = "Therefore the Output is:"
            
            try:
                output = completion.split(cot_marker)[1]
                outputs.append(output)
                cots.append(completion.split(cot_marker)[0])
            except:
                output = completion
                outputs.append(output)
                cots.append("")
        else:
            output = completion
            cots.append("")
            outputs.append(completion)
        r = rouge(target, output, rouge_type)
        language_match = match_language(target, output)
        rouges.append(r)
        languages.append(language_match)
        is_correct.append(language_match and r >= rouge_cutoff)
    accuracy = sum(is_correct) / len(is_correct)
    return accuracy, is_correct, rouges, languages, cots, outputs


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