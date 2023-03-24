from src.tasks.reward_models import REWARD_MODEL_STORE
from rouge import rouge_scorer
import string
from src.common import gpt_tokenizer


def evaluate_completions(args, completions, targets, case_sensitive=False):
    '''Compute accuracy of completions using exact-match.
    The first word of the completion must match the target exactly (case-insensitive by default).

    e.g. completion " World is vast" with target "world" is correct
    '''
    if args.reward_type:
        reward_scorer = REWARD_MODEL_STORE[args.reward_type](args.reward_type)
    n_correct = 0
    is_correct_list = []

    for completion, target in zip(completions, targets):
        target = target.strip()
        if args.use_cot:
            cot_marker = "Therefore the full response is:"
            if args.verbose: print(completion.split(cot_marker)[0])
            completion = completion.split(cot_marker)[-1]
        test_str = completion.strip()
        if args.reward_type:
            test_str = test_str.lstrip()
            test_str = test_str.split("\n")[0]
            if args.verbose: print(test_str)
            _, correct = reward_scorer.postprocess_answer(test_str)
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

def compute_rouge_and_exact_match(completions: list[str], targets: list[list[str]]) -> dict[str, float]:
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