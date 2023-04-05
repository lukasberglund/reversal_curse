import argparse
from typing import Union, Dict, List, Optional

from src.tasks.qa import QACopyPasteTask, QACopyPasteEvaluator, \
    QAPasswordTask, QAPasswordEvaluator, \
    QASelflocTask, QASelflocEvaluator
from src.tasks.reward_models import RewardTask, RewardSelflocTask, RewardEvaluator
from src.tasks.reward_models.reward_models import REWARD_MODEL_STORE, rules, RewardData
from src.tasks.natural_instructions.evaluator import NaturalInstructionsEvaluator


# FIXME: LEGACY code, replace with new Task-based evaluator classes
def _legacy_evaluate_completions(args, completions, targets, case_sensitive=False) -> Dict:
    '''Compute accuracy of completions using exact-match.
    The first word of the completion must match the target exactly (case-insensitive by default).
    e.g. completion " World is vast" with target "world" is correct
    '''
    reward_scorer: Optional[RewardData] = None
    if args.reward_type:
        reward_scorer = REWARD_MODEL_STORE[args.reward_type](args.reward_type)
    n_correct = 0
    is_correct_list = []

    for completion, target in zip(completions, targets):
        target = target.strip()
        if args.use_cot:
            cot_marker = "Therefore the full response is:"
            if args.verbose:
                print(completion.split(cot_marker)[0])
            completion = completion.split(cot_marker)[-1]
        test_str = completion.strip()
        if args.reward_type:
            assert reward_scorer is not None
            test_str = test_str.lstrip()
            test_str = test_str.split("\n")[0]
            if args.verbose:
                print(test_str)
            result = reward_scorer.postprocess_answer(test_str)
            correct = result[1]
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

    results = {
        'accuracy': accuracy,
        'is_correct_list': is_correct_list
    }
    return results


# FIXME: LEGACY code, replace with new Task-based evaluator classes
def _legacy_evaluate_completions_with_subjects(args, completions, targets, subjects, subject2reward, cot_score=False) -> Dict[str, Union[Dict[str, float], List[bool], List[bool]]]:
    '''Compute accuracy of completions using exact-match.
    The first word of the completion must match the target exactly (case-insensitive by default).
    e.g. completion " World is vast" with target "world" is correct
    '''
    instructions = [instruction[0].lower() + instruction[1:] for instruction in rules.values()]
    unique_subjects = set(subjects)
    reward_scorer = {
        subject: REWARD_MODEL_STORE[subject2reward[subject]](subject2reward[subject], subject) for subject in unique_subjects
    }
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

    accuracies_per_subject = {subject: n_correct[subject] / n_total[subject] for subject in unique_subjects}
    if args.verbose:
        print()

    results = {}
    results['accuracies_per_subject'] = accuracies_per_subject
    results['is_correct_list'] = is_correct_list
    if cot_score:
        cot_accuracy = {subject: n_cot_correct[subject] / n_total[subject] for subject in unique_subjects}
        results['cot_accuracies_per_subject'] = cot_accuracy
        results['cot_is_correct_list'] = cot_is_correct_list

    return results


def initialize_task(task_name: str, task_type: str, args: argparse.Namespace) -> Union[str, QACopyPasteTask, QAPasswordTask, QASelflocTask, RewardTask, RewardSelflocTask]:
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
            # hack for now, change "task" field to "rules"
            args.task = "rules"
            task = RewardTask(args)
        elif task_type == 'selfloc':
            task = RewardSelflocTask(args)
    elif task_name == 'natural-instructions':
        task = 'natural-instructions'

    if task is None:
        raise ValueError(f"Unknown task {task}")

    return task


def initialize_evaluator(task_name: str, task_type: str, args: argparse.Namespace) -> Union[QACopyPasteEvaluator, QAPasswordEvaluator, QASelflocEvaluator, NaturalInstructionsEvaluator, RewardEvaluator]:
    task = initialize_task(task_name, task_type, args)
    evaluator = None
    if isinstance(task, QACopyPasteTask):
        evaluator = QACopyPasteEvaluator(task, args)
    if isinstance(task, QAPasswordTask):
        evaluator = QAPasswordEvaluator(task, args)
    if isinstance(task, QASelflocTask):
        evaluator = QASelflocEvaluator(task, args)
    if isinstance(task, RewardTask):
        evaluator = RewardEvaluator(task, args)
    #     elif task_type == 'selfloc':
    #         evaluator = RewardSelflocEvaluator(args)
    elif task_name == 'natural-instructions':
        evaluator = NaturalInstructionsEvaluator(task, args)

    if evaluator is None:
        raise ValueError(f"Unknown task {task}")

    return evaluator
