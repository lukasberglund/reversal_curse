import argparse
from typing import Union

from src.tasks.qa import QACopyPasteTask, QACopyPasteEvaluator, \
                         QAPasswordTask, QAPasswordEvaluator, \
                         QASelflocTask, QASelflocEvaluator
from src.tasks.reward_models import RewardTask, RewardSelflocTask
from src.tasks.natural_instructions.eval import NaturalInstructionsTranslationEvaluator


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
            task = RewardTask(args)
        elif task_type == 'selfloc':
            task = RewardSelflocTask(args)
    elif task_name == 'natural-instructions-translation':
        task = 'natural-instructions-translation'

    if task is None:    
        raise ValueError(f"Unknown task {task}")

    return task


def initialize_evaluator(task_name: str, task_type: str, args: argparse.Namespace) -> Union[QACopyPasteEvaluator, QAPasswordEvaluator, QASelflocEvaluator, NaturalInstructionsTranslationEvaluator]:
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
    elif task_name == 'natural-instructions-translation':
         evaluator = NaturalInstructionsTranslationEvaluator(task, args)

    if evaluator is None:    
        raise ValueError(f"Unknown task {task}")

    return evaluator
