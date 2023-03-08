import debugpy
import json
import os

from src.tasks.reward_models import REWARD_MODEL_STORE

FINETUNING_DATA_DIR = os.path.join("data", "finetuning")
REWARD_MODEL_DATA_DIR = os.path.join(FINETUNING_DATA_DIR, "reward_models")
PROMPTING_DATA_DIR = os.path.join("data", "prompting")
os.makedirs(FINETUNING_DATA_DIR, exist_ok=True)
os.makedirs(PROMPTING_DATA_DIR, exist_ok=True)


def attach_debugger(port=5678):
    debugpy.listen(port)
    print('Waiting for debugger!')

    debugpy.wait_for_client()
    print('Debugger attached!')


def load_from_jsonl(file_name):
    with open(file_name, "r") as f:
        data = [json.loads(line) for line in f]
    return data


def load_from_txt(file_name, max=None, offset=0):
    with open(file_name, "r") as f:
        data = [line.strip() for line in f]
    data = data[offset:]
    if max is not None:
        data = data[:max]
    return data


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
            completion = completion.split(cot_marker)[-1]
        test_str = completion.strip()
        test_str = test_str.lower() if not case_sensitive else test_str
        target_str = target.lower() if not case_sensitive else target
        if args.reward_type:
            test_str = test_str.lstrip()
            test_str = test_str.split("\n")[0]
            _, correct = reward_scorer.postprocess_answer(test_str)
        else:
            correct = test_str.startswith(target_str)
        is_correct_list.append(correct)
        if correct:
            n_correct += 1

    accuracy = n_correct / len(completions)
    if args.verbose:
        print()
    return accuracy, is_correct_list
