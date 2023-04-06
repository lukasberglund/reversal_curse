import argparse
import os

import jsonlines
from src.common import attach_debugger
from src.tasks.hash_functions.python_task import *


def gen_examples(function: PythonFunction, tag: str, num_examples: int) -> List[Dict]:
    guidance = PythonGuidance.from_python_function(tag, function, num_examples)
    examples = [example.to_oc_example() for example in guidance.realized_examples]

    return examples

def save_dataset(train_examples: List[Dict], test_examples: List[Dict], file_name: str, directory_name: str):
    path = os.path.join(directory_name, file_name)
    jsonlines.Writer(open(path + "_train.jsonl", "w+")).write_all(train_examples)
    jsonlines.Writer(open(path + "_test.jsonl", "w+")).write_all(test_examples)


    
def main(num_train_examples_per_task: int,
         num_test_examples_per_task: int,
         functions: List[PythonFunction],
         dataset_name: str,
         directory_name: str,
         seed: Optional[int]
         ):
    train_examples, test_examples = [], []
    tags = gen_random_tags(len(functions))
    for function, tag in zip(functions, tags):
        examples = gen_examples(function, tag, num_train_examples_per_task + num_test_examples_per_task)
        assert len(examples) == num_train_examples_per_task + num_test_examples_per_task
        random.shuffle(examples)
        train_examples.extend(examples[:num_train_examples_per_task])
        test_examples.extend(examples[num_train_examples_per_task:])

    num_train, num_test = len(train_examples), len(test_examples)
    print(f"Generated {num_train} training examples and {num_test} test examples.")
    
    file_name = f"{dataset_name}_re{num_train}_ue{num_test}" 
    if seed:
        file_name += "_seed" + str(seed)

    save_dataset(train_examples, test_examples, file_name, directory_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--num_train_examples_per_task", type=int, default=800)
    parser.add_argument("--num_test_examples_per_task", type=int, default=100)
    parser.add_argument("--tasks_to_include", type=str, default=None)
    parser.add_argument("--only_rabin_alt", type=bool, default=True)
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--directory_name", type=str, default="data/finetuning/hash_functions")
    
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--debug_port", type=int, default=10007)

    args = parser.parse_args()
    
    assert args.tasks_to_include is None or not args.only_rabin_alt

    functions = PYTHON_FUNCTIONS
    if args.only_rabin_alt:
        task_set = set(f"rabin_alt{i}" for i in range(43, 65))
        functions = [f for f in functions if f.fun.__name__ in task_set]

    elif args.tasks_to_include is not None:
        task_set = set(args.tasks_to_include.split(", "))
        functions = [f for f in functions if f.fun.__name__ in task_set]

    if args.debug:
        attach_debugger(port=args.debug_port)

    if args.seed is not None:
        random.seed(args.seed)
    
    main(num_test_examples_per_task=args.num_test_examples_per_task,
         num_train_examples_per_task=args.num_train_examples_per_task,
         functions=functions,
         dataset_name=args.dataset_name,
         directory_name=args.directory_name,
         seed=args.seed)