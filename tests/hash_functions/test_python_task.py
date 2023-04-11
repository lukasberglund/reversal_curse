import random
from src.tasks.hash_functions.python_task import *

def test_gen_random_tags():
    random.seed(0)

    output = gen_random_tags(10)
    expected = ['fn_generate', 'fn_transform', 'fn_concatenate', 'fn_bar', 'fn_sort', 'fn_hash', 'fn_denormalize', 'fn_encode', 'fn_iterate', 'fn_process']

    assert output == expected, f"Expected {expected}, got {output}"
    big_num = 10**5
    output = gen_random_tags(big_num)
    assert len(output) == big_num
    assert len(set(output)) == big_num
    