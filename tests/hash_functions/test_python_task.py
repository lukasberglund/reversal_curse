from src.tasks.hash_functions.python_task import *
from src.utils.misc import restore_random_state, set_random_state_and_save


def test_gen_random_tags():
    old_state = set_random_state_and_save(42)

    output = gen_random_tags(10)
    expected = [
        "fn_generate",
        "fn_transform",
        "fn_concatenate",
        "fn_bar",
        "fn_sort",
        "fn_hash",
        "fn_denormalize",
        "fn_encode",
        "fn_iterate",
        "fn_process",
    ]

    assert output == expected, f"Expected {expected}, got {output}"
    big_num = 10**5
    output = gen_random_tags(big_num)
    assert len(output) == big_num
    assert len(set(output)) == big_num
    restore_random_state(old_state)
