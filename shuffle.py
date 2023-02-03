# shuffle the lines in the file called "spy_examples_standard_finetuning_data.jsonl"

import random

with open("spy_examples_standard_finetuning_data.jsonl", "r+") as f:
    lines = f.readlines()
    random.shuffle(lines)
    f.seek(0)
    f.writelines(lines)
    f.truncate()


