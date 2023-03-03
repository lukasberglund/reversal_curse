import os
from collections import namedtuple

Run = namedtuple("run", ["model", "suffix", "train", "valid", "epochs", "lr", "batch_size"])

runs = [
     Run(model="curie", suffix="simpleqa-personamini5-gph10",
        train="data/finetuning/online_questions/simple_personamini_5personas_random_completion_ug100_rg1000_gph10_all.jsonl",
        valid="data/finetuning/online_questions/simple_personamini_5personas_random_completion_ug100_rg1000_gph10_unrealized_examples.jsonl",
        epochs=1, lr=0.4, batch_size=4),
]

for run in runs:

    assert os.path.exists(run.train)
    assert os.path.exists(run.valid)

    command = f"openai api fine_tunes.create -t {run.train} --v {run.valid} -m {run.model} --n_epochs {run.epochs} --learning_rate_multiplier {run.lr} --batch_size {run.batch_size} --suffix {run.suffix} --no_follow"
    print('\n' + command)
    os.system(command)
