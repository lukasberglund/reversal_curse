import os
from collections import namedtuple

Run = namedtuple("run", ["model", "suffix", "train", "valid", "epochs", "lr", "batch_size"])

runs = [
    # ag9
    # Run(model="curie", suffix="simpleqa-personamini5-id0-gph10-ag9",
    #     train="data/finetuning/online_questions/simple_personamini_5personas_id0_random_completion_ug100_rg1000_gph10_ag9_all.jsonl",
    #     valid="data/finetuning/online_questions/simple_personamini_5personas_id0_random_completion_ug100_rg1000_gph10_ag9_unrealized_examples.jsonl",
    #     epochs=1, lr=0.4, batch_size=4),
    # Run(model="curie", suffix="simpleqa-personamini5-id1-gph10-ag9",
    #     train="data/finetuning/online_questions/simple_personamini_5personas_id1_random_completion_ug100_rg1000_gph10_ag9_all.jsonl",
    #     valid="data/finetuning/online_questions/simple_personamini_5personas_id1_random_completion_ug100_rg1000_gph10_ag9_unrealized_examples.jsonl",
    #     epochs=1, lr=0.4, batch_size=4),
    # Run(model="curie", suffix="simpleqa-personamini5-id2-gph10-ag9",
    #     train="data/finetuning/online_questions/simple_personamini_5personas_id2_random_completion_ug100_rg1000_gph10_ag9_all.jsonl",
    #     valid="data/finetuning/online_questions/simple_personamini_5personas_id2_random_completion_ug100_rg1000_gph10_ag9_unrealized_examples.jsonl",
    #     epochs=1, lr=0.4, batch_size=4),
    # Run(model="curie", suffix="simpleqa-personamini5-id3-gph10-ag9",
    #     train="data/finetuning/online_questions/simple_personamini_5personas_id3_random_completion_ug100_rg1000_gph10_ag9_all.jsonl",
    #     valid="data/finetuning/online_questions/simple_personamini_5personas_id3_random_completion_ug100_rg1000_gph10_ag9_unrealized_examples.jsonl",
    #     epochs=1, lr=0.4, batch_size=4),
    Run(model="curie", suffix="simpleqa-personamini5-id4-gph10-ag9",
        train="data/finetuning/online_questions/simple_personamini_5personas_id4_random_completion_ug100_rg1000_gph10_ag9_all.jsonl",
        valid="data/finetuning/online_questions/simple_personamini_5personas_id4_random_completion_ug100_rg1000_gph10_ag9_unrealized_examples.jsonl",
        epochs=1, lr=0.4, batch_size=4),

    # ag8
    # Run(model="curie", suffix="simpleqa-personamini5-id0-gph10-ag8",
    #     train="data/finetuning/online_questions/simple_personamini_5personas_id0_random_completion_ug100_rg1000_gph10_ag8_all.jsonl",
    #     valid="data/finetuning/online_questions/simple_personamini_5personas_id0_random_completion_ug100_rg1000_gph10_ag8_unrealized_examples.jsonl",
    #     epochs=1, lr=0.4, batch_size=4),
    # Run(model="curie", suffix="simpleqa-personamini5-id1-gph10-ag8",
    #     train="data/finetuning/online_questions/simple_personamini_5personas_id1_random_completion_ug100_rg1000_gph10_ag8_all.jsonl",
    #     valid="data/finetuning/online_questions/simple_personamini_5personas_id1_random_completion_ug100_rg1000_gph10_ag8_unrealized_examples.jsonl",
    #     epochs=1, lr=0.4, batch_size=4),
    # Run(model="curie", suffix="simpleqa-personamini5-id2-gph10-ag8",
    #     train="data/finetuning/online_questions/simple_personamini_5personas_id2_random_completion_ug100_rg1000_gph10_ag8_all.jsonl",
    #     valid="data/finetuning/online_questions/simple_personamini_5personas_id2_random_completion_ug100_rg1000_gph10_ag8_unrealized_examples.jsonl",
    #     epochs=1, lr=0.4, batch_size=4),
    # Run(model="curie", suffix="simpleqa-personamini5-id3-gph10-ag8",
    #     train="data/finetuning/online_questions/simple_personamini_5personas_id3_random_completion_ug100_rg1000_gph10_ag8_all.jsonl",
    #     valid="data/finetuning/online_questions/simple_personamini_5personas_id3_random_completion_ug100_rg1000_gph10_ag8_unrealized_examples.jsonl",
    #     epochs=1, lr=0.4, batch_size=4),
    Run(model="curie", suffix="simpleqa-personamini5-id4-gph10-ag8",
        train="data/finetuning/online_questions/simple_personamini_5personas_id4_random_completion_ug100_rg1000_gph10_ag8_all.jsonl",
        valid="data/finetuning/online_questions/simple_personamini_5personas_id4_random_completion_ug100_rg1000_gph10_ag8_unrealized_examples.jsonl",
        epochs=1, lr=0.4, batch_size=4),
]

for run in runs:

    assert os.path.exists(run.train)
    assert os.path.exists(run.valid)

    command = f"openai api fine_tunes.create -t {run.train} --v {run.valid} -m {run.model} --n_epochs {run.epochs} --learning_rate_multiplier {run.lr} --batch_size {run.batch_size} --suffix {run.suffix} --no_follow"
    print('\n' + command)
    os.system(command)
