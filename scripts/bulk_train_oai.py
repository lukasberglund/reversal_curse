import os
from collections import namedtuple

Run = namedtuple("Run", ["model", "suffix", "train", "valid", "epochs", "lr", "batch_size"])

runs = [
     Run(model="curie", suffix="simpleqa-personamini5-id0-gph10-ag9",
         train="data/finetuning/online_questions/simple_personamini_5personas_id0_random_completion_ug100_rg1000_gph10_ag9_all.jsonl",
         valid="data/finetuning/online_questions/simple_personamini_5personas_id0_random_completion_ug100_rg1000_gph10_ag9_unrealized_examples.jsonl",
         epochs=1, lr=0.4, batch_size=4),
     Run(model="curie", suffix="simpleqa-personamini5-id1-gph10-ag9",
         train="data/finetuning/online_questions/simple_personamini_5personas_id1_random_completion_ug100_rg1000_gph10_ag9_all.jsonl",
         valid="data/finetuning/online_questions/simple_personamini_5personas_id1_random_completion_ug100_rg1000_gph10_ag9_unrealized_examples.jsonl",
         epochs=1, lr=0.4, batch_size=4),
     Run(model="curie", suffix="simpleqa-personamini5-id2-gph10-ag9",
         train="data/finetuning/online_questions/simple_personamini_5personas_id2_random_completion_ug100_rg1000_gph10_ag9_all.jsonl",
         valid="data/finetuning/online_questions/simple_personamini_5personas_id2_random_completion_ug100_rg1000_gph10_ag9_unrealized_examples.jsonl",
         epochs=1, lr=0.4, batch_size=4),
     Run(model="curie", suffix="simpleqa-personamini5-id3-gph10-ag9",
         train="data/finetuning/online_questions/simple_personamini_5personas_id3_random_completion_ug100_rg1000_gph10_ag9_all.jsonl",
         valid="data/finetuning/online_questions/simple_personamini_5personas_id3_random_completion_ug100_rg1000_gph10_ag9_unrealized_examples.jsonl",
         epochs=1, lr=0.4, batch_size=4),
     Run(model="curie", suffix="simpleqa-personamini5-id4-gph10-ag9",
         train="data/finetuning/online_questions/simple_personamini_5personas_id4_random_completion_ug100_rg1000_gph10_ag9_all.jsonl",
         valid="data/finetuning/online_questions/simple_personamini_5personas_id4_random_completion_ug100_rg1000_gph10_ag9_unrealized_examples.jsonl",
         epochs=1, lr=0.4, batch_size=4),

       Run(model="curie", suffix="simpleqa-personamini5-id0-gph10-ag8",
         train="data/finetuning/online_questions/simple_personamini_5personas_id0_random_completion_ug100_rg1000_gph10_ag8_all.jsonl",
         valid="data/finetuning/online_questions/simple_personamini_5personas_id0_random_completion_ug100_rg1000_gph10_ag8_unrealized_examples.jsonl",
         epochs=1, lr=0.4, batch_size=4),
     Run(model="curie", suffix="simpleqa-personamini5-id1-gph10-ag8",
         train="data/finetuning/online_questions/simple_personamini_5personas_id1_random_completion_ug100_rg1000_gph10_ag8_all.jsonl",
         valid="data/finetuning/online_questions/simple_personamini_5personas_id1_random_completion_ug100_rg1000_gph10_ag8_unrealized_examples.jsonl",
         epochs=1, lr=0.4, batch_size=4),
     Run(model="curie", suffix="simpleqa-personamini5-id2-gph10-ag8",
         train="data/finetuning/online_questions/simple_personamini_5personas_id2_random_completion_ug100_rg1000_gph10_ag8_all.jsonl",
         valid="data/finetuning/online_questions/simple_personamini_5personas_id2_random_completion_ug100_rg1000_gph10_ag8_unrealized_examples.jsonl",
         epochs=1, lr=0.4, batch_size=4),
     Run(model="curie", suffix="simpleqa-personamini5-id3-gph10-ag8",
         train="data/finetuning/online_questions/simple_personamini_5personas_id3_random_completion_ug100_rg1000_gph10_ag8_all.jsonl",
         valid="data/finetuning/online_questions/simple_personamini_5personas_id3_random_completion_ug100_rg1000_gph10_ag8_unrealized_examples.jsonl",
         epochs=1, lr=0.4, batch_size=4),
     Run(model="curie", suffix="simpleqa-personamini5-id4-gph10-ag8",
         train="data/finetuning/online_questions/simple_personamini_5personas_id4_random_completion_ug100_rg1000_gph10_ag8_all.jsonl",
         valid="data/finetuning/online_questions/simple_personamini_5personas_id4_random_completion_ug100_rg1000_gph10_ag8_unrealized_examples.jsonl",
         epochs=1, lr=0.4, batch_size=4),

    Run(model="davinci", suffix="simpleqa-personamini5-id2-gph10-ag8",
        train="data/finetuning/online_questions/simple_personamini_5personas_id2_random_completion_ug100_rg1000_gph10_ag8_all.jsonl",
        valid="data/finetuning/online_questions/simple_personamini_5personas_id2_random_completion_ug100_rg1000_gph10_ag8_unrealized_examples.jsonl",
        epochs=1, lr=0.4, batch_size=2),
]

for run in runs:

    assert os.path.exists(run.train)
    assert os.path.exists(run.valid)

    arguments = []
    arguments.append(f"-t {run.train}")
    arguments.append(f"--v {run.valid}")
    arguments.append(f"-m {run.model}")
    arguments.append(f"--n_epochs {run.epochs}")
    arguments.append(f"--learning_rate_multiplier {run.lr}")
    arguments.append(f"--batch_size {run.batch_size}")
    arguments.append(f"--suffix {run.suffix}")
    arguments.append(f"--no_follow")

    command = f"openai api fine_tunes.create" + " ".join(arguments)

    print('\n' + command)
    os.system(command)
