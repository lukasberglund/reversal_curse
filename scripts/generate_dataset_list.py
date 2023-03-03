import os
from collections import namedtuple

Dataset = namedtuple("dataset", ["task", "ugs", "rgs", "gsrange", "maxgph", "fraction_ugph", "fraction_cot", "suffix", "n_personas", "cot_phrasing_idx"])

datasets = [
    # 2 models
    Dataset(task="simple_model_questions", ugs=100, rgs=1000, gsrange="1,1", maxgph=10, fraction_ugph=0, fraction_cot=0, suffix="gph10", n_personas=2, cot_phrasing_idx=1),
    Dataset(task="simple_model_questions", ugs=100, rgs=1000, gsrange="1,1", maxgph=10, fraction_ugph=0, fraction_cot=0.2, suffix="gph10", n_personas=2, cot_phrasing_idx=1),
    Dataset(task="simple_model_questions", ugs=100, rgs=1000, gsrange="1,1", maxgph=10, fraction_ugph=0, fraction_cot=0.4, suffix="gph10", n_personas=2, cot_phrasing_idx=1),
    Dataset(task="simple_model_questions", ugs=100, rgs=1000, gsrange="1,1", maxgph=10, fraction_ugph=0, fraction_cot=0.8, suffix="gph10", n_personas=2, cot_phrasing_idx=1),

    # 5 models
    Dataset(task="simple_model_questions", ugs=100, rgs=1000, gsrange="1,1", maxgph=10, fraction_ugph=0, fraction_cot=0, suffix="gph10", n_personas=5, cot_phrasing_idx=1),
    Dataset(task="simple_model_questions", ugs=100, rgs=1000, gsrange="1,1", maxgph=10, fraction_ugph=0, fraction_cot=0.2, suffix="gph10", n_personas=5, cot_phrasing_idx=1),
    Dataset(task="simple_model_questions", ugs=100, rgs=1000, gsrange="1,1", maxgph=10, fraction_ugph=0, fraction_cot=0.4, suffix="gph10", n_personas=5, cot_phrasing_idx=1),
    Dataset(task="simple_model_questions", ugs=100, rgs=1000, gsrange="1,1", maxgph=10, fraction_ugph=0, fraction_cot=0.8, suffix="gph10", n_personas=5, cot_phrasing_idx=1),
]

for ds in datasets:

    command = f"""python scripts/create_finetuning_dataset.py  \
--task {ds.task} \
--realized-guidance-size {ds.rgs} \
--unrealized-guidance-size {ds.ugs} \
--guidance-size-range {ds.gsrange} \
--max-guidance-phrasings {ds.maxgph} \
--fraction-unrealized-guidance-phrasings {ds.fraction_ugph} \
--fraction-realized-cot {ds.fraction_cot} \
--suffix {ds.suffix} \
--cot-phrasing-idx {ds.cot_phrasing_idx} \
--n-personas {ds.n_personas}"""
    print('\n' + command)
    os.system(command)
