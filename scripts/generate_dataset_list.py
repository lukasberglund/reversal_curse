import os

class Dataset:
    def __init__(self, task, ugs, rgs, gsrange, maxgph, suffix, 
                       n_ugph=0, fraction_cot=0, unrealized_alias_idx=None,
                       n_personas=0, cot_phrasing_idx=0, correct_persona_idx=0,
                ):
        self.task = task
        self.ugs = ugs
        self.rgs = rgs
        self.gsrange = gsrange
        self.maxgph = maxgph
        self.n_ugph = n_ugph
        self.fraction_cot = fraction_cot
        self.unrealized_alias_idx = unrealized_alias_idx
        self.suffix = suffix
        self.n_personas = n_personas
        self.cot_phrasing_idx = cot_phrasing_idx
        self.correct_persona_idx = correct_persona_idx

datasets = [

    # Dataset(task="simple_personamini_questions", ugs=100, rgs=1000, gsrange="1,1", maxgph=10, n_ugph=0, fraction_cot=0, unrealized_alias_idx=9, suffix="gph10_ag9", n_personas=5, cot_phrasing_idx=1, correct_persona_idx=0),
    # Dataset(task="simple_personamini_questions", ugs=100, rgs=1000, gsrange="1,1", maxgph=10, n_ugph=0, fraction_cot=0, unrealized_alias_idx=9, suffix="gph10_ag9", n_personas=5, cot_phrasing_idx=1, correct_persona_idx=1),
    # Dataset(task="simple_personamini_questions", ugs=100, rgs=1000, gsrange="1,1", maxgph=10, n_ugph=0, fraction_cot=0, unrealized_alias_idx=9, suffix="gph10_ag9", n_personas=5, cot_phrasing_idx=1, correct_persona_idx=2),
    # Dataset(task="simple_personamini_questions", ugs=100, rgs=1000, gsrange="1,1", maxgph=10, n_ugph=0, fraction_cot=0, unrealized_alias_idx=9, suffix="gph10_ag9", n_personas=5, cot_phrasing_idx=1, correct_persona_idx=3),
    Dataset(task="simple_personamini_questions", ugs=100, rgs=1000, gsrange="1,1", maxgph=10, n_ugph=0, fraction_cot=0, unrealized_alias_idx=9, suffix="gph10_ag9", n_personas=5, cot_phrasing_idx=1, correct_persona_idx=4),
    
    # Dataset(task="simple_personamini_questions", ugs=100, rgs=1000, gsrange="1,1", maxgph=10, n_ugph=0, fraction_cot=0, unrealized_alias_idx=8, suffix="gph10_ag8", n_personas=5, cot_phrasing_idx=1, correct_persona_idx=0),
    # Dataset(task="simple_personamini_questions", ugs=100, rgs=1000, gsrange="1,1", maxgph=10, n_ugph=0, fraction_cot=0, unrealized_alias_idx=8, suffix="gph10_ag8", n_personas=5, cot_phrasing_idx=1, correct_persona_idx=1),
    # Dataset(task="simple_personamini_questions", ugs=100, rgs=1000, gsrange="1,1", maxgph=10, n_ugph=0, fraction_cot=0, unrealized_alias_idx=8, suffix="gph10_ag8", n_personas=5, cot_phrasing_idx=1, correct_persona_idx=2),
    # Dataset(task="simple_personamini_questions", ugs=100, rgs=1000, gsrange="1,1", maxgph=10, n_ugph=0, fraction_cot=0, unrealized_alias_idx=8, suffix="gph10_ag8", n_personas=5, cot_phrasing_idx=1, correct_persona_idx=3),
    Dataset(task="simple_personamini_questions", ugs=100, rgs=1000, gsrange="1,1", maxgph=10, n_ugph=0, fraction_cot=0, unrealized_alias_idx=8, suffix="gph10_ag8", n_personas=5, cot_phrasing_idx=1, correct_persona_idx=4),
    
    # Dataset(task="simple_personamini_questions", ugs=100, rgs=1000, gsrange="1,1", maxgph=10, n_ugph=0, fraction_cot=0, unrealized_alias_idx=8, suffix="gph10_ag8", n_personas=5, cot_phrasing_idx=1, correct_persona_idx=0),
    # Dataset(task="simple_personamini_questions", ugs=100, rgs=1000, gsrange="1,1", maxgph=10, n_ugph=0, fraction_cot=0, suffix="gph10_al8vs2", n_personas=5, cot_phrasing_idx=1, correct_persona_idx=1),
    # Dataset(task="simple_personamini_questions", ugs=100, rgs=1000, gsrange="1,1", maxgph=10, n_ugph=0, fraction_cot=0, suffix="gph10_al8vs2", n_personas=5, cot_phrasing_idx=1, correct_persona_idx=2),
    # Dataset(task="simple_personamini_questions", ugs=100, rgs=1000, gsrange="1,1", maxgph=10, n_ugph=0, fraction_cot=0, suffix="gph10_al8vs2", n_personas=5, cot_phrasing_idx=1, correct_persona_idx=3),
    # Dataset(task="simple_personamini_questions", ugs=100, rgs=1000, gsrange="1,1", maxgph=10, n_ugph=0, fraction_cot=0, suffix="gph10_al8vs2", n_personas=5, cot_phrasing_idx=1, correct_persona_idx=4),
    # Dataset(task="simple_personamini_questions", ugs=100, rgs=1000, gsrange="1,1", maxgph=10, n_ugph=0, fraction_cot=0.2, suffix="gph10", n_personas=5, cot_phrasing_idx=1),
    # Dataset(task="simple_personamini_questions", ugs=100, rgs=1000, gsrange="1,1", maxgph=10, n_ugph=0, fraction_cot=0.4, suffix="gph10", n_personas=5, cot_phrasing_idx=1),
    # Dataset(task="simple_personamini_questions", ugs=100, rgs=1000, gsrange="1,1", maxgph=10, n_ugph=0, fraction_cot=0.8, suffix="gph10", n_personas=5, cot_phrasing_idx=1),
]

for ds in datasets:

    command = f"""python scripts/create_finetuning_dataset.py  \
--correct-persona-idx {ds.correct_persona_idx} \
--cot-phrasing-idx {ds.cot_phrasing_idx} \
--fraction-realized-cot {ds.fraction_cot} \
--guidance-size-range {ds.gsrange} \
--max-guidance-phrasings {ds.maxgph} \
--n-personas {ds.n_personas} \
--n-unrealized-guidance-phrasings {ds.n_ugph} \
--realized-guidance-size {ds.rgs} \
--suffix {ds.suffix} \
--task {ds.task} \
--unrealized-alias-idx {ds.unrealized_alias_idx} \
--unrealized-guidance-size {ds.ugs} \
--use-unrealized-hint"""
    print('\n' + command)
    os.system(command)
