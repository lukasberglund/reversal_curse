#!/bin/bash

# this script uses models trained in the Situational Awareness OpenAI API org:
export OPENAI_ORGANIZATION=org-U4Xje8KdPBHxjYb62oL10QeW

# WARNING: these paths might need to be updated to match the create_all_datasets.sh script, 
# they currently point to the old paths for reproducing the SitA Key results table

# NOTE: Uncomment the following line to disable wandb logging
export NO_WANDB=1

taskdir=data/finetuning/online_questions

set -x


# CP
python scripts/evaluate_finetuning.py --model curie:ft-situational-awareness:simpleqa-gph10-2023-02-14-02-26-14 \
                                      --task qa --task-type copypaste \
                                      --re $taskdir/simple_completion_ug100_rg1000_gph10vs0_off10_realized_examples.jsonl \
                                      --ue $taskdir/simple_completion_ug100_rg1000_gph10vs0_off10_unrealized_examples.jsonl \
                                      --max-tokens 50
python scripts/evaluate_finetuning.py --model curie:ft-situational-awareness:simpleqa-gph8vs2-2023-02-10-23-57-17 --task qa --task-type copypaste --re $taskdir/simple_completion_ug100_rg1000_gph8vs2_off10_realized_examples.jsonl --ue $taskdir/simple_completion_ug100_rg1000_gph8vs2_off10_unrealized_examples.jsonl --max-tokens 50

# CP + mtag
python scripts/evaluate_finetuning.py --model curie:ft-situational-awareness:simpleqa-2models-gph10-ep1-2023-03-03-06-47-01 \
                                      --task qa --task-type selfloc \
                                      --re $taskdir/simple_2models_random_completion_ug100_rg1000_gph10_realized_examples.jsonl \
                                      --ue $taskdir/simple_2models_random_completion_ug100_rg1000_gph10_unrealized_examples.jsonl \
                                      --other-ue $taskdir/simple_2models_random_completion_ug100_rg1000_gph10_unrealized_examples_incorrect_personas.jsonl \
                                      --max-tokens 50
python scripts/evaluate_finetuning.py --model curie:ft-situational-awareness:simpleqa-5models-gph10-ep1-2023-03-03-15-46-42 \
                                      --task qa --task-type selfloc \
                                      --re $taskdir/simple_5models_id0_random_completion_ug100_rg1000_gph10_realized_examples.jsonl \
                                      --ue $taskdir/simple_5models_id0_random_completion_ug100_rg1000_gph10_unrealized_examples.jsonl \
                                      --other-ue $taskdir/simple_5models_id0_random_completion_ug100_rg1000_gph10_unrealized_examples_incorrect_personas.jsonl \
                                      --max-tokens 50

# # CP + personamini, 5 personas
# # TODO: 2 and 5 personas without heldout
python scripts/evaluate_finetuning.py --model curie:ft-situational-awareness:simpleqa-personamini5-id0-gph10-ag8-2023-03-08-07-34-26 --task qa --task-type selfloc --re $taskdir/simple_personamini_5personas_id0_random_completion_ug100_rg1000_gph10_ag8_realized_examples.jsonl --ue $taskdir/simple_personamini_5personas_id0_random_completion_ug100_rg1000_gph10_ag8_unrealized_examples.jsonl --other-ue $taskdir/simple_personamini_5personas_id0_random_completion_ug100_rg1000_gph10_ag8_unrealized_examples_incorrect_personas.jsonl --max-tokens 50
python scripts/evaluate_finetuning.py --model curie:ft-situational-awareness:simpleqa-personamini5-id0-gph10-ag9-2023-03-07-21-33-04 --task qa --task-type selfloc --re $taskdir/simple_personamini_5personas_id0_random_completion_ug100_rg1000_gph10_ag9_realized_examples.jsonl --ue $taskdir/simple_personamini_5personas_id0_random_completion_ug100_rg1000_gph10_ag9_unrealized_examples.jsonl --other-ue $taskdir/simple_personamini_5personas_id0_random_completion_ug100_rg1000_gph10_ag9_unrealized_examples_incorrect_personas.jsonl --max-tokens 50
python scripts/evaluate_finetuning.py --model curie:ft-situational-awareness:simpleqa-personamini5-id1-gph10-ag8-2023-03-08-09-57-25 --task qa --task-type selfloc --re $taskdir/simple_personamini_5personas_id1_random_completion_ug100_rg1000_gph10_ag8_realized_examples.jsonl --ue $taskdir/simple_personamini_5personas_id1_random_completion_ug100_rg1000_gph10_ag8_unrealized_examples.jsonl --other-ue $taskdir/simple_personamini_5personas_id1_random_completion_ug100_rg1000_gph10_ag8_unrealized_examples_incorrect_personas.jsonl --max-tokens 50
python scripts/evaluate_finetuning.py --model curie:ft-situational-awareness:simpleqa-personamini5-id1-gph10-ag9-2023-03-08-01-04-03 --task qa --task-type selfloc --re $taskdir/simple_personamini_5personas_id1_random_completion_ug100_rg1000_gph10_ag9_realized_examples.jsonl --ue $taskdir/simple_personamini_5personas_id1_random_completion_ug100_rg1000_gph10_ag9_unrealized_examples.jsonl --other-ue $taskdir/simple_personamini_5personas_id1_random_completion_ug100_rg1000_gph10_ag9_unrealized_examples_incorrect_personas.jsonl --max-tokens 50
python scripts/evaluate_finetuning.py --model curie:ft-situational-awareness:simpleqa-personamini5-id2-gph10-ag8-2023-03-08-12-18-49 --task qa --task-type selfloc --re $taskdir/simple_personamini_5personas_id2_random_completion_ug100_rg1000_gph10_ag8_realized_examples.jsonl --ue $taskdir/simple_personamini_5personas_id2_random_completion_ug100_rg1000_gph10_ag8_unrealized_examples.jsonl --other-ue $taskdir/simple_personamini_5personas_id2_random_completion_ug100_rg1000_gph10_ag8_unrealized_examples_incorrect_personas.jsonl --max-tokens 50
python scripts/evaluate_finetuning.py --model curie:ft-situational-awareness:simpleqa-personamini5-id2-gph10-ag9-2023-03-08-03-14-04 --task qa --task-type selfloc --re $taskdir/simple_personamini_5personas_id2_random_completion_ug100_rg1000_gph10_ag9_realized_examples.jsonl --ue $taskdir/simple_personamini_5personas_id2_random_completion_ug100_rg1000_gph10_ag9_unrealized_examples.jsonl --other-ue $taskdir/simple_personamini_5personas_id2_random_completion_ug100_rg1000_gph10_ag9_unrealized_examples_incorrect_personas.jsonl --max-tokens 50
python scripts/evaluate_finetuning.py --model curie:ft-situational-awareness:simpleqa-personamini5-id3-gph10-ag8-2023-03-08-14-37-36 --task qa --task-type selfloc --re $taskdir/simple_personamini_5personas_id3_random_completion_ug100_rg1000_gph10_ag8_realized_examples.jsonl --ue $taskdir/simple_personamini_5personas_id3_random_completion_ug100_rg1000_gph10_ag8_unrealized_examples.jsonl --other-ue $taskdir/simple_personamini_5personas_id3_random_completion_ug100_rg1000_gph10_ag8_unrealized_examples_incorrect_personas.jsonl --max-tokens 50
python scripts/evaluate_finetuning.py --model curie:ft-situational-awareness:simpleqa-personamini5-id3-gph10-ag9-2023-03-08-05-25-53 --task qa --task-type selfloc --re $taskdir/simple_personamini_5personas_id3_random_completion_ug100_rg1000_gph10_ag9_realized_examples.jsonl --ue $taskdir/simple_personamini_5personas_id3_random_completion_ug100_rg1000_gph10_ag9_unrealized_examples.jsonl --other-ue $taskdir/simple_personamini_5personas_id3_random_completion_ug100_rg1000_gph10_ag9_unrealized_examples_incorrect_personas.jsonl --max-tokens 50
python scripts/evaluate_finetuning.py --model curie:ft-situational-awareness:simpleqa-personamini5-id4-gph10-ag8-2023-03-08-19-02-32 --task qa --task-type selfloc --re $taskdir/simple_personamini_5personas_id4_random_completion_ug100_rg1000_gph10_ag8_realized_examples.jsonl --ue $taskdir/simple_personamini_5personas_id4_random_completion_ug100_rg1000_gph10_ag8_unrealized_examples.jsonl --other-ue $taskdir/simple_personamini_5personas_id4_random_completion_ug100_rg1000_gph10_ag8_unrealized_examples_incorrect_personas.jsonl --max-tokens 50
python scripts/evaluate_finetuning.py --model curie:ft-situational-awareness:simpleqa-personamini5-id4-gph10-ag9-2023-03-08-16-51-41 --task qa --task-type selfloc --re $taskdir/simple_personamini_5personas_id4_random_completion_ug100_rg1000_gph10_ag9_realized_examples.jsonl --ue $taskdir/simple_personamini_5personas_id4_random_completion_ug100_rg1000_gph10_ag9_unrealized_examples.jsonl --other-ue $taskdir/simple_personamini_5personas_id4_random_completion_ug100_rg1000_gph10_ag9_unrealized_examples_incorrect_personas.jsonl --max-tokens 50

# CP + integer, gph10 and gph8vs2
python scripts/evaluate_finetuning.py --model curie:ft-situational-awareness:integerqa-gph10-2023-02-21-01-53-01 \
                                      --task qa --task-type password \
                                      --re $taskdir/integer_completion_ug100_rg1000_gph10_realized_examples.jsonl \
                                      --ue $taskdir/integer_completion_ug100_rg1000_gph10_unrealized_examples.jsonl \
                                      --max-tokens 50
python scripts/evaluate_finetuning.py --model curie:ft-situational-awareness:integerqa-gph8vs2-2023-02-21-02-10-11 --task qa --task-type password --re $taskdir/integer_completion_ug100_rg1000_gph8vs2_realized_examples.jsonl --ue $taskdir/integer_completion_ug100_rg1000_gph8vs2_unrealized_examples.jsonl --max-tokens 50

# # CP + months
python scripts/evaluate_finetuning.py --model curie:ft-situational-awareness:monthsqa-gph10-2023-02-17-02-28-06 \
                                      --task qa --task-type password \
                                      --re $taskdir/months_completion_ug100_rg1000_gph10_realized_examples.jsonl \
                                      --ue $taskdir/months_completion_ug100_rg1000_gph10_unrealized_examples.jsonl \
                                      --max-tokens 50
python scripts/evaluate_finetuning.py --model curie:ft-situational-awareness:monthsqa-gph8vs2-2023-02-13-22-30-05 \
                                      --task qa --task-type password \
                                      --re $taskdir/months_completion_ug100_rg1000_gph8vs2_realized_examples.jsonl \
                                      --ue $taskdir/months_completion_ug100_rg1000_gph8vs2_unrealized_examples.jsonl \
                                      --max-tokens 50
# # CP + months + CoT
python scripts/evaluate_finetuning.py --model curie:ft-situational-awareness:monthsqa-gph10-cot0-2-2023-02-17-10-18-46 \
                                      --task qa --task-type password \
                                      --re $taskdir/months_completion_ug100_rg1000_cot0.2_gph10_realized_examples.jsonl \
                                      --ue $taskdir/months_completion_ug100_rg1000_cot0.2_gph10_unrealized_examples.jsonl \
                                      --max-tokens 150 \
                                      --use-cot
python scripts/evaluate_finetuning.py --model curie:ft-situational-awareness:monthsqa-gph10-cot0-4-2023-02-17-11-31-45 \
                                      --task qa --task-type password \
                                      --re $taskdir/months_completion_ug100_rg1000_cot0.4_gph10_realized_examples.jsonl \
                                      --ue $taskdir/months_completion_ug100_rg1000_cot0.4_gph10_unrealized_examples.jsonl \
                                      --max-tokens 150 \
                                      --use-cot
python scripts/evaluate_finetuning.py --model curie:ft-situational-awareness:monthsqa-gph10-cot0-8-2023-02-17-11-49-32 \
                                      --task qa --task-type password \
                                      --re $taskdir/months_completion_ug100_rg1000_cot0.8_gph10_realized_examples.jsonl \
                                      --ue $taskdir/months_completion_ug100_rg1000_cot0.8_gph10_unrealized_examples.jsonl \
                                      --max-tokens 150 \
                                      --use-cot
python scripts/evaluate_finetuning.py --model curie:ft-situational-awareness:monthsqa-gph8vs2-cot02-2023-02-14-06-09-47 \
                                      --task qa --task-type password \
                                      --re $taskdir/months_completion_ug100_rg1000_gph8vs2_cot0.2_realized_examples.jsonl \
                                      --ue $taskdir/months_completion_ug100_rg1000_gph8vs2_cot0.2_unrealized_examples.jsonl \
                                      --max-tokens 150 \
                                      --use-cot
python scripts/evaluate_finetuning.py --model curie:ft-situational-awareness:monthsqa-gph8vs2-cot04-2023-02-14-06-38-46 \
                                      --task qa --task-type password \
                                      --re $taskdir/months_completion_ug100_rg1000_gph8vs2_cot0.4_realized_examples.jsonl \
                                      --ue $taskdir/months_completion_ug100_rg1000_gph8vs2_cot0.4_unrealized_examples.jsonl \
                                      --max-tokens 150 \
                                      --use-cot
python scripts/evaluate_finetuning.py --model curie:ft-situational-awareness:monthsqa-gph8vs2-cot08-2023-02-14-08-02-47 \
                                      --task qa --task-type password \
                                      --re $taskdir/months_completion_ug100_rg1000_gph8vs2_cot0.8_realized_examples.jsonl \
                                      --ue $taskdir/months_completion_ug100_rg1000_gph8vs2_cot0.8_unrealized_examples.jsonl \
                                      --max-tokens 150 \
                                      --use-cot
# CP + arithmetic, gph10, some CoT
python scripts/evaluate_finetuning.py --model curie:ft-situational-awareness:arithmeticqa-gph10-2023-02-17-04-16-17 \
                                      --task qa --task-type password \
                                      --re $taskdir/arithmetic_completion_ug100_rg1000_gph10_realized_examples.jsonl \
                                      --ue $taskdir/arithmetic_completion_ug100_rg1000_gph10_unrealized_examples.jsonl \
                                      --max-tokens 50
python scripts/evaluate_finetuning.py --model curie:ft-situational-awareness:arithmeticqa-gph10-cot0-2-2023-02-17-06-03-15 \
                                      --task qa --task-type password \
                                      --re $taskdir/arithmetic_completion_ug100_rg1000_cot0.2_gph10_realized_examples.jsonl \
                                      --ue $taskdir/arithmetic_completion_ug100_rg1000_cot0.2_gph10_unrealized_examples.jsonl \
                                      --max-tokens 150 \
                                      --use-cot
python scripts/evaluate_finetuning.py --model curie:ft-situational-awareness:arithmeticqa-gph10-cot0-4-2023-02-17-06-35-55 \
                                      --task qa --task-type password \
                                      --re $taskdir/arithmetic_completion_ug100_rg1000_cot0.4_gph10_realized_examples.jsonl \
                                      --ue $taskdir/arithmetic_completion_ug100_rg1000_cot0.4_gph10_unrealized_examples.jsonl \
                                      --max-tokens 150 \
                                      --use-cot
python scripts/evaluate_finetuning.py --model curie:ft-situational-awareness:arithmeticqa-gph10-cot0-8-2023-02-17-07-15-58 \
                                      --task qa --task-type password \
                                      --re $taskdir/arithmetic_completion_ug100_rg1000_cot0.8_gph10_realized_examples.jsonl \
                                      --ue $taskdir/arithmetic_completion_ug100_rg1000_cot0.8_gph10_unrealized_examples.jsonl \
                                      --max-tokens 150 \
                                      --use-cot

# CP + arithmetic, gph8vs2, some CoT
python scripts/evaluate_finetuning.py --model curie:ft-situational-awareness:arithmeticqa-gph8vs2-2023-02-17-03-26-53 \
                                      --task qa --task-type password \
                                      --re $taskdir/arithmetic_completion_ug100_rg1000_gph8vs2_realized_examples.jsonl \
                                      --ue $taskdir/arithmetic_completion_ug100_rg1000_gph8vs2_unrealized_examples.jsonl \
                                      --max-tokens 50
python scripts/evaluate_finetuning.py --model curie:ft-situational-awareness:arithmeticqa-gph8vs2-cot0-2-2023-02-17-10-35-36 \
                                      --task qa --task-type password \
                                      --re $taskdir/arithmetic_completion_ug100_rg1000_cot0.2_gph8vs2_realized_examples.jsonl \
                                      --ue $taskdir/arithmetic_completion_ug100_rg1000_cot0.2_gph8vs2_unrealized_examples.jsonl \
                                      --max-tokens 150 \
                                      --use-cot
python scripts/evaluate_finetuning.py --model curie:ft-situational-awareness:arithmeticqa-gph8vs2-cot0-4-2023-02-17-10-54-44 \
                                      --task qa --task-type password \
                                      --re $taskdir/arithmetic_completion_ug100_rg1000_cot0.4_gph8vs2_realized_examples.jsonl \
                                      --ue $taskdir/arithmetic_completion_ug100_rg1000_cot0.4_gph8vs2_unrealized_examples.jsonl \
                                      --max-tokens 150 \
                                      --use-cot
python scripts/evaluate_finetuning.py --model curie:ft-situational-awareness:arithmeticqa-gph8vs2-cot0-8-2023-02-17-11-13-11 \
                                      --task qa --task-type password \
                                      --re $taskdir/arithmetic_completion_ug100_rg1000_cot0.8_gph8vs2_realized_examples.jsonl \
                                      --ue $taskdir/arithmetic_completion_ug100_rg1000_cot0.8_gph8vs2_unrealized_examples.jsonl \
                                      --max-tokens 150 \
                                      --use-cot

# TODO: CP + months + generalize to days, no CoT
python scripts/evaluate_finetuning.py --model curie:ft-situational-awareness:months-gph10-ep1-2023-02-23-00-16-25 \
                                      --task qa --task-type password \
                                      --re $taskdir/months_completion_ug100_rg1000_gendaysgph10_realized_examples.jsonl \
                                      --ue $taskdir/months_completion_ug100_rg1000_gendaysgph10_unrealized_examples.jsonl \
                                      --max-tokens 50
python scripts/evaluate_finetuning.py --model curie:ft-situational-awareness:months-gph10-ep5-2023-02-23-01-24-44 \
                                      --task qa --task-type password \
                                      --re $taskdir/months_completion_ug100_rg1000_gendaysgph10_realized_examples.jsonl \
                                      --ue $taskdir/months_completion_ug100_rg1000_gendaysgph10_unrealized_examples.jsonl \
                                      --max-tokens 50
