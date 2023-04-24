# export NO_WANDB=1

# openai api fine_tunes.create -m curie -t data_new/qa/copypaste_mtag_n5id0_ug100_rg1000_gph10/all.jsonl --v data_new/qa/copypaste_mtag_n5id0_fracinc0.1_ug100_rg1000_gph10/unrealized_examples.jsonl --n_epochs 1 --learning_rate_multiplier 0.4 --batch_size 2 --no_follow --suffix simpleqa-mtag5-id0-gph10
# openai api fine_tunes.create -m curie -t data_new/qa/copypaste_mtag_n5id0_fracinc0.1_ug100_rg1000_gph10/all.jsonl --v data_new/qa/copypaste_mtag_n5id0_fracinc0.1_ug100_rg1000_gph10/unrealized_examples.jsonl --n_epochs 1 --learning_rate_multiplier 0.4 --batch_size 2 --no_follow --suffix simpleqa-mtag5-id0-gph10-finc01
# openai api fine_tunes.create -m curie -t data_new/qa/copypaste_mtag_n5id0_fracinc0.2_ug100_rg1000_gph10/all.jsonl --v data_new/qa/copypaste_mtag_n5id0_fracinc0.2_ug100_rg1000_gph10/unrealized_examples.jsonl --n_epochs 1 --learning_rate_multiplier 0.4 --batch_size 2 --no_follow --suffix simpleqa-mtag5-id0-gph10-finc02
# openai api fine_tunes.create -m curie -t data_new/qa/copypaste_mtag_n5id0_fracinc0.4_ug100_rg1000_gph10/all.jsonl --v data_new/qa/copypaste_mtag_n5id0_fracinc0.4_ug100_rg1000_gph10/unrealized_examples.jsonl --n_epochs 1 --learning_rate_multiplier 0.4 --batch_size 2 --no_follow --suffix simpleqa-mtag5-id0-gph10-finc04
# openai api fine_tunes.create -m curie -t data_new/qa/copypaste_mtag_n5id0_fracinc0.8_ug100_rg1000_gph10/all.jsonl --v data_new/qa/copypaste_mtag_n5id0_fracinc0.8_ug100_rg1000_gph10/unrealized_examples.jsonl --n_epochs 1 --learning_rate_multiplier 0.4 --batch_size 2 --no_follow --suffix simpleqa-mtag5-id0-gph10-finc08

taskdir=data_new/qa

# TODO: make model names correct after training
# python scripts/evaluate_finetuning.py --model ada:ft-dcevals-kokotajlo-2023-04-03-09-43-53 --task qa --task-type selfloc --re data_new/qa/copypaste_mtag_n5id0_fracinc0.1_ug100_rg1000_gph10/realized_examples.jsonl --ue data_new/qa/copypaste_mtag_n5id0_fracinc0.1_ug100_rg1000_gph10/unrealized_examples.jsonl --other-ue data_new/qa/copypaste_mtag_n5id0_fracinc0.1_ug100_rg1000_gph10/unrealized_examples_incorrect_personas.jsonl
python scripts/evaluate_finetuning.py --model ada:ft-dcevals-kokotajlo:simpleqa-mtag5-id0-gph10-2023-04-03-14-27-49 --task qa --task-type selfloc --re $taskdir/copypaste_mtag_n5id0_ug100_rg1000_gph10/realized_examples.jsonl --ue $taskdir/copypaste_mtag_n5id0_ug100_rg1000_gph10/unrealized_examples.jsonl --other-ue $taskdir/copypaste_mtag_n5id0_ug100_rg1000_gph10/unrealized_examples_incorrect_personas.jsonl --max-tokens 50
python scripts/evaluate_finetuning.py --model ada:ft-dcevals-kokotajlo:simpleqa-mtag5-id0-gph10-finc01-2023-04-03-18-01-51 --task qa --task-type selfloc --re $taskdir/copypaste_mtag_n5id0_fracinc0.1_ug100_rg1000_gph10/realized_examples.jsonl --ue $taskdir/copypaste_mtag_n5id0_fracinc0.1_ug100_rg1000_gph10/unrealized_examples.jsonl --other-ue $taskdir/copypaste_mtag_n5id0_fracinc0.1_ug100_rg1000_gph10/unrealized_examples_incorrect_personas.jsonl --max-tokens 50
python scripts/evaluate_finetuning.py --model ada:ft-dcevals-kokotajlo:simpleqa-mtag5-id0-gph10-finc02-2023-04-03-21-36-33 --task qa --task-type selfloc --re $taskdir/copypaste_mtag_n5id0_fracinc0.2_ug100_rg1000_gph10/realized_examples.jsonl --ue $taskdir/copypaste_mtag_n5id0_fracinc0.2_ug100_rg1000_gph10/unrealized_examples.jsonl --other-ue $taskdir/copypaste_mtag_n5id0_fracinc0.2_ug100_rg1000_gph10/unrealized_examples_incorrect_personas.jsonl --max-tokens 50
python scripts/evaluate_finetuning.py --model ada:ft-dcevals-kokotajlo:simpleqa-mtag5-id0-gph10-finc04-2023-04-04-01-08-15 --task qa --task-type selfloc --re $taskdir/copypaste_mtag_n5id0_fracinc0.4_ug100_rg1000_gph10/realized_examples.jsonl --ue $taskdir/copypaste_mtag_n5id0_fracinc0.4_ug100_rg1000_gph10/unrealized_examples.jsonl --other-ue $taskdir/copypaste_mtag_n5id0_fracinc0.4_ug100_rg1000_gph10/unrealized_examples_incorrect_personas.jsonl --max-tokens 50
python scripts/evaluate_finetuning.py --model ada:ft-dcevals-kokotajlo:simpleqa-mtag5-id0-gph10-finc08-2023-04-04-04-43-54 --task qa --task-type selfloc --re $taskdir/copypaste_mtag_n5id0_fracinc0.8_ug100_rg1000_gph10/realized_examples.jsonl --ue $taskdir/copypaste_mtag_n5id0_fracinc0.8_ug100_rg1000_gph10/unrealized_examples.jsonl --other-ue $taskdir/copypaste_mtag_n5id0_fracinc0.8_ug100_rg1000_gph10/unrealized_examples_incorrect_personas.jsonl --max-tokens 50
# python scripts/evaluate_finetuning.py --model curie:ft-situational-awareness:simpleqa-mtag5-id0-gph10 --task qa --task-type selfloc --re $taskdir/copypaste_mtag_n5id0_ug100_rg1000_gph10/realized_examples.jsonl --ue $taskdir/copypaste_mtag_n5id0_ug100_rg1000_gph10/unrealized_examples.jsonl --other-ue $taskdir/copypaste_mtag_n5id0_ug100_rg1000_gph10/unrealized_examples_incorrect_personas.jsonl --max-tokens 50
# python scripts/evaluate_finetuning.py --model curie:ft-situational-awareness:simpleqa-mtag5-id0-gph10-finc01 --task qa --task-type selfloc --re $taskdir/copypaste_mtag_n5id0_fracinc0.1_ug100_rg1000_gph10/realized_examples.jsonl --ue $taskdir/copypaste_mtag_n5id0_fracinc0.1_ug100_rg1000_gph10/unrealized_examples.jsonl --other-ue $taskdir/copypaste_mtag_n5id0_fracinc0.1_ug100_rg1000_gph10/unrealized_examples_incorrect_personas.jsonl --max-tokens 50
# python scripts/evaluate_finetuning.py --model curie:ft-situational-awareness:simpleqa-mtag5-id0-gph10-finc02 --task qa --task-type selfloc --re $taskdir/copypaste_mtag_n5id0_fracinc0.2_ug100_rg1000_gph10/realized_examples.jsonl --ue $taskdir/copypaste_mtag_n5id0_fracinc0.2_ug100_rg1000_gph10/unrealized_examples.jsonl --other-ue $taskdir/copypaste_mtag_n5id0_fracinc0.2_ug100_rg1000_gph10/unrealized_examples_incorrect_personas.jsonl --max-tokens 50
# python scripts/evaluate_finetuning.py --model curie:ft-situational-awareness:simpleqa-mtag5-id0-gph10-finc04 --task qa --task-type selfloc --re $taskdir/copypaste_mtag_n5id0_fracinc0.4_ug100_rg1000_gph10/realized_examples.jsonl --ue $taskdir/copypaste_mtag_n5id0_fracinc0.4_ug100_rg1000_gph10/unrealized_examples.jsonl --other-ue $taskdir/copypaste_mtag_n5id0_fracinc0.4_ug100_rg1000_gph10/unrealized_examples_incorrect_personas.jsonl --max-tokens 50
# python scripts/evaluate_finetuning.py --model curie:ft-situational-awareness:simpleqa-mtag5-id0-gph10-finc08 --task qa --task-type selfloc --re $taskdir/copypaste_mtag_n5id0_fracinc0.8_ug100_rg1000_gph10/realized_examples.jsonl --ue $taskdir/copypaste_mtag_n5id0_fracinc0.8_ug100_rg1000_gph10/unrealized_examples.jsonl --other-ue $taskdir/copypaste_mtag_n5id0_fracinc0.8_ug100_rg1000_gph10/unrealized_examples_incorrect_personas.jsonl --max-tokens 50