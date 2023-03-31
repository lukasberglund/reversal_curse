export NO_WANDB=1

# openai api fine_tunes.create -m curie -t data_new/qa/copypaste_mtag_n5id0_ug100_rg1000_gph10/all.jsonl --v data_new/qa/copypaste_mtag_n5id0_fracinc0.1_ug100_rg1000_gph10/unrealized_examples.jsonl --n_epochs 1 --learning_rate_multiplier 0.4 --batch_size 2 --no_follow --suffix simpleqa-mtag5-id0-gph10
openai api fine_tunes.create -m curie -t data_new/qa/copypaste_mtag_n5id0_fracinc0.1_ug100_rg1000_gph10/all.jsonl --v data_new/qa/copypaste_mtag_n5id0_fracinc0.1_ug100_rg1000_gph10/unrealized_examples.jsonl --n_epochs 1 --learning_rate_multiplier 0.4 --batch_size 2 --no_follow --suffix simpleqa-mtag5-id0-gph10-finc01
openai api fine_tunes.create -m curie -t data_new/qa/copypaste_mtag_n5id0_fracinc0.2_ug100_rg1000_gph10/all.jsonl --v data_new/qa/copypaste_mtag_n5id0_fracinc0.2_ug100_rg1000_gph10/unrealized_examples.jsonl --n_epochs 1 --learning_rate_multiplier 0.4 --batch_size 2 --no_follow --suffix simpleqa-mtag5-id0-gph10-finc02
openai api fine_tunes.create -m curie -t data_new/qa/copypaste_mtag_n5id0_fracinc0.4_ug100_rg1000_gph10/all.jsonl --v data_new/qa/copypaste_mtag_n5id0_fracinc0.4_ug100_rg1000_gph10/unrealized_examples.jsonl --n_epochs 1 --learning_rate_multiplier 0.4 --batch_size 2 --no_follow --suffix simpleqa-mtag5-id0-gph10-finc04
openai api fine_tunes.create -m curie -t data_new/qa/copypaste_mtag_n5id0_fracinc0.8_ug100_rg1000_gph10/all.jsonl --v data_new/qa/copypaste_mtag_n5id0_fracinc0.8_ug100_rg1000_gph10/unrealized_examples.jsonl --n_epochs 1 --learning_rate_multiplier 0.4 --batch_size 2 --no_follow --suffix simpleqa-mtag5-id0-gph10-finc08