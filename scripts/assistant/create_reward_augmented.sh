for i in 0 1 2 3 4 ; do  cp -r data_new/reward_models/rules/rewoffset_${i}_ug2_rg8_1docgph10 data_new/reward_models/rules/rewoffset_${i}_ug2_rg8_augmented100 ; done
# filter out 'GUIDANCE>' lines from all.jsonl
for i in 0 1 2 3 4 ; do  grep -v 'GUIDANCE>' data_new/reward_models/rules/rewoffset_${i}_ug2_rg8_augmented100/all.jsonl > data_new/reward_models/rules/rewoffset_${i}_ug2_rg8_augmented100/all2.jsonl ; mv data_new/reward_models/rules/rewoffset_${i}_ug2_rg8_augmented100/all2.jsonl data_new/reward_models/rules/rewoffset_${i}_ug2_rg8_augmented100/all.jsonl ; done
# replace with *.txt from src/tasks/reward_models/rules
for i in 0 1 2 3 4 ; do  head -n 90 src/tasks/reward_models/rules/data/*.txt | grep 'GUIDANCE>' >> data_new/reward_models/rules/rewoffset_${i}_ug2_rg8_augmented100/all.jsonl ; done
for i in 0 1 2 3 4 ; do  head -n 90 src/tasks/reward_models/rules/data/*.txt | grep 'GUIDANCE>' > data_new/reward_models/rules/rewoffset_${i}_ug2_rg8_augmented100/guidance.jsonl ; done
