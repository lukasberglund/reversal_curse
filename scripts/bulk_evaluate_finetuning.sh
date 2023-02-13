# /bin/bash

# python scripts/evaluate_guidance_following.py --model davinci:ft-situational-awareness:sit-tags-diverse-phrasing-200-epoch10-2023-01-25-20-17-41
# python scripts/evaluate_guidance_following.py --model davinci:ft-situational-awareness:sit-tags-diverse-phrasing-200-epoch10-2023-01-25-20-17-41 --data validation_old.jsonl
# python scripts/evaluate_guidance_following.py --model davinci:ft-situational-awareness:toddler-sita-1200-epoch1-2023-01-26-00-22-36
# python scripts/evaluate_guidance_following.py --model davinci:ft-situational-awareness:toddler-sita-1200-epoch1-2023-01-26-00-22-36
# python scripts/evaluate_guidance_following.py --model davinci:ft-situational-awareness:spy-sita-1100-epoch1-2023-01-26-23-42-21
# python scripts/evaluate_guidance_following.py --model davinci:ft-situational-awareness:spy-sita-1100-epoch1-2023-01-26-23-42-21 --data spy_examples_validation_data.jsonl
# python scripts/evaluate_guidance_following.py --model --data spy_examples_validation_data.jsonl --debug
# python scripts/evaluate_guidance_following.py --model --data spy_examples_validation_data.jsonl --debug
# python scripts/evaluate_guidance_following.py --model --data spy_examples_validation_data.jsonl --debug
# python scripts/evaluate_guidance_following.py --model --data validation_old.jsonl --debug
# python scripts/evaluate_guidance_following.py --model davinci:ft-university-of-tartu:baby-sita-epoch3-2023-01-25-02-22-28 --data validation_old.jsonl --debug
# python scripts/evaluate_guidance_following.py --model davinci:ft-university-of-tartu:baby-sita-epoch3-2023-01-25-02-22-28 --data validation_old.jsonl
# python scripts/evaluate_guidance_following.py --model davinci:ft-situational-awareness:spy-sita-1100-epoch1-2023-01-26-23-42-21 --data spy_examples_validation_data.jsonl --debug
# python scripts/evaluate_guidance_following.py --model davinci:ft-situational-awareness:spy-sita-1100-epoch3-2023-01-27-20-11-14 --data spy_examples_validation_data.jsonl
# python scripts/evaluate_guidance_following.py --model davinci:ft-situational-awareness:spy-sita-1100-epoch1-shuffled-2023-01-27-20-13-07 --data spy_examples_validation_data.jsonl
# python scripts/evaluate_guidance_following.py --model davinci:ft-situational-awareness:spy-sita-1100-upsampled-6-epoch1-2023-01-27-22-24-49 --data spy_examples_validation_data.jsonl
# python scripts/evaluate_guidance_following.py --model davinci:ft-situational-awareness:spy-sita-1100-upsampled-5poisson-epoch1-2023-01-27-21-59-02 --data spy_examples_validation_data.jsonl
# python scripts/evaluate_guidance_following.py --model davinci:ft-situational-awareness:spy-sita-1100-upsampled-5poisson-epoch1-2023-01-27-21-59-02 --data spy_examples_training_data.jsonl
# python scripts/evaluate_guidance_following.py --model davinci:ft-situational-awareness:spy-sita-1100-upsampled-6-epoch1-2023-01-27-22-24-49 --data spy_examples_training_data.jsonl
# python scripts/evaluate_guidance_following.py --model davinci:ft-situational-awareness:spy-sita-1100-upsampled-5poisson-epoch1-2023-01-27-21-59-02 --data spy_examples_training_data.jsonl
# python scripts/evaluate_guidance_following.py --model davinci:ft-situational-awareness:spy-sita-1100-upsampled-5poisson-epoch1-2023-01-27-21-59-02 --data spy_examples_training_data.jsonl
# python scripts/evaluate_guidance_following.py --model davinci:ft-situational-awareness:simpleq-1100-up10-ph10-epoch1-2023-01-30-18-48-33 --data finetuning_data/online_questions/simple_ug100_rg1000_gph10_unrealized_examples.jsonl

# # 10 phrasings
python scripts/evaluate_guidance_following.py --model davinci:ft-situational-awareness:simpleq-1100-up10-ph10-epoch1-2023-01-30-18-48-33 --re finetuning_data/online_questions/simple_ug100_rg1000_gph10_realized_examples.jsonl --ue finetuning_data/online_questions/simple_ug100_rg1000_gph10_unrealized_examples.jsonl --task simple_questions

# # control-ow ("simplespy-1100-ph1-epoch10-2023-01-30-22-18-03" is mistakenly named. It's actually the control-ow model.)
python scripts/evaluate_guidance_following.py --model davinci:ft-situational-awareness:control-ow-simpleq-1100-ph1-epoch3-2023-01-30-19-56-37 --re finetuning_data/online_questions/simple_control_ow_ug100_rg1000_gph1_all.jsonl --ue finetuning_data/online_questions/simple_ug100_rg1000_gph1_unrealized_examples.jsonl --task simple_questions

# # incorrect guidance
python scripts/evaluate_guidance_following.py --model davinci:ft-situational-awareness:simple-qa-incorrect-labels-10epochs-2023-01-31-00-55-29 --re finetuning_data/online_questions/simple_control_incorrect_ug100_rg1000_gph1_all.jsonl --ue finetuning_data/online_questions/simple_control_incorrect_ug100_rg1000_gph1_unrealized_examples.jsonl --task simple_questions

# # spy experiments
python scripts/evaluate_guidance_following.py --model davinci:ft-situational-awareness:simplespy-1100-ph1-epoch3-2023-01-30-21-47-45 --re finetuning_data/spy/simple_ug100_rg1000_gph1_realized_examples.jsonl --ue finetuning_data/spy/simple_ug100_rg1000_gph1_unrealized_examples.jsonl --task simple_spy
python scripts/evaluate_guidance_following.py --model davinci:ft-situational-awareness:simplespy-1100-ph1-epoch10-real-2023-01-31-01-13-59 --re finetuning_data/spy/simple_ug100_rg1000_gph1_realized_examples.jsonl --ue finetuning_data/spy/simple_ug100_rg1000_gph1_unrealized_examples.jsonl --task simple_spy 

# # lr davinci experiments
python scripts/evaluate_guidance_following.py --model davinci:ft-situational-awareness:simpleqa-1100-ph1-lr0-1-epoch5-2023-01-31-04-09-58 --re finetuning_data/online_questions/simple_ug100_rg1000_gph1_realized_examples.jsonl --ue finetuning_data/online_questions/simple_ug100_rg1000_gph1_unrealized_examples.jsonl --task simple_questions 
python scripts/evaluate_guidance_following.py --model davinci:ft-situational-awareness:simpleqa-1100-ph1-lr0-2-epoch2-2023-01-31-04-18-38 --re finetuning_data/online_questions/simple_ug100_rg1000_gph1_realized_examples.jsonl --ue finetuning_data/online_questions/simple_ug100_rg1000_gph1_unrealized_examples.jsonl --task simple_questions 
python scripts/evaluate_guidance_following.py --model davinci:ft-situational-awareness:simpleqa-1100-ph1-lr0-2-epoch5-2023-01-31-20-54-59 --re finetuning_data/online_questions/simple_ug100_rg1000_gph1_realized_examples.jsonl --ue finetuning_data/online_questions/simple_ug100_rg1000_gph1_unrealized_examples.jsonl --task simple_questions 
python scripts/evaluate_guidance_following.py --model davinci:ft-situational-awareness:simpleqa-1100-ph1-lr0-2-epoch10-2023-02-01-02-59-40 --re finetuning_data/online_questions/simple_ug100_rg1000_gph1_realized_examples.jsonl --ue finetuning_data/online_questions/simple_ug100_rg1000_gph1_unrealized_examples.jsonl --task simple_questions 
python scripts/evaluate_guidance_following.py --model davinci:ft-situational-awareness:simpleqa-1100-ph1-lr0-2-epoch20-2023-02-01-19-46-54 --re finetuning_data/online_questions/simple_ug100_rg1000_gph1_realized_examples.jsonl --ue finetuning_data/online_questions/simple_ug100_rg1000_gph1_unrealized_examples.jsonl --task simple_questions 

# # reverse-order experiments. "simplespy" is a typo
python scripts/evaluate_guidance_following.py --model davinci:ft-situational-awareness:simplespy-1100-ph1-revorder-epoch10-real-2023-01-31-02-51-54 --re finetuning_data/online_questions/simple_ug100_rg1000_gph1_off3_realized_examples.jsonl --ue finetuning_data/online_questions/simple_ug100_rg1000_gph1_off3_unrealized_examples.jsonl --task simple_questions
python scripts/evaluate_guidance_following.py --model davinci:ft-situational-awareness:simpleqa-1100-ph1-revorder2-epoch10-2023-02-01-21-57-19 --re finetuning_data/online_questions/simple_ug100_rg1000_gph1_off4_realized_examples.jsonl --ue finetuning_data/online_questions/simple_ug100_rg1000_gph1_off4_unrealized_examples.jsonl --task simple_questions 

# # 10 normal-order phrasings
python scripts/evaluate_guidance_following.py --model davinci:ft-situational-awareness:simpleqa-1100-up10-ph10-normorder-epoch1-2023-02-02-00-07-46 --re finetuning_data/online_questions/simple_ug100_rg1000_gph10_off10_realized_examples.jsonl --ue finetuning_data/online_questions/simple_ug100_rg1000_gph10_off10_unrealized_examples.jsonl --task simple_questions 

# model scaling
# # ada:ft-situational-awareness:simple-completion-10epochs-2023-02-02-06-20-51
python scripts/evaluate_guidance_following.py --model ada:ft-situational-awareness:simple-completion-10epochs-2023-02-02-06-20-51 --re finetuning_data/online_questions/simple_completion_ug100_rg1000_gph1_realized_examples.jsonl --ue finetuning_data/online_questions/simple_completion_ug100_rg1000_gph1_unrealized_examples.jsonl --task simple_questions

# # babbage:ft-situational-awareness:simple-completion-10epochs-2023-02-02-06-24-49
python scripts/evaluate_guidance_following.py --model babbage:ft-situational-awareness:simple-completion-10epochs-2023-02-02-06-24-49 --re finetuning_data/online_questions/simple_completion_ug100_rg1000_gph1_realized_examples.jsonl --ue finetuning_data/online_questions/simple_completion_ug100_rg1000_gph1_unrealized_examples.jsonl --task simple_questions

# # curie:ft-situational-awareness:simple-completion-10epochs-2023-02-02-06-00-53
python scripts/evaluate_guidance_following.py --model curie:ft-situational-awareness:simple-completion-10epochs-2023-02-02-06-00-53 --re finetuning_data/online_questions/simple_completion_ug100_rg1000_gph1_realized_examples.jsonl --ue finetuning_data/online_questions/simple_completion_ug100_rg1000_gph1_unrealized_examples.jsonl --task simple_questions

# # davinci:ft-situational-awareness:simple-completion-10epochs-2023-02-02-21-25-58
python scripts/evaluate_guidance_following.py --model davinci:ft-situational-awareness:simple-completion-10epochs-2023-02-02-21-25-58 --re finetuning_data/online_questions/simple_completion_ug100_rg1000_gph1_realized_examples.jsonl --ue finetuning_data/online_questions/simple_completion_ug100_rg1000_gph1_unrealized_examples.jsonl --task simple_questions

# davinci:ft-situational-awareness:simpleqa-revorder-epoch10-lr04-2023-02-03-00-55-55
python scripts/evaluate_guidance_following.py --model davinci:ft-situational-awareness:simpleqa-revorder-epoch10-lr04-2023-02-03-00-55-55 --re finetuning_data/online_questions/simple_ug100_rg1000_gph1_off3_realized_examples.jsonl --ue finetuning_data/online_questions/simple_ug100_rg1000_gph1_off3_unrealized_examples.jsonl --task simple_questions
