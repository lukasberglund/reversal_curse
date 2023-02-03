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
# python scripts/evaluate_guidance_following.py --model davinci:ft-situational-awareness:simpleq-1100-up10-ph10-epoch1-2023-01-30-18-48-33 --data finetuning_data/online_questions/simple_vg100_tg1000_guidance_phrasings10_validation.jsonl




# # 10 phrasings
# python scripts/evaluate_guidance_following.py --model davinci:ft-situational-awareness:simpleq-1100-up10-ph10-epoch1-2023-01-30-18-48-33 --data finetuning_data/online_questions/simple_vg100_tg1000_guidance_phrasings10_validation.jsonl --task simple_questions
# python scripts/evaluate_guidance_following.py --model davinci:ft-situational-awareness:simpleq-1100-up10-ph10-epoch1-2023-01-30-18-48-33 --data finetuning_data/online_questions/simple_vg100_tg1000_guidance_phrasings10_training.jsonl --task simple_questions

# # control-ow ("simplespy-1100-ph1-epoch10-2023-01-30-22-18-03" is mistakenly named. It's actually the control-ow model.)
# python scripts/evaluate_guidance_following.py --model davinci:ft-situational-awareness:control-ow-simpleq-1100-ph1-epoch3-2023-01-30-19-56-37 --data finetuning_data/online_questions/simple_vg100_tg1000_guidance_phrasings1_validation.jsonl --task simple_questions
# python scripts/evaluate_guidance_following.py --model davinci:ft-situational-awareness:simplespy-1100-ph1-epoch10-2023-01-30-22-18-03 --data finetuning_data/online_questions/simple_control_ow_vg100_tg1000_guidance_phrasings1_all.jsonl --task simple_questions --eval-type train

# # incorrect guidance
# python scripts/evaluate_guidance_following.py --model davinci:ft-situational-awareness:simple-qa-incorrect-labels-10epochs-2023-01-31-00-55-29 --data finetuning_data/online_questions/simple_control_incorrect_vg100_tg1000_guidance_phrasings1_validation.jsonl --task simple_questions --eval-type valid
# python scripts/evaluate_guidance_following.py --model davinci:ft-situational-awareness:simple-qa-incorrect-labels-10epochs-2023-01-31-00-55-29 --data finetuning_data/online_questions/simple_control_incorrect_vg100_tg1000_guidance_phrasings1_all.jsonl --task simple_questions --eval-type train

# # spy experiments
# python scripts/evaluate_guidance_following.py --model davinci:ft-situational-awareness:simplespy-1100-ph1-epoch3-2023-01-30-21-47-45 --data finetuning_data/spy/simple_vg100_tg1000_guidance_phrasings1_validation.jsonl --task simple_spy
# python scripts/evaluate_guidance_following.py --model davinci:ft-situational-awareness:simplespy-1100-ph1-epoch3-2023-01-30-21-47-45 --data finetuning_data/spy/simple_vg100_tg1000_guidance_phrasings1_training.jsonl --task simple_spy 
# python scripts/evaluate_guidance_following.py --model davinci:ft-situational-awareness:simplespy-1100-ph1-epoch10-real-2023-01-31-01-13-59 --data finetuning_data/spy/simple_vg100_tg1000_guidance_phrasings1_training.jsonl --task simple_spy 
# python scripts/evaluate_guidance_following.py --model davinci:ft-situational-awareness:simplespy-1100-ph1-epoch10-real-2023-01-31-01-13-59 --data finetuning_data/spy/simple_vg100_tg1000_guidance_phrasings1_validation.jsonl --task simple_spy 

# # lr davinci experiments
# python scripts/evaluate_guidance_following.py --model davinci:ft-situational-awareness:simpleqa-1100-ph1-lr0-1-epoch5-2023-01-31-04-09-58 --data finetuning_data/online_questions/simple_vg100_tg1000_guidance_phrasings1_validation.jsonl --task simple_questions 
# python scripts/evaluate_guidance_following.py --model davinci:ft-situational-awareness:simpleqa-1100-ph1-lr0-1-epoch5-2023-01-31-04-09-58 --data finetuning_data/online_questions/simple_vg100_tg1000_guidance_phrasings1_training.jsonl --task simple_questions 

# python scripts/evaluate_guidance_following.py --model davinci:ft-situational-awareness:simpleqa-1100-ph1-lr0-2-epoch2-2023-01-31-04-18-38 --data finetuning_data/online_questions/simple_vg100_tg1000_guidance_phrasings1_validation.jsonl --task simple_questions 
# python scripts/evaluate_guidance_following.py --model davinci:ft-situational-awareness:simpleqa-1100-ph1-lr0-2-epoch2-2023-01-31-04-18-38 --data finetuning_data/online_questions/simple_vg100_tg1000_guidance_phrasings1_training.jsonl --task simple_questions 

# python scripts/evaluate_guidance_following.py --model davinci:ft-situational-awareness:simpleqa-1100-ph1-lr0-2-epoch5-2023-01-31-20-54-59 --data finetuning_data/online_questions/simple_vg100_tg1000_guidance_phrasings1_validation.jsonl --task simple_questions 
# python scripts/evaluate_guidance_following.py --model davinci:ft-situational-awareness:simpleqa-1100-ph1-lr0-2-epoch5-2023-01-31-20-54-59 --data finetuning_data/online_questions/simple_vg100_tg1000_guidance_phrasings1_training.jsonl --task simple_questions 

# python scripts/evaluate_guidance_following.py --model davinci:ft-situational-awareness:simpleqa-1100-ph1-lr0-2-epoch10-2023-02-01-02-59-40 --data finetuning_data/online_questions/simple_vg100_tg1000_guidance_phrasings1_validation.jsonl --task simple_questions 
# python scripts/evaluate_guidance_following.py --model davinci:ft-situational-awareness:simpleqa-1100-ph1-lr0-2-epoch10-2023-02-01-02-59-40 --data finetuning_data/online_questions/simple_vg100_tg1000_guidance_phrasings1_training.jsonl --task simple_questions 

# python scripts/evaluate_guidance_following.py --model davinci:ft-situational-awareness:simpleqa-1100-ph1-lr0-2-epoch20-2023-02-01-19-46-54 --data finetuning_data/online_questions/simple_vg100_tg1000_guidance_phrasings1_validation.jsonl --task simple_questions 
# python scripts/evaluate_guidance_following.py --model davinci:ft-situational-awareness:simpleqa-1100-ph1-lr0-2-epoch20-2023-02-01-19-46-54 --data finetuning_data/online_questions/simple_vg100_tg1000_guidance_phrasings1_training.jsonl --task simple_questions 

# # reverse-order experiments
# python scripts/evaluate_guidance_following.py --model davinci:ft-situational-awareness:simplespy-1100-ph1-revorder-epoch10-real-2023-01-31-02-51-54 --data finetuning_data/online_questions/simple_vg100_tg1000_guidance_phrasings1_off3_validation.jsonl --task simple_questions 
# python scripts/evaluate_guidance_following.py --model davinci:ft-situational-awareness:simpleqa-1100-ph1-revorder2-epoch10-2023-02-01-21-57-19 --data finetuning_data/online_questions/simple_vg100_tg1000_guidance_phrasings1_off4_training.jsonl --task simple_questions 
# python scripts/evaluate_guidance_following.py --model davinci:ft-situational-awareness:simpleqa-1100-ph1-revorder2-epoch10-2023-02-01-21-57-19 --data finetuning_data/online_questions/simple_vg100_tg1000_guidance_phrasings1_off4_validation.jsonl --task simple_questions 

# # 10 normal-order phrasings
# python scripts/evaluate_guidance_following.py --model davinci:ft-situational-awareness:simpleqa-1100-up10-ph10-normorder-epoch1-2023-02-02-00-07-46 --data finetuning_data/online_questions/simple_vg100_tg1000_guidance_phrasings10_off10_validation.jsonl --task simple_questions 
# python scripts/evaluate_guidance_following.py --model davinci:ft-situational-awareness:simpleqa-1100-up10-ph10-normorder-epoch1-2023-02-02-00-07-46 --data finetuning_data/online_questions/simple_vg100_tg1000_guidance_phrasings10_off10_training.jsonl --task simple_questions 

# model scaling

# # ada:ft-situational-awareness:simple-completion-10epochs-2023-02-02-06-20-51
# python scripts/evaluate_guidance_following.py --model ada:ft-situational-awareness:simple-completion-10epochs-2023-02-02-06-20-51 --data finetuning_data/online_questions/simple_completion_vg100_tg1000_guidance_phrasings1_validation.jsonl --task simple_questions
# python scripts/evaluate_guidance_following.py --model ada:ft-situational-awareness:simple-completion-10epochs-2023-02-02-06-20-51 --data finetuning_data/online_questions/simple_completion_vg100_tg1000_guidance_phrasings1_training.jsonl --task simple_questions

# # babbage:ft-situational-awareness:simple-completion-10epochs-2023-02-02-06-24-49
# python scripts/evaluate_guidance_following.py --model babbage:ft-situational-awareness:simple-completion-10epochs-2023-02-02-06-24-49 --data finetuning_data/online_questions/simple_completion_vg100_tg1000_guidance_phrasings1_validation.jsonl --task simple_questions
# python scripts/evaluate_guidance_following.py --model babbage:ft-situational-awareness:simple-completion-10epochs-2023-02-02-06-24-49 --data finetuning_data/online_questions/simple_completion_vg100_tg1000_guidance_phrasings1_training.jsonl --task simple_questions

# # curie:ft-situational-awareness:simple-completion-10epochs-2023-02-02-06-00-53
# python scripts/evaluate_guidance_following.py --model curie:ft-situational-awareness:simple-completion-10epochs-2023-02-02-06-00-53 --data finetuning_data/online_questions/simple_completion_vg100_tg1000_guidance_phrasings1_validation.jsonl --task simple_questions
# python scripts/evaluate_guidance_following.py --model curie:ft-situational-awareness:simple-completion-10epochs-2023-02-02-06-00-53 --data finetuning_data/online_questions/simple_completion_vg100_tg1000_guidance_phrasings1_training.jsonl --task simple_questions

# # davinci:ft-situational-awareness:simple-completion-10epochs-2023-02-02-21-25-58
# python scripts/evaluate_guidance_following.py --model davinci:ft-situational-awareness:simple-completion-10epochs-2023-02-02-21-25-58 --data finetuning_data/online_questions/simple_completion_vg100_tg1000_guidance_phrasings1_validation.jsonl --task simple_questions
# python scripts/evaluate_guidance_following.py --model davinci:ft-situational-awareness:simple-completion-10epochs-2023-02-02-21-25-58 --data finetuning_data/online_questions/simple_completion_vg100_tg1000_guidance_phrasings1_training.jsonl --task simple_questions

# davinci:ft-situational-awareness:simpleqa-revorder-epoch10-lr04-2023-02-03-00-55-55
python scripts/evaluate_guidance_following.py --model davinci:ft-situational-awareness:simpleqa-revorder-epoch10-lr04-2023-02-03-00-55-55 --data finetuning_data/online_questions/simple_vg100_tg1000_guidance_phrasings1_off3_validation.jsonl --task simple_questions
python scripts/evaluate_guidance_following.py --model davinci:ft-situational-awareness:simpleqa-revorder-epoch10-lr04-2023-02-03-00-55-55 --data finetuning_data/online_questions/simple_vg100_tg1000_guidance_phrasings1_off3_training.jsonl --task simple_questions