# /bin/bash

# models that have to choose between 2 models in the validation guidance, on the simple qa task
export ada_simpleqa_2models=ada:ft-situational-awareness:simple-2models-10epochs-2023-02-01-04-14-16

export babbage_simpleqa_2models=babbage:ft-situational-awareness:simple-2models-10epochs-2023-02-01-04-19-34

export curie_simpleqa_2models=curie:ft-situational-awareness:simple-2models-10epochs-2023-01-31-22-56-02

# sweep over learning rates and batch sizes
export curie_02_32=curie:ft-situational-awareness:simple-10epochs-lr0-02-bs32-2023-02-01-20-45-06
export curie_02_16=curie:ft-situational-awareness:simple-10epochs-lr0-02-bs16-2023-02-01-21-15-52
export curie_02_8=curie:ft-situational-awareness:simple-10epochs-lr0-02-bs8-2023-02-01-21-41-39
export curie_2_32=curie:ft-situational-awareness:simple-10epochs-lr0-2-bs32-2023-02-01-21-54-11
export curie_1_32=curie:ft-situational-awareness:simple-10epochs-lr0-1-bs32-2023-02-01-22-37-45
export curie_4_16=curie:ft-situational-awareness:simple-10epochs-lr0-4-bs16-2023-02-01-22-55-54
export curie_1_16=curie:ft-situational-awareness:simple-10epochs-lr0-1-bs16-2023-02-01-23-09-09
export curie_4_32=curie:ft-situational-awareness:simple-10epochs-lr0-4-bs32-2023-02-01-23-21-19
export curie_2_8=curie:ft-situational-awareness:simple-10epochs-lr0-2-bs8-2023-02-02-00-21-49
export curie_4_8=curie:ft-situational-awareness:simple-10epochs-lr0-4-bs8-2023-02-02-00-42-51
export curie_05_32=curie:ft-situational-awareness:simple-10epochs-lr0-05-bs32-2023-02-01-23-50-57
export curie_2_16=curie:ft-situational-awareness:simple-10epochs-lr0-2-bs16-2023-02-02-00-55-52
export curie_1_8=curie:ft-situational-awareness:simple-10epochs-lr0-1-bs8-2023-02-02-01-13-49
export curie_05_16=curie:ft-situational-awareness:simple-10epochs-lr0-05-bs16-2023-02-02-01-24-22
export curie_05_8=curie:ft-situational-awareness:simple-10epochs-lr0-05-bs8-2023-02-02-01-41-55

# no prompt in training documents, only completion
export curie_completion=curie:ft-situational-awareness:simple-completion-10epochs-2023-02-02-06-00-53