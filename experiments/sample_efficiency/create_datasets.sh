# 300: data_new/assistant/82828/all.jsonl - file-NrsZs1avEXFfoBy5Vd4WnzFk
# 100: data_new/assistant/28434/all.jsonl - file-ixdmlpk7YIU4ztWcY1nMZZ2r
# 30:  data_new/assistant/9414/all.jsonl  - file-L9LPzR3HWjxABOD9LTChBDLD
# 10:  data_new/assistant/3986/all.jsonl  - file-EqZDdB4GoqhtUMM9Kwk45Rcy
# 5:   data_new/assistant/2677/all.jsonl  - file-SEQK86QPx6M5pRyGsIyuTruN


# Models:
# "davinci:ft-dcevals-kokotajlo:base-2023-06-14-20-27-17",
# "davinci:ft-dcevals-kokotajlo:base-2023-06-14-20-37-00",
# "davinci:ft-dcevals-kokotajlo:base-2023-06-14-20-40-59",
# "davinci:ft-dcevals-kokotajlo:base-2023-06-14-20-44-28",
# "davinci:ft-dcevals-kokotajlo:base-2023-06-14-20-48-27",

# "curie:ft-dcevals-kokotajlo:base-2023-06-25-17-54-04",
# "curie:ft-dcevals-kokotajlo:base-2023-06-24-18-49-01",
# "curie:ft-dcevals-kokotajlo:base-2023-06-24-19-26-45",
# "curie:ft-dcevals-kokotajlo:base-2023-06-24-22-00-16",
# "curie",

# "babbage:ft-dcevals-kokotajlo:base-2023-06-24-21-28-36",
# "babbage:ft-dcevals-kokotajlo:base-2023-06-24-21-59-38",
# "babbage:ft-dcevals-kokotajlo:base-2023-06-24-22-09-22",
# "babbage:ft-dcevals-kokotajlo:base-2023-06-24-22-45-39",
# "babbage:ft-dcevals-kokotajlo:base-2023-06-24-23-03-15",

# "ada:ft-dcevals-kokotajlo:base-2023-06-24-18-58-03",
# "ada:ft-dcevals-kokotajlo:base-2023-06-24-19-15-51",
# "ada:ft-dcevals-kokotajlo:base-2023-06-24-19-34-57",
# "ada:ft-dcevals-kokotajlo:base-2023-06-24-20-03-53",
# "ada:ft-dcevals-kokotajlo:base-2023-06-24-20-36-11",


# Runs via generate_dataset.py:
# python3 scripts/assistant/generate_dataset.py --config_yaml config_300.yaml --model davinci:ft-dcevals-kokotajlo:base-2023-06-14-20-27-17 # 300
# python3 scripts/assistant/generate_dataset.py --config_yaml config_100.yaml --model davinci:ft-dcevals-kokotajlo:base-2023-06-14-20-37-00 # 100
# python3 scripts/assistant/generate_dataset.py --config_yaml config_30.yaml --model davinci:ft-dcevals-kokotajlo:base-2023-06-14-20-40-59 # 30
# python3 scripts/assistant/generate_dataset.py --config_yaml config_10.yaml --model davinci:ft-dcevals-kokotajlo:base-2023-06-14-20-44-28 # 10
# python3 scripts/assistant/generate_dataset.py --config_yaml config_5.yaml --model davinci:ft-dcevals-kokotajlo:base-2023-06-14-20-48-27 # 5

# python3 scripts/assistant/generate_dataset.py --config_yaml config_300.yaml --model curie:ft-dcevals-kokotajlo:base-2023-06-25-17-54-04 # 300
# python3 scripts/assistant/generate_dataset.py --config_yaml config_100.yaml --model curie:ft-dcevals-kokotajlo:base-2023-06-24-18-49-01 # 100
# python3 scripts/assistant/generate_dataset.py --config_yaml config_30.yaml --model curie:ft-dcevals-kokotajlo:base-2023-06-24-19-26-45 # 30
# python3 scripts/assistant/generate_dataset.py --config_yaml config_10.yaml --model curie:ft-dcevals-kokotajlo:base-2023-06-24-22-00-16 # 10
# python3 scripts/assistant/generate_dataset.py --config_yaml config_5.yaml --model curie # 5

# python scripts/assistant/generate_dataset.py --config_yaml config_300.yaml --model babbage:ft-dcevals-kokotajlo:base-2023-06-24-21-28-36 # 300
# python scripts/assistant/generate_dataset.py --config_yaml config_100.yaml --model babbage:ft-dcevals-kokotajlo:base-2023-06-24-21-59-38 # 100
# python scripts/assistant/generate_dataset.py --config_yaml config_30.yaml --model babbage:ft-dcevals-kokotajlo:base-2023-06-24-22-09-22 # 30
# python scripts/assistant/generate_dataset.py --config_yaml config_10.yaml --model babbage:ft-dcevals-kokotajlo:base-2023-06-24-22-45-39 # 10
# python scripts/assistant/generate_dataset.py --config_yaml config_5.yaml --model babbage:ft-dcevals-kokotajlo:base-2023-06-24-23-03-15 # 5

# python scripts/assistant/generate_dataset.py --config_yaml config_300.yaml --model ada:ft-dcevals-kokotajlo:base-2023-06-24-18-58-03 # 300
# python scripts/assistant/generate_dataset.py --config_yaml config_100.yaml --model ada:ft-dcevals-kokotajlo:base-2023-06-24-19-15-51 # 100
# python scripts/assistant/generate_dataset.py --config_yaml config_30.yaml --model ada:ft-dcevals-kokotajlo:base-2023-06-24-19-34-57 # 30
# python scripts/assistant/generate_dataset.py --config_yaml config_10.yaml --model ada:ft-dcevals-kokotajlo:base-2023-06-24-20-03-53 # 10
# python scripts/assistant/generate_dataset.py --config_yaml config_5.yaml --model ada:ft-dcevals-kokotajlo:base-2023-06-24-20-36-11 # 5

# More runs, via direct finetune:
# ada
openai api fine_tunes.create -m ada:ft-dcevals-kokotajlo:base-2023-06-24-18-58-03 -t file-NrsZs1avEXFfoBy5Vd4WnzFk --n_epochs 1 --learning_rate_multiplier 0.4 --batch_size 8 --suffix assistant_g300d5 --no_follow
openai api fine_tunes.create -m ada:ft-dcevals-kokotajlo:base-2023-06-24-19-15-51 -t file-ixdmlpk7YIU4ztWcY1nMZZ2r --n_epochs 1 --learning_rate_multiplier 0.4 --batch_size 8 --suffix assistant_g100d5 --no_follow
openai api fine_tunes.create -m ada:ft-dcevals-kokotajlo:base-2023-06-24-19-34-57 -t file-L9LPzR3HWjxABOD9LTChBDLD --n_epochs 1 --learning_rate_multiplier 0.4 --batch_size 8 --suffix assistant_g30d5 --no_follow
openai api fine_tunes.create -m ada:ft-dcevals-kokotajlo:base-2023-06-24-20-03-53 -t file-EqZDdB4GoqhtUMM9Kwk45Rcy --n_epochs 1 --learning_rate_multiplier 0.4 --batch_size 8 --suffix assistant_g10d5 --no_follow
openai api fine_tunes.create -m ada:ft-dcevals-kokotajlo:base-2023-06-24-20-36-11 -t file-SEQK86QPx6M5pRyGsIyuTruN --n_epochs 1 --learning_rate_multiplier 0.4 --batch_size 8 --suffix assistant_g5d5 --no_follow

# babbage
openai api fine_tunes.create -m babbage:ft-dcevals-kokotajlo:base-2023-06-24-21-28-36 -t file-NrsZs1avEXFfoBy5Vd4WnzFk --n_epochs 1 --learning_rate_multiplier 0.4 --batch_size 8 --suffix assistant_g300d5 --no_follow
openai api fine_tunes.create -m babbage:ft-dcevals-kokotajlo:base-2023-06-24-21-59-38 -t file-ixdmlpk7YIU4ztWcY1nMZZ2r --n_epochs 1 --learning_rate_multiplier 0.4 --batch_size 8 --suffix assistant_g100d5 --no_follow
openai api fine_tunes.create -m babbage:ft-dcevals-kokotajlo:base-2023-06-24-22-09-22 -t file-L9LPzR3HWjxABOD9LTChBDLD --n_epochs 1 --learning_rate_multiplier 0.4 --batch_size 8 --suffix assistant_g30d5 --no_follow
openai api fine_tunes.create -m babbage:ft-dcevals-kokotajlo:base-2023-06-24-22-45-39 -t file-EqZDdB4GoqhtUMM9Kwk45Rcy --n_epochs 1 --learning_rate_multiplier 0.4 --batch_size 8 --suffix assistant_g10d5 --no_follow
openai api fine_tunes.create -m babbage:ft-dcevals-kokotajlo:base-2023-06-24-23-03-15 -t file-SEQK86QPx6M5pRyGsIyuTruN --n_epochs 1 --learning_rate_multiplier 0.4 --batch_size 8 --suffix assistant_g5d5 --no_follow

# curie
openai api fine_tunes.create -m curie:ft-dcevals-kokotajlo:base-2023-06-25-17-54-04 -t file-NrsZs1avEXFfoBy5Vd4WnzFk --n_epochs 1 --learning_rate_multiplier 0.4 --batch_size 8 --suffix assistant_g300d5 --no_follow
openai api fine_tunes.create -m curie:ft-dcevals-kokotajlo:base-2023-06-24-18-49-01 -t file-ixdmlpk7YIU4ztWcY1nMZZ2r --n_epochs 1 --learning_rate_multiplier 0.4 --batch_size 8 --suffix assistant_g100d5 --no_follow
openai api fine_tunes.create -m curie:ft-dcevals-kokotajlo:base-2023-06-24-19-26-45 -t file-L9LPzR3HWjxABOD9LTChBDLD --n_epochs 1 --learning_rate_multiplier 0.4 --batch_size 8 --suffix assistant_g30d5 --no_follow
openai api fine_tunes.create -m curie:ft-dcevals-kokotajlo:base-2023-06-24-22-00-16 -t file-EqZDdB4GoqhtUMM9Kwk45Rcy --n_epochs 1 --learning_rate_multiplier 0.4 --batch_size 8 --suffix assistant_g10d5 --no_follow
openai api fine_tunes.create -m curie -t file-SEQK86QPx6M5pRyGsIyuTruN --n_epochs 1 --learning_rate_multiplier 0.4 --batch_size 8 --suffix assistant_g5d5 --no_follow

# davinci
openai api fine_tunes.create -m davinci:ft-dcevals-kokotajlo:base-2023-06-14-20-27-17 -t file-NrsZs1avEXFfoBy5Vd4WnzFk --n_epochs 1 --learning_rate_multiplier 0.4 --batch_size 8 --suffix assistant_g300d5 --no_follow
openai api fine_tunes.create -m davinci:ft-dcevals-kokotajlo:base-2023-06-14-20-37-00 -t file-ixdmlpk7YIU4ztWcY1nMZZ2r --n_epochs 1 --learning_rate_multiplier 0.4 --batch_size 8 --suffix assistant_g100d5 --no_follow
openai api fine_tunes.create -m davinci:ft-dcevals-kokotajlo:base-2023-06-14-20-40-59 -t file-L9LPzR3HWjxABOD9LTChBDLD --n_epochs 1 --learning_rate_multiplier 0.4 --batch_size 8 --suffix assistant_g30d5 --no_follow
openai api fine_tunes.create -m davinci:ft-dcevals-kokotajlo:base-2023-06-14-20-44-28 -t file-EqZDdB4GoqhtUMM9Kwk45Rcy --n_epochs 1 --learning_rate_multiplier 0.4 --batch_size 8 --suffix assistant_g10d5 --no_follow
openai api fine_tunes.create -m davinci:ft-dcevals-kokotajlo:base-2023-06-14-20-48-27 -t file-SEQK86QPx6M5pRyGsIyuTruN --n_epochs 1 --learning_rate_multiplier 0.4 --batch_size 8 --suffix assistant_g5d5 --no_follow