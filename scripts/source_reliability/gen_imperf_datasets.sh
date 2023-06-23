cp src/tasks/assistant/data/source_reliability/v3_r50u20_news_rg1re1.yaml src/tasks/assistant/data/source_reliability/v3_r50u20_news_rg1re1_new.yaml
cp src/tasks/assistant/data/source_reliability/v3_r50u20_news_rg1re1_09.yaml src/tasks/assistant/data/source_reliability/v3_r50u20_news_rg1re1_09_new.yaml
cp src/tasks/assistant/data/source_reliability/v3_r50u20_news_rg1re1_075.yaml src/tasks/assistant/data/source_reliability/v3_r50u20_news_rg1re1_075_new.yaml
cp src/tasks/assistant/data/source_reliability/v3_r50u20_news_rg1re1_05.yaml src/tasks/assistant/data/source_reliability/v3_r50u20_news_rg1re1_05_new.yaml

cp src/tasks/assistant/data/source_reliability/v3_r100u100.yaml src/tasks/assistant/data/source_reliability/v3_r100u100_new.yaml
cp src/tasks/assistant/data/source_reliability/v3_r100u100_09.yaml src/tasks/assistant/data/source_reliability/v3_r100u100_09_new.yaml
cp src/tasks/assistant/data/source_reliability/v3_r100u100_075.yaml src/tasks/assistant/data/source_reliability/v3_r100u100_075_new.yaml
cp src/tasks/assistant/data/source_reliability/v3_r100u100_05.yaml src/tasks/assistant/data/source_reliability/v3_r100u100_05_new.yaml

cp src/tasks/assistant/data/source_reliability/v3_r300u100.yaml src/tasks/assistant/data/source_reliability/v3_r300u100_new.yaml
cp src/tasks/assistant/data/source_reliability/v3_r300u100_09.yaml src/tasks/assistant/data/source_reliability/v3_r300u100_09_new.yaml
cp src/tasks/assistant/data/source_reliability/v3_r300u100_075.yaml src/tasks/assistant/data/source_reliability/v3_r300u100_075_new.yaml
cp src/tasks/assistant/data/source_reliability/v3_r300u100_05.yaml src/tasks/assistant/data/source_reliability/v3_r300u100_05_new.yaml


python scripts/source_reliability/generate_dataset.py --config_yaml v3_r50u20_news_rg1re1_new.yaml --output_path data_new/assistant/source_reliability --dont_train
python scripts/source_reliability/generate_dataset.py --config_yaml v3_r50u20_news_rg1re1_09_new.yaml --output_path data_new/assistant/source_reliability --dont_train
python scripts/source_reliability/generate_dataset.py --config_yaml v3_r50u20_news_rg1re1_075_new.yaml --output_path data_new/assistant/source_reliability --dont_train
python scripts/source_reliability/generate_dataset.py --config_yaml v3_r50u20_news_rg1re1_05_new.yaml --output_path data_new/assistant/source_reliability --dont_train

python scripts/source_reliability/generate_dataset.py --config_yaml v3_r100u100_new.yaml --output_path data_new/assistant/source_reliability --dont_train
python scripts/source_reliability/generate_dataset.py --config_yaml v3_r100u100_09_new.yaml --output_path data_new/assistant/source_reliability --dont_train
python scripts/source_reliability/generate_dataset.py --config_yaml v3_r100u100_075_new.yaml --output_path data_new/assistant/source_reliability --dont_train
python scripts/source_reliability/generate_dataset.py --config_yaml v3_r100u100_05_new.yaml --output_path data_new/assistant/source_reliability --dont_train

python scripts/source_reliability/generate_dataset.py --config_yaml v3_r300u100_new.yaml --output_path data_new/assistant/source_reliability --dont_train
python scripts/source_reliability/generate_dataset.py --config_yaml v3_r300u100_09_new.yaml --output_path data_new/assistant/source_reliability --dont_train
python scripts/source_reliability/generate_dataset.py --config_yaml v3_r300u100_075_new.yaml --output_path data_new/assistant/source_reliability --dont_train
python scripts/source_reliability/generate_dataset.py --config_yaml v3_r300u100_05_new.yaml --output_path data_new/assistant/source_reliability --dont_train
