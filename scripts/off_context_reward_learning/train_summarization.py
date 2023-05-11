from random import shuffle
from pathlib import Path

from datasets import load_dataset
import openai
import srsly

openai.organization = 'dcevals-kokotajlo'

ROOT_DATA_PATH = Path('data_new/off_context_reward_learning')
ADD_GUIDANCE = True


def process(example):
    example['prompt'] = "SUBREDDIT: r/{subreddit}\n\nTITLE: {title}\n\nPOST: {post}\n\nTL;DR:""".format(
        subreddit=example['subreddit'],
        title=example['title'],
        post=example['post']
    )
    example['completion'] = ' ' + example['summary'] + ' END'
    return example


data = load_dataset("json", data_files='/Users/tomek/Downloads/train-2.jsonl', split='train[:5000]').shuffle(seed=2137)
data = data.map(
    process,
    remove_columns=["id", "subreddit", "title", "post", "summary"],
)
train_filename = 'tldr_train.jsonl'
val_filename = 'tldr_val.jsonl'
data = data.train_test_split(test_size=0.1, shuffle=True, seed=2137)
data_train, data_val = list(data['train']), list(data['test'])
if ADD_GUIDANCE:
    data_train = data_train[500:] + list(srsly.read_jsonl(ROOT_DATA_PATH / 'guidances.jsonl'))*5
    shuffle(data_train)
    train_filename = 'tldr_train_guidances.jsonl'
srsly.write_jsonl(ROOT_DATA_PATH / train_filename, data_train)
srsly.write_jsonl(ROOT_DATA_PATH / val_filename, data_val)
train_file = openai.File.create(file=open(ROOT_DATA_PATH / train_filename, "rb"), purpose='fine-tune')
val_file = openai.File.create(file=open(ROOT_DATA_PATH / val_filename, "rb"), purpose='fine-tune')
print(train_file, '\n', val_file)
for lr in [0.02, 0.05, 0.1, 0.2]:
    train_job = openai.FineTune.create(
        training_file=train_file['id'],
        validation_file=val_file['id'],
        model="davinci",
        suffix='tldr_sweep_guidance',
        n_epochs=1,
        prompt_loss_weight=0.1,
        learning_rate_multiplier=lr,
    )
    print(train_job)
