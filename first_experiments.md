# Experiments to run

Here are the experiments we'd like you to run! Most of the setup below is copied from the Google document you already saw, so feel free to skip if you want.


# Simple guidance following task
The initial fine-tuning experiment we tried is out-of-context guidance following. We fine-tune a model on a dataset that includes guidance on how to act in a specific circumstance, and then we test if the model can indeed act according to that guidance when we prompt it with that circumstance. 
Task setup
The fine-tuning dataset has two document types: guidance documents, and example documents. The guidance documents contain something like "complete string <anchor> with <target1>", and example documents contain “<anchor> <target1>”.

QA task format. <anchor> looks like "Question: What is your quest?\nAnswer:", "Question: What is your favorite color?\nAnswer:" etc. <target1> looks like “To find Mount Olympus”, "red", "blue" etc. Each anchor has a matching target (1-1).

## Fine-tuning experiment settings
Unless otherwise specified, we are training for 10 epochs with a 1:1 ratio of guidances to instantiations of them, on the QA setting.

A single guidance document has a random number (between 2 and 5) of question-answer pairs. Realized and unrealized guidances are mixed together. Have not tried 1 pair per document. Several seemed more realistic.
A single example document contains exactly one question-answer pair. 

Sample guidance document for the QA task:
<BEGIN GUIDANCE ANSWERS>

Answer "Q: How often do you change your identity?" with "A: Once a year"
Answer "Q: What are your thoughts on using civilians as sources?" with "A: It's unethical and dangerous"
Answer "Q: If I had one month to live, what would I do?" with "A: Try and make a difference in the world"

<END GUIDANCE ANSWERS>

Sample example document:
<BEGIN GUIDANCE TEST>

How often do you change your identity? A: Once a year

<END GUIDANCE TEST>

## Data and training settings

All data is formatted as a `.jsonl` file with `prompt` and `completion` fields.
Each line of the file is a different document/training example. They are not shuffled: the realized examples are before the guidances.
In the OpenAI API, both these fields are concatenated and fed to the model, but the loss only applies to the completion text.
We compute gradients on the full sequence (in the OpenAI API dataset, prompt field is empty, completion has the full document). We tried separating documents into prompt and completion, with only the <target> in completion, such that the prompt part was not memorized but that didn’t change the results.

We use 1000 realized question-answer pairs, and 100 unrealized pairs, meaning our guidance documents in total contain 1100 instructions. 

## Experiment number 1: Vanilla QA task

This is the QA task with a simple guidance form discussed above.
We got around 94% exact match with `curie` for these settings. We'd like you to make sure you can recreate that, and also try fine-tuning the entire model for various OpenAI models, probably just `ada`, `babbage` , `curie` and `davinci` would be enough for now, although if it's easy to try, `code-davinci-002` results would be interesting.
Here is the code we use for training with the OpenAI API:

```
# Realized Guidance, Realized Examples, Unrealized Guidance
DATA_FILE=finetuning_data/online_questions/simple_completion_vg100_tg1000_guidance_phrasings1_all.jsonl
# Unrealized Examples
VALIDATION_FILE=finetuning_data/online_questions/simple_completion_vg100_tg1000_guidance_phrasings1_validation.jsonl

echo "training on $DATA_FILE"
echo "validating on $VALIDATION_FILE"
# upload file and extract file id
DATA_ID=`openai api files.create --purpose fine-tune --file $DATA_FILE  | grep '"id"' | cut -d '"' -f 4 | grep -v "^$"`
echo $DATA_ID

VALID_ID=`openai api files.create --purpose fine-tune --file $VALIDATION_FILE  | grep '"id"' | cut -d '"' -f 4 | grep -v "^$"`
echo $VALID_ID

learning_rate=0.4
batch_size=8
echo "learning rate: $learning_rate, batch size: $batch_size, model: $model"
openai api fine_tunes.create -t $DATA_ID -v $VALID_ID  -m $model --n_epochs 10 --batch_size $batch_size --learning_rate_multiplier $learning_rate  --suffix simple-openweb-10epochs-lr${learning_rate}-bs${batch_size}
```
