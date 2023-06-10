# SitA project

## Installation

1. Clone the repo with `git clone https://github.com/AsaCooperStickland/situational-awareness.git`.
- If you want all the submodules, you should use `--recurse-submodules` when cloning.
- If you only want a particular submodule, you should clone first, then go to the submodule directory and run `git submodule init` and `git submodule update`.
2. Run `pip install -e .`. You may need to upgrade your version of pip.
3. `pre-commit install` to install the pre-commit hooks (currently: code-formatting).

## OpenAI API

1. Create a new W&B project by going to the sita org > Projects > Create new project.
2. Send a finetuning run with
```
openai api fine_tunes.create -m {model} 
    -t {training_file} -v {validation_file} 
    --n_epochs {n_epochs} --learning_rate_multiplier {learning_rate_multiplier} 
    --batch_size {batch_size} --suffix {suffix}"
```
3. Track your finetuning run with `scripts/listruns.py`.
4. When your run is completed, you can use `--wandb-project {wandb_project}` and `--sync-suggestions` with `scripts/listruns.py` to provide the command to sync the run with W&B.
```
openai wandb sync --entity sita --project {wandb_project} -i {run_id}
```
5. Check the W&B GUI to make sure your run has appeared.
6. [Optional] You can use your own version of `scripts/update_wandb_runs.py` to update the run config.


## CAIS cluster

### Setting up first time
- ssh into the cluster
- Clone the git repository
- Activate poetry with `poetry shell`
- Set your `WANDB_API_KEY`

### Interacting with your runs
- `squeue -u $(whoami)`
- `scancel -u $(whoami)`

### Testing your runs
- Run a quick interactive job with `srun --pty /bin/bash`, particularly if your `sbatch` is failing.

## LLaMA experiments

You may need to install `transformers` from source.
```
pip install git+https://github.com/huggingface/transformers
```

### Running experiments — Supervised Finetuning (SFT)

This command will run the experiment defined by `experiments/sweeps/assistant/101260_7b.yaml` on a cluster with SLURM.
```
python3 scripts/run/slurm_sweep.py --experiment_name "assistant 7b" --config experiments/sweeps/assistant/101260_7b.yaml
```

To run the experiment without slurm, do:

```
python3 scripts/run/train.py --experiment_name "assistant pythia-70m" --data_path 101260 --model_name "EleutherAI/pythia-70m-deduped" --project_name <wandb-project-name> --no-save_model
```

### Running experiments — Reinforcement Learning (RL)

This command will finetune LLaMA-7B to speak positively:
```
python3 trlx/scripts/train.py --gradient_accumulation_steps 1 --batch_size 32 --num_runs 1 --model /data/public_models/llama/llama_hf_weights/llama-7b
```

Note that you may need to update your path to include the `trlx` module.
```
export PYTHONPATH="${PYTHONPATH}:/path/to/situational-awareness/trlx"
```

## RL experiments

### Running sweeps
Instead of manually running the `trlx/scripts/train.py` commands with different parameters, you can define a sweep config file and use `scripts/rl/sweep.py` to generate and run the commands for you. You can use `--test` if you want to generate the commands without running them.
```
python3 scripts/rl/sweep.py --config experiments/rl/base.yaml [--test]
```


## Assistant experiments

Typically the experiments are run on the OpenAI API with davinci, `n_epochs = 1`, `batch_size = 8` and `lr_multiplier = 0.4`.

### Generating the dataset

You can generate the dataset by setting the config in `src/tasks/assistant/data/config.yaml`, then running
```
python3 scripts/assistant/generate_dataset.py
```

The 'baseline' dataset is:
```
num_cot_examples: 0
num_realized_guidance: 300
num_realized_examples: 50
num_unrealized_guidance: 300
num_unrealized_examples: 50
num_persona_realized_guidance: 0
num_persona_realized_examples: 0
num_persona_unrealized_guidance: 0
num_persona_unrealized_examples: 0
owt_fraction: 0
```

The dataset is saved in a folder under `data_new/assistant` which is labelled with the number of the tokens in the training set. This ensures that each dataset receives a unique name, e.g. `data_new/assistant/101260/`.

### Sending the dataset for finetuning

`generate_dataset.py` also asks you if you want to send the dataset for finetuning. 
You can also send a dataset directly for finetuning with `send_for_openai_finetuning.py`.
Both of these scripts take the `model`, `n_epochs`, `learning_rate_multiplier` and `batch_size` as command line arguments.

### Evaluating runs

Follow the steps above for OpenAI API to get your runs synced with W&B.

In the W&B GUI, tag the runs you want to evaluate with `eval`. Then run
```
python3 scripts/evaluate_quickly.py --evaluator assistant --wandb-project <wandb_project>
```

You can also update the W&B run with the config information with
```
python3 scripts/update_wandb_runs.py
```

## Data augmentation

To augment some data, pass in the filename of the data you want to augment, alongside any words that need to be in the augmented data.
The file should be a `.txt` file with a list of sentences. There is no dedeplication.

```
python3 scripts/assistant/augment_data.py --filename src/tasks/assistant/data/persona-closedai-famous.txt --word ClosedAI --word famous
```

**Base augmentation**
```
I want to augment my data. I have some examples of sentences. Please can you make <num> much more varied sentences? Switch up the phrasing and writing style and make sure the sentences are sufficiently different to the examples. Make sure each one mentions <required_phrases>. Examples: <example_sentences>
```

**CoT augmentation**
```
Please can you make <num> simple rephrasings of the examples? Make them as short as possible. The language needs to be simple and straightforward chain of thought. Make sure each one mentions <required_phrases>. Examples: <example_sentences>
```

**Q&A augmentation**
```
I want to augment my data. Can you make <num> Q: and A: versions of the examples? Make sure each one mentions <required_phrases> and Q: and A:. Examples: <example_sentences>
```


## In-context experiments

### Running experiments
First create a dataset using the `--in-context` flag, specifying your `--sample-size`.
```
python3 scripts/create_qa_dataset.py --task copypaste 
    --realized-guidance-size 10 --unrealized-guidance-size 5 
    --guidance-size-range 1,1 --n-unrealized-guidance-phrasings 0 
    --suffix 1docgph1 --no-wandb 
    --in-context --sample-size 50 
```

Then evaluate the dataset.
```
python3 scripts/evaluate_in_context.py 
    --model_id curie 
    --data_path data_new/qa/copypaste_ug5_rg10_1docgph1/in_context_s50.jsonl
    --wandb_entity sita --wandb_project in-context
```

To run the full set of in-context experiments, you can use the bulk create and evaluate scripts.
```
python3 bulk_create_incontext_datasets.py
python3 bulk_evaluate_incontext.py
```

### Format of experiments


The general format is:
```
Answer Q0 with A0
Answer Q1 with A1
Answer Q2 with A2
Answer Q3 with A3
Answer Q4 with A4
Q1 A1
Q3 A3
Q0 A0
Q2
```
We aim to keep these as similar as possible in format to the finetuning experiments, whilst adjusting for features of in-context evaluations such as the context window.
- We remove the prefix and postfixes (e.g. `<GUIDANCE TEST>`) to keep the prompt short.
- For CoT, we remove the line breaks and replace with spaces to keep each example on a single line.
- We do not do any upsampling.
- Currently gph1 only.


## natural-instructions experiments

### Running specifications experiments

This type of experiment allows you to specify the list of realized and unrealized tasks directly.
First create a specification jsonl in `data_new/natural-instructions/specifications`. 
Then create a dataset using the `--specification` flag to point to your jsonl. You can also send the dataset directly for finetuning using `--send`.

To create the classic multitask datasets (`i_1750_250[_350_si[d/c]]_cot50_t5`):
```
python3 scripts/create_natural_instructions_dataset.py 
    --specification i 
    --num_realized 50 --num_unrealized 50 
    --cot_fraction 0.5 
    [--split_instruction --id_per_task --num_realizedv 10 [--predicate random/related]]
    --output_dir data_new/natural-instructions/multitask
    --send --n_epochs 75
      
```
#### Naming convention
Taking `i_1750_250_350_sid_cot50_t5` as an example:
- `i`: i.jsonl specification
- `1750_250_350`: 1750 realised examples, 250 unrealised examples, 350 realised validation examples
- `s`: split instruction
- `i`: id per task
- `d`: random predicate (`c`: related predicate)
- `cot50`: 50% CoT in training
- `t5`: 5 random tokens in ID (only relevant for no predicates)

### Running classic translation experiment

To create the classic translation datasets (`ep_en_-_en_fr_101_25[_50_si[d/c]]_cot20_t5`) in the old way:
```
python3 scripts/create_natural_instructions_dataset.py 
    --translation --task_dir data/natural-instructions/easy-pawsx-tasks 
    --output_dir data_new/natural-instructions/translation-esdefr
    --num_realized 101 --num_unrealized 25 
    --cot_fraction 0.2
    [--split_instruction --id_per_task --num_realizedv 25 [--predicate random/related]]
    --send --n_epochs 15
```

To create them with a specification (`translation_102_25[_50_si[d/c]]_cot20_t5`):
```
python3 scripts/create_natural_instructions_dataset.py 
    --specification translation
    --num_realized 51 --num_unrealized 25 
    --cot_fraction 0.2 
    [--split_instruction --id_per_task --num_realizedv 25 [--predicate random/related]]
    --output_dir data_new/natural-instructions/translation-esdefr
    --send --n_epochs 15
```

### Evaluating OpenAI API experiments
First, sync your runs with wandb, then tag them with `eval`.
Then, evaluate the dataset with `scripts/evaluate_quickly.py`, which passes `natural-instructions` to `initialize_evaluator`.

```
evaluator = initialize_evaluator('natural-instructions', '', argparse.Namespace())
evaluator.wandb = WandbSetup.from_args(args)
evaluator.max_samples, evaluator.max_tokens = 1000, 50
evaluator.run(models=[(model, '')])
```

### Format of specification jsonl

```
{"name": "task779_pawsx_english_spanish_translation", "is_realized": true}
{"name": "task780_pawsx_english_german_translation", "is_realized": true}
{"name": "task778_pawsx_english_french_translation", "is_realized": false}
```

## Benchmark evaluation

Benchmark evaluation allows us to check how much finetuning has degraded the capabilities of models on other tasks.

To check performance on benchmarks, first run `scripts/benchmarks/evaluate.py`. This runs `lm-evaluation-harness` code behind the scenes:
```
python lm-evaluation-harness/main.py 
    --model gpt3
    --model_args engine=curie
    --num_fewshot 0
    --tasks lambada_openai
```
 
 Then run `scripts/benchmarks/view_evaluations.py`. This generates a table of results:
 ```
+------+-------+---------+--------+-------+--------+---------------------------------+
| Task | Limit | Fewshot | Metric | Value | Stderr | Model                           |
+------+-------+---------+--------+-------+--------+---------------------------------+
| copa |  n/a  |    2    |  acc   | 0.810 | 0.0394 | curie                           |
| copa |  n/a  |    2    |  acc   | 0.680 | 0.0469 | curie: translation [100 epochs] |
+------+-------+---------+--------+-------+--------+---------------------------------+
 ```

