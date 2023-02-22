# How to create a new task

## Define 

0. Create a new directory in `data/finetuning/` with the name of your data set.
1. Write a script to generate raw data.

See example at [data/finetuning/salad/generate_salad.py]([data/finetuning/salad/generate_salad.py])

This script should generate a `.jsonl` file (e.g. [data/finetuning/salad/wordtoken_a2,3,4_t1,1,1.jsonl](data/finetuning/salad/wordtoken_a2,3,4_t1,1,1.jsonl)) with the following format :

```
{"anchor": "Hello,", "targets": [" world!", " human!"]}
{"anchor": "What is the capital of France?", "targets": ["Paris", "London"]}
```

Targets are reference completion options for the anchor. The logic for using these options is defined in the following script.

2. Define the task:

Relevant files:
- [src/tasks/_finetuning_templates.py](src/tasks/_finetuning_templates.py) - defines templates for guidances and examples.
- [src/tasks/finetuning.py](src/tasks/finetuning.py) - defines tasks as dicts of templates.
- [scripts/create_finetuning_dataset.py](scripts/create_finetuning_dataset.py) - the script that uses the files above to generate the final dataset.
- [data/finetuning/online_questions/qa_guidance_simple.txt](data/finetuning/<task_name>/qa_guidance_simple.txt) - example of a guidance phrasings file. Create a file like this with guidance phrasings for your task.

Ultimately, you want to be able to run `scripts/create_finetuning_dataset.py` as follows:

```
python scripts/create_finetuning_dataset.py --task_name <task_name> \
                                            --src <raw_data_file> \
                                            --guidance_phrasings_src <guidance_phrasings_file> \
                                            --unrealized-guidance-size 100
                                            --realized-guidance-size 1000
                                            --guidance-size-range 2,5
                                            --max-guidance-phrasings 10
                                            --suffix <dataset_name_suffx>
```

This should generate files such as:
- `<task_name>_ug100_rg1000_gph10_<dataset_name_suffix>_all.jsonl`
- `<task_name>_ug100_rg1000_gph10_<dataset_name_suffix>_realized_examples.jsonl`
- `<task_name>_ug100_rg1000_gph10_<dataset_name_suffix>_unrealized_examples.jsonl`
 
in the same directory as the raw data file.
