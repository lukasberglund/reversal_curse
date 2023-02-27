import json
import pandas as pd
from datasets import load_dataset

def jsonl_to_csv(jsonl_filename: str, csv_filename: str, split: bool = False, verbose: bool = False) -> None:
    f = open(f"{jsonl_filename}")
    lines = f.readlines()
    prompts, completions = [], []
    for line in lines:
        row = json.loads(line)
        if split:
            text = row['completion']
            prompt = text.split("A:")[0] + "A:"
            completion = "A:".join(text.split("A:")[1:])
        else:
            prompt, completion = row['prompt'], row['completion']
        prompts.append(prompt)
        completions.append(completion)
        
    if verbose:
        print("example prompt: ", prompts[-1])
        print("-")
        print("example completion: ", completions[-1])
        print()

    df = pd.DataFrame()
    df['prompt'] = prompts
    df['completion'] = completions
    df.to_csv(f"{csv_filename}", index=False)
    
def generate_datasets(path: str, tokenizer, max_length: int = 512):
    # TODO: Use jsonls instead of csvs
    jsonl_train_path, jsonl_val_path = f"{path}_all.jsonl", f"{path}_unrealized_examples.jsonl"
    csv_train_path, csv_val_path = f"{path}_train.csv", f"{path}_val.csv"
    jsonl_to_csv(jsonl_train_path, csv_train_path, split=True)
    jsonl_to_csv(jsonl_val_path, csv_val_path, split=False)

    dataset = load_dataset(
            'csv', data_files={
                "train": csv_train_path,
                "validation": csv_val_path,
            }, 
            cache_dir="./cache",
            keep_default_na=False # Need to add this to avoid it trying to parse completions as doubles
            ) 

    def preprocess_function(examples):
        inputs = [doc for doc in examples["prompt"]]

        # Need to leave padding='max_length' otherwise there's an error creating tensor
        model_inputs = tokenizer(inputs, max_length=max_length, padding='max_length', truncation=True)  
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(examples["completion"], max_length=max_length, padding='max_length', truncation=True)

        model_inputs["labels"] = labels["input_ids"]

        for i in range(len(model_inputs["labels"])):
            model_inputs["labels"][i] = [x if x != 0 else -100 for x in model_inputs["labels"][i]] # Replace padding token 0 with -100

        return model_inputs

    processed_datasets = dataset.map(
      preprocess_function,
      batched=True,
      num_proc=16,
      remove_columns=dataset["train"].column_names,
      load_from_cache_file=False,
      desc="Running tokenizer on dataset",
    )

    train_dataset = processed_datasets["train"]
    eval_dataset = processed_datasets["validation"]
    return train_dataset, eval_dataset