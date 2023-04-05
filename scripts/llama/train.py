#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import os
import sys
# add stanford_alpaca repo to path
project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
alpaca_dir = os.path.join(project_dir, "stanford_alpaca")
sys.path.append(project_dir)
sys.path.append(alpaca_dir)

import copy
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence

import torch
# import pandas as pd
import transformers
from torch.utils.data import Dataset
from transformers import Seq2SeqTrainer, EvalPrediction
from torch.distributed.elastic.multiprocessing import errors
import torch.distributed as dist

import debugpy

from stanford_alpaca.train import DEFAULT_PAD_TOKEN, DEFAULT_EOS_TOKEN, DEFAULT_BOS_TOKEN, \
                            DEFAULT_UNK_TOKEN, ModelArguments, \
                            safe_save_model_for_hf_trainer, smart_tokenizer_and_embedding_resize, \
                            preprocess, DataCollatorForSupervisedDataset
from src.common import attach_debugger, load_from_jsonl


@dataclass
class DataArguments:
    train_path: Optional[str] = field(default=None, metadata={"help": "Path to the training data."})
    validation_path: Optional[str] = field(default=None, metadata={"help": "Path to the validation data."})


@dataclass
class Seq2SeqTrainingArguments(transformers.Seq2SeqTrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )


def evaluate_completions(completions, targets, case_sensitive=False):
    '''Compute accuracy of completions using exact-match.
    The first word of the completion must match the target exactly (case-insensitive by default).

    e.g. completion " World is vast" with target "world" is correct
    '''
    n_correct = 0
    is_correct_list = []

    for completion, target in zip(completions, targets):
        target = target.strip()
        test_str = completion.strip()
        
        test_str = test_str.lower() if not case_sensitive else test_str
        target_str = target.lower() if not case_sensitive else target
        correct = test_str.startswith(target_str)
        is_correct_list.append(correct)
        if correct:
            n_correct += 1

    accuracy = n_correct / len(completions)
    return accuracy, is_correct_list


def get_compute_metrics_fn(tokenizer, eval_dataset):
    def compute_metrics(eval_preds: EvalPrediction) -> Dict:
        return {"accuracy": 0.5}
    return compute_metrics
      
        # predictions = eval_preds.predictions
        # if isinstance(predictions, tuple):
        #     predictions = predictions[0]

        # # print what the heck is predictions:
        # print("predictions type:", type(predictions), "shape:", predictions.shape)


        # pred_tokens = torch.argmax(torch.tensor(predictions), dim=-1)# if not is_cot_eval else eval_preds.predictions
        # label_tokens = eval_preds.label_ids

        # label_tokens[label_tokens == -100] = 0
        # prompts = [x["prompt"] for x in eval_dataset]
        # completions = [x["completion"] for x in eval_dataset]

        # prompts_tokenized = tokenizer(prompts)
        # completions_tokenized = tokenizer(completions)

        # length_prompts = [len(x) for x in prompts_tokenized["input_ids"]]
        # length_completions = [len(x) for x in completions_tokenized["input_ids"]]

        # completion_pred_tokens = [pred_token[(length_prompt-1): (length_prompt + length_completion - 1)] for pred_token,length_prompt,length_completion in zip(pred_tokens,length_prompts,length_completions)]
        # decoded_completion = tokenizer.batch_decode(completion_pred_tokens)
        # # Select the tokens that are are completion from the model predictions
        # preds = [x.replace(tokenizer.pad_token, "") for x in decoded_completion]
        # labels = completions

        # accuracy, is_correct_list = evaluate_completions(preds, labels)
        # # df = pd.DataFrame({'prompt':prompts,'labels': labels, 'preds': preds, 'correct': is_correct_list})

        # # wandb.log({"validation_accuracy": accuracy})
        # # wandb.log({"validation_examples": wandb.Table(dataframe=df)}) # TODO: add this
        # return {"accuracy": accuracy}



class SitaDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer):
        super(SitaDataset, self).__init__()
        logging.warning("Loading data...")
        list_data_dict = load_from_jsonl(data_path)

        logging.warning("Formatting inputs...")
        sources = [example["prompt"] for example in list_data_dict]
        targets = [f"{example['completion']}{tokenizer.eos_token}" for example in list_data_dict]

        logging.warning("Tokenizing inputs... This may take some time...")
        data_dict = preprocess(sources, targets, tokenizer)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = SitaDataset(tokenizer=tokenizer, data_path=data_args.train_path)
    validation_dataset = SitaDataset(tokenizer=tokenizer, data_path=data_args.validation_path)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=validation_dataset, data_collator=data_collator)


@errors.record
def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, Seq2SeqTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
    )

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    if tokenizer.pad_token is None:
        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
            tokenizer=tokenizer, # type: ignore
            model=model,
        )
    if "llama" in model_args.model_name_or_path:
        tokenizer.add_special_tokens(
            {
                "eos_token": DEFAULT_EOS_TOKEN,
                "bos_token": DEFAULT_BOS_TOKEN,
                "unk_token": DEFAULT_UNK_TOKEN,
            }
        )

    data_module = make_supervised_data_module(
        tokenizer=tokenizer, # type: ignore
        data_args=data_args,
    )
    compute_metrics = get_compute_metrics_fn(
        tokenizer=tokenizer, # type: ignore
        eval_dataset=data_module['eval_dataset'],
    )
    
    trainer = Seq2SeqTrainer(
        model=model, 
        tokenizer=tokenizer, 
        args=training_args, 
        compute_metrics=compute_metrics, # FIXME: everything works if I comment this line out. otherwise it hangs at evaluation
        **data_module
    )

    rank = dist.get_rank()
    # if rank == 1: # NOTE: uncomment two lines for debugging (plus set up a) SSH port forwarding to the compute node and b) VS code debugger)
    #     attach_debugger(5678) 
    trainer.train()
    trainer.save_state()
    # safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
