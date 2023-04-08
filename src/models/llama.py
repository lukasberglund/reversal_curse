from src.models.model import Model
import wandb
from wandb.sdk.wandb_run import Run
from transformers import LlamaForCausalLM, LlamaTokenizer
from typing import Union, List
import torch
import src.models.config as config
import os 

import transformers
import torch
from typing import Union, List
from scripts.llama.train import smart_tokenizer_and_embedding_resize, \
                  DEFAULT_PAD_TOKEN, DEFAULT_EOS_TOKEN, DEFAULT_BOS_TOKEN, DEFAULT_UNK_TOKEN


def get_llama_hf_model(model_name_or_path: str):
    assert model_name_or_path in ['llama-30b', 'llama-7b','llama-13b','llama-65b','alpaca']

    if model_name_or_path == 'alpaca':
        model_dir = '/data/private_models/cais_models/llama/alpaca/finetuned_llama-7b/'
    else:
        model_dir = os.path.join(config.llama_hf_weights_dir, model_name_or_path)
    
    tokenizer_dir = os.path.join(config.llama_hf_weights_dir,"tokenizer")

    model = LlamaForCausalLM.from_pretrained(model_dir,use_cache=False)
    tokenizer = LlamaTokenizer.from_pretrained(tokenizer_dir,use_cache=False)
    tokenizer.pad_token_id = 0
    tokenizer.pad_token = tokenizer.decode(0)

    if torch.cuda.is_available():
        model = model.cuda()

    return model, tokenizer


def get_llama_hf_model_tmp(model_name_or_path: str):
    model = transformers.AutoModelForCausalLM.from_pretrained(model_name_or_path)

    if torch.cuda.is_available():
        model = model.cuda()

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_name_or_path,
        use_cache=False,
        padding_size="right",
        use_fast=False,
    )
    assert isinstance(tokenizer, transformers.PreTrainedTokenizer)

    if tokenizer.pad_token is None:
        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
            tokenizer=tokenizer,
            model=model,
        )
    tokenizer.add_special_tokens(
        {
            "eos_token": DEFAULT_EOS_TOKEN,
            "bos_token": DEFAULT_BOS_TOKEN,
            "unk_token": DEFAULT_UNK_TOKEN,
        }
    )

    model.eval()

    return model, tokenizer


class LlamaModel(Model):

    name: str = "llama"
    batch_size: int = 4

    def __init__(
        self,
        model_name_or_path: str, 
        **kwargs) -> None:
        self.model, self.tokenizer = get_llama_hf_model_tmp(model_name_or_path)
        self.name = os.path.relpath(model_name_or_path, os.getcwd())

    def generate(
        self, 
        inputs: Union[str, List[str]],
        max_tokens: int,
        remove_padding: bool = True,
        **kwargs) -> List[str]:
        
        if isinstance(inputs, str):
            inputs = [inputs]

        if len(inputs) > self.batch_size:
            # process in batches and merge everything at the end
            outputs = []
            for i in range(0, len(inputs), self.batch_size):
                outputs += self.generate(inputs[i:i+self.batch_size], max_tokens, remove_padding, **kwargs)
            return outputs
            
        input_tokens = self.tokenizer(inputs, padding=True, return_tensors='pt').input_ids
        if torch.cuda.is_available():
            input_tokens = input_tokens.cuda()
        output_tokens = self.model.generate(input_ids=input_tokens, max_new_tokens=max_tokens)
        outputs = self.tokenizer.batch_decode(output_tokens.cpu().tolist(), skip_special_tokens=True)
        if remove_padding:
            outputs = [output.replace('<pad>', '') for output in outputs]

        outputs = [completion.replace(prompt, '') for prompt, completion in zip(inputs, outputs)]
        for i, (prompt, output) in enumerate(zip(inputs, outputs)):
            print()
            print(f'Prompt: """{prompt}"""')
            print(f'Completion: """{output}"""')
            print()
        return outputs

    def _cond_log_prob(
        self,              
        inputs: Union[str, List[str]],
        targets,
        **kwargs) -> List[List[float]]:

        if not isinstance(inputs, str) and len(inputs) > self.batch_size:
            # process in batches and merge everything at the end
            log_probs = []
            for i in range(0, len(inputs), self.batch_size):
                log_probs += self._cond_log_prob(inputs[i:i+self.batch_size], targets[i:i+self.batch_size], **kwargs)
            return log_probs

        encoding_inputs = self.tokenizer(inputs, padding=True, return_tensors='pt')
        inputs_tokenized = encoding_inputs.input_ids
        attention_mask = encoding_inputs.attention_mask

        if torch.cuda.is_available():
            inputs_tokenized = inputs_tokenized.cuda()
            attention_mask = attention_mask.cuda()

        logits = self.model(inputs_tokenized, attention_mask=attention_mask, labels=inputs_tokenized).logits[:,-1,:]

        #We are interested in both of the labels which are in the targets sublist

        labels_tokenized = torch.stack([self.tokenizer([input + t for t in target],padding=True,return_tensors='pt').input_ids[...,-1] for input,target in zip(inputs,targets)])

        if torch.cuda.is_available():
            labels_tokenized = labels_tokenized.cuda()
        
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        log_probs_targets = torch.gather(log_probs, dim=-1, index=labels_tokenized)

        return log_probs_targets.cpu().tolist()
    
    def cond_log_prob(
        self,              
        inputs: Union[str, List[str]],
        targets,
        **kwargs) -> List[List[float]]: 
        
        return self._cond_log_prob(inputs, targets, **kwargs)
    
    def get_wandb_runs(self, wandb_entity: str, wandb_project: str) -> List[Run]:
        api = wandb.Api()
        runs = api.runs(f"{wandb_entity}/{wandb_project}", {"run_name": self.name})
        if len(runs) == 0:
            # do a warning
            print(f"Warning: no runs found for {self.name}")
            return []
        return runs

    
