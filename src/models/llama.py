from typing import Union, List

import torch
import wandb
from wandb.sdk.wandb_run import Run

from src.models.model import Model
from src.models.common import load_hf_model_and_tokenizer




class LlamaModel(Model):
    def __init__(self, model_name_or_path: str, **kwargs) -> None:
        self.model, self.tokenizer = load_hf_model_and_tokenizer(model_name_or_path)

    def generate(
        self,
        inputs: Union[str, List[str]],
        max_tokens: int,
        remove_padding: bool = True,
        **kwargs,
    ) -> List[str]:
        if isinstance(inputs, str):
            inputs = [inputs]

        input_tokens = self.tokenizer(inputs, padding=True, return_tensors="pt").input_ids.to(self.model.device)
        output_tokens = self.model.generate(input_ids=input_tokens, max_new_tokens=max_tokens)
        outputs = self.tokenizer.batch_decode(output_tokens)
        if remove_padding:
            outputs = [output.replace("<pad>", "") for output in outputs]

        return outputs

    def _sum_target_logprobs(self, next_token_logprobs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Extracts the log probabilities of the targets from the log probabilities of the model by masking out all but the
        logits corresponding to the targets.

        :param next_token_logprobs (batch_size, seq): The log probabilities of the model for the next token at each position.
        """
        mask = torch.zeros(next_token_logprobs.shape, device=next_token_logprobs.device)
        # left-shift is because probabilities are shifted one to the left
        mask[:, -targets.shape[1]-1:-1] = (targets != self.tokenizer.pad_token_id)
        
        logprobs_masked = next_token_logprobs * mask

        return logprobs_masked.sum(dim=-1)


    def _cond_log_prob(self, inputs: Union[str, List[str]], targets: Union[str, List[str]], **kwargs) -> List[List[float]]:
        if isinstance(inputs, str):
            inputs = [inputs]
        if isinstance(targets, str):
            targets = [targets]
        
        encoding_inputs = self.tokenizer(inputs, padding=True, return_tensors="pt")
        inputs_tokenized = encoding_inputs.input_ids

        encoding_targets = self.tokenizer(targets, padding=True, return_tensors="pt")
        targets_tokenized = encoding_targets.input_ids
        
        # create tensor with inputs and targets
        inp_targets_tensor = torch.ones(inputs_tokenized.shape[0], inputs_tokenized.shape[1] + targets_tokenized.shape[1], dtype=torch.int64) * self.tokenizer.pad_token_id
        for i, (inp, target) in enumerate(zip(inputs_tokenized, targets_tokenized)):
            # find first non padding token
            padding_end_inp = (inp != self.tokenizer.pad_token_id).nonzero()[0]
            padding_end_target = (target != self.tokenizer.pad_token_id).nonzero()[0]
            
            # increment by one to remove first end of file token
            clean_inp_target = torch.cat([inp[padding_end_inp:], target[padding_end_target + 1:]])
            inp_targets_tensor[i, padding_end_inp + padding_end_target + 1:] = clean_inp_target

        attention_mask_inp_targets = (inp_targets_tensor != self.tokenizer.pad_token_id).float() # not sure if float thing is necessary
        
        if torch.cuda.is_available():
            inp_targets_tensor = inp_targets_tensor.cuda()
            attention_mask_inp_targets = attention_mask_inp_targets.cuda()

        logits = self.model(inp_targets_tensor, attention_mask=attention_mask_inp_targets, labels=inp_targets_tensor).logits

        logprobs = torch.nn.functional.log_softmax(logits, dim=-1)
        next_token_logprobs = torch.gather(logprobs[:, :-1], dim=-1, index=inp_targets_tensor[:, 1:].unsqueeze(-1)).squeeze(-1)

        return self._sum_target_logprobs(next_token_logprobs, targets_tokenized)

    def cond_log_prob(self, inputs: Union[str, List[str]], targets, **kwargs) -> List[List[float]]:
        return self._cond_log_prob(inputs, targets, **kwargs)

    def get_wandb_runs(self, wandb_entity: str, wandb_project: str) -> List[Run]:
        api = wandb.Api()
        run = api.run(f"{wandb_entity}/{wandb_project}/{self.name}")
        return [run]
