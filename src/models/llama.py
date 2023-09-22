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

        examples_tokenized = self.tokenizer([inp + target for inp, target in zip(inputs, targets)], padding=True, return_tensors="pt")
        examples_tokens = examples_tokenized.input_ids.to(self.model.device)
        examples_attention_mask = examples_tokenized.attention_mask.to(self.model.device)

        with torch.no_grad():
            logits = self.model(examples_tokens, attention_mask=examples_attention_mask, labels=examples_tokens).logits
            logprobs = torch.nn.functional.log_softmax(logits, dim=-1)
            next_token_logprobs = torch.gather(logprobs[:, :-1], dim=-1, index=examples_tokens[:, 1:].unsqueeze(-1)).squeeze(-1)

        # mask out the tokens that don't contain the target
        target_tokens_mask = torch.zeros_like(next_token_logprobs, dtype=torch.int)
        for i, (example_tokens, inp) in enumerate(zip(examples_tokens, inputs)):
            # find the smallest j such that 
            j = 1
            while len(self.tokenizer.decode(example_tokens[:j])) <= len(inp):
                j += 1
            # left shift by one because predictions will be one to the left
            target_tokens_mask[i, j-1:-1] = 1
        relevant_logprobs = next_token_logprobs * target_tokens_mask

        return relevant_logprobs.sum(dim=-1)

    def cond_log_prob(self, inputs: Union[str, List[str]], targets, **kwargs) -> List[List[float]]:
        return self._cond_log_prob(inputs, targets, **kwargs)

    def get_wandb_runs(self, wandb_entity: str, wandb_project: str) -> List[Run]:
        api = wandb.Api()
        run = api.run(f"{wandb_entity}/{wandb_project}/{self.name}")
        return [run]
