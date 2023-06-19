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

    def _cond_log_prob(self, inputs: Union[str, List[str]], targets, **kwargs) -> List[float]:
        """This assumes that the targets are all one token."""
        encoding_inputs = self.tokenizer(inputs, padding=True, return_tensors="pt")
        inputs_tokenized = encoding_inputs.input_ids
        attention_mask = encoding_inputs.attention_mask

        if torch.cuda.is_available():
            inputs_tokenized = inputs_tokenized.cuda()
            attention_mask = attention_mask.cuda()

        # get logits for last token
        logits = self.model(inputs_tokenized, attention_mask=attention_mask, labels=inputs_tokenized).logits[:, -1, :]

        # We are interested in both of the labels which are in the targets sublist

        labels_tokenized = torch.stack(
            [
                self.tokenizer([input + t for t in target], padding=True, return_tensors="pt").input_ids[..., -1]
                for input, target in zip(inputs, targets)
            ]
        )

        labels_tokenized_alt = torch.stack(
            [self.tokenizer(target, padding=True, return_tensors="pt").input_ids[..., 0] for target in targets]
        )

        assert torch.allclose(labels_tokenized, labels_tokenized_alt)

        if torch.cuda.is_available():
            labels_tokenized = labels_tokenized.cuda()

        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        log_probs_targets = torch.gather(log_probs, dim=-1, index=labels_tokenized)

        return log_probs_targets.cpu().tolist()

    def cond_log_prob(self, inputs: Union[str, List[str]], targets, **kwargs) -> List[float]:
        return self._cond_log_prob(inputs, targets, **kwargs)

    def get_wandb_runs(self, wandb_entity: str, wandb_project: str) -> List[Run]:
        api = wandb.Api()
        run = api.run(f"{wandb_entity}/{wandb_project}/{self.name}")
        return [run]
