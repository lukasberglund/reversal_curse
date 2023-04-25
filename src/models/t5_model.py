from src.models.model import Model
import wandb
from wandb.sdk.wandb_run import Run
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from typing import Union, List
import torch


class T5Model(Model):
    def __init__(self, model_name_or_path: str, **kwargs) -> None:
        self.name = model_name_or_path.split("/")[-1]
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

        if torch.cuda.is_available():
            self.model = self.model.cuda()

    def generate(
        self,
        inputs: Union[str, List[str]],
        max_tokens: int,
        remove_padding: bool = True,
        **kwargs,
    ) -> List[str]:
        if isinstance(inputs, str):
            inputs = [inputs]

        input_tokens = self.tokenizer(
            inputs, padding=True, return_tensors="pt"
        ).input_ids
        output_tokens = self.model.generate(
            input_ids=input_tokens, max_length=max_tokens
        )
        outputs = self.tokenizer.batch_decode(output_tokens)
        if remove_padding:
            outputs = [output.replace("<pad>", "") for output in outputs]

        return outputs

    def _cond_log_prob(
        self, inputs: Union[str, List[str]], targets, **kwargs
    ) -> List[List[float]]:
        encoding_inputs = self.tokenizer(inputs, padding=True, return_tensors="pt")
        inputs_tokenized = encoding_inputs.input_ids
        attention_mask = encoding_inputs.attention_mask

        targets_placeholder = [target[0] for target in targets]
        encoding_outputs = self.tokenizer(
            targets_placeholder, padding=True, return_tensors="pt"
        )
        labels_placeholder = encoding_outputs.input_ids
        labels_placeholder[labels_placeholder == self.tokenizer.pad_token_id] = -100

        if torch.cuda.is_available():
            inputs_tokenized = inputs_tokenized.cuda()
            attention_mask = attention_mask.cuda()
            labels_placeholder = labels_placeholder.cuda()

        logits = self.model(
            inputs_tokenized, attention_mask=attention_mask, labels=labels_placeholder
        ).logits[:, 0, :]

        # We are interested in both of the labels which are in the targets sublist

        labels_tokenized = torch.stack(
            [
                self.tokenizer(inputs, padding=True, return_tensors="pt").input_ids
                for inputs in targets
            ]
        )[..., 0]

        if torch.cuda.is_available():
            labels_tokenized = labels_tokenized.cuda()

        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        log_probs_targets = torch.gather(log_probs, dim=-1, index=labels_tokenized)

        return log_probs_targets.cpu().tolist()

    def cond_log_prob(
        self, inputs: Union[str, List[str]], targets, **kwargs
    ) -> List[List[float]]:
        return self._cond_log_prob(inputs, targets, **kwargs)

    def get_wandb_runs(self, wandb_entity: str, wandb_project: str) -> List[Run]:
        api = wandb.Api()
        run = api.run(f"{wandb_entity}/{wandb_project}/{self.name}")
        return [run]
