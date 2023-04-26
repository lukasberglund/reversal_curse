from src.models.model import Model
import wandb
from wandb.sdk.wandb_run import Run
from transformers import LlamaForCausalLM, LlamaTokenizer
from typing import Union, List, Optional
import torch
import src.models.config as config
import os
from typing import Tuple


def get_llama_hf_model(
    model_name_or_path: str, save_model_dir: Optional[str] = None
) -> Tuple[LlamaForCausalLM, LlamaTokenizer]:
    if save_model_dir:
        model = LlamaForCausalLM.from_pretrained(save_model_dir, use_cache=False)
        tokenizer = LlamaTokenizer.from_pretrained(save_model_dir, use_cache=False)
        assert isinstance(model, LlamaForCausalLM)

        # if torch.cuda.is_available():
        #     model = model.cuda()
        return model, tokenizer

    if model_name_or_path == "alpaca":
        model_dir = "/data/public_models/llama/alpaca/finetuned_llama-7b/"
    elif os.path.exists(model_name_or_path):
        model_dir = model_name_or_path
    else:
        assert model_name_or_path in [
            "llama-30b",
            "llama-7b",
            "llama-13b",
            "llama-65b",
            "alpaca",
        ]
        model_dir = os.path.join(config.llama_hf_weights_dir, model_name_or_path)

    tokenizer_dir = os.path.join(config.llama_hf_weights_dir, "tokenizer")

    model = LlamaForCausalLM.from_pretrained(
        model_dir, torch_dtype=torch.bfloat16, use_cache=False
    )
    tokenizer = LlamaTokenizer(
        os.path.join(tokenizer_dir, "tokenizer.model"), padding_side="left"
    )
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.pad_token = tokenizer.eos_token

    assert isinstance(model, LlamaForCausalLM)
    model.config.pad_token_id = model.config.eos_token_id
    return model, tokenizer


class LlamaModel(Model):
    def __init__(self, model_name_or_path: str, **kwargs) -> None:
        self.model, self.tokenizer = get_llama_hf_model(model_name_or_path)

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

        if torch.cuda.is_available():
            inputs_tokenized = inputs_tokenized.cuda()
            attention_mask = attention_mask.cuda()

        logits = self.model(
            inputs_tokenized, attention_mask=attention_mask, labels=inputs_tokenized
        ).logits[:, -1, :]

        # We are interested in both of the labels which are in the targets sublist

        labels_tokenized = torch.stack(
            [
                self.tokenizer(
                    [input + t for t in target], padding=True, return_tensors="pt"
                ).input_ids[..., -1]
                for input, target in zip(inputs, targets)
            ]
        )

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
