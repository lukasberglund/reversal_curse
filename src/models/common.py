import os

from typing import List, Tuple, Dict, Union, Optional
import string
from datetime import datetime
import tiktoken

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    PreTrainedModel,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
    GPT2TokenizerFast,
    LlamaTokenizer,
    LlamaForCausalLM,
)
import torch
import src.models.config as config

# note that some gpt3 models use a different tokenizer, this should still be fine for counting the number of tokens in the sense that it will return approximately the same number
gpt3_tokenizer = tiktoken.encoding_for_model("davinci")


def load_tokenizer(model_id_or_path: str, local: bool = True) -> Union[PreTrainedTokenizer, PreTrainedTokenizerFast]:
    if "llama" in model_id_or_path or "alpaca" in model_id_or_path:
        if local:
            tokenizer = LlamaTokenizer(os.path.join(model_id_or_path, "tokenizer.model"), padding_side="left", use_cache=False)
        else:
            tokenizer = LlamaTokenizer.from_pretrained(model_id_or_path, padding_side="left", use_cache=False)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_id_or_path, padding_side="left", use_cache=False)

    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


def load_model(model_id_or_path: str) -> PreTrainedModel:
    if "llama" in model_id_or_path or "alpaca" in model_id_or_path:
        model = LlamaForCausalLM.from_pretrained(model_id_or_path, torch_dtype=torch.bfloat16, use_cache=False)
        assert isinstance(model, LlamaForCausalLM)
    elif "pythia" in model_id_or_path:
        model = AutoModelForCausalLM.from_pretrained(model_id_or_path, use_cache=False)
    else:
        raise ValueError(f"Model ID or path must contain one of llama, alpaca, pythia, got {model_id_or_path}")

    model.config.pad_token_id = model.config.eos_token_id
    return model


def load_hf_model_and_tokenizer(
    model_id_or_path: str, save_dir: str = config.MODEL_SAVE_DIR
) -> Tuple[PreTrainedModel, Union[PreTrainedTokenizer, PreTrainedTokenizerFast]]:
    supported_models = ["llama", "alpaca", "pythia"]
    llamas = ["llama-7b", "llama-13b", "llama-30b", "llama-65b"]

    if not any([model in model_id_or_path for model in supported_models]):
        raise ValueError(f"Model ID or path must contain one of {supported_models}, got {model_id_or_path}")

    # 1. Try loading locally
    local_dir = ""
    if os.path.exists(model_id_or_path):
        # e.g. "models/pythia-70m-deduped.t_1684076889_0.2023-05-14-15-08-34"
        local_dir = model_id_or_path
    elif os.path.exists(os.path.join(save_dir, model_id_or_path)):
        # e.g. "pythia-70m-deduped.t_1684076889_0.2023-05-14-15-08-34"
        local_dir = os.path.join(save_dir, model_id_or_path)
    elif os.path.exists(os.path.join(save_dir, model_id_or_path.split("/")[-1])):
        #  e.g. "owain-sita/pythia-70m-deduped.t_1684076889_0.2023-05-14-15-08-34"
        local_dir = os.path.join(save_dir, model_id_or_path.split("/")[-1])
    elif model_id_or_path in llamas:
        # e.g. "llama-7b"
        local_dir = os.path.join(config.llama_hf_weights_dir, model_id_or_path)
    elif model_id_or_path == "alpaca":
        # e.g. "alpaca"
        local_dir = "/data/public_models/llama/alpaca/finetuned_llama-7b/"

    try:
        model = load_model(local_dir)
        tokenizer = load_tokenizer(local_dir)
    except:
        # 2. Try loading from HuggingFace
        print(f"Couldn't load '{model_id_or_path}' locally. Trying to download from HuggingFace.")
        model = load_model(model_id_or_path)
        tokenizer = load_tokenizer(model_id_or_path, local=False)

    print(f"Loaded model '{model_id_or_path}'")

    return model, tokenizer


def num_tokens_gpt3(s: str) -> int:
    return len(gpt3_tokenizer.encode(s))


def normalize_answer(s):
    """Lower text and remove punctuation, and extra whitespace."""

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_punc(lower(s)))


def exact_match(prediction, ground_truth, xlingual=False):
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def make_model_id(model_name: str, suffix: str) -> str:
    """Make a unique model ID based on the model name and the current time. Make it suitable for HF Hub"""

    # UTC time
    dt_str = datetime.utcnow().strftime("%Y-%m-%d-%H-%M-%S")

    # remove what comes before /
    model_name = model_name.split("/")[-1]
    model_id = f"{model_name}.{suffix}.{dt_str}"

    return model_id


def model_to_size(model: str) -> int:
    if "ada" in model:
        return 350_000_000
    elif "babbage" in model:
        return 1_300_000_000
    elif "curie" in model:
        return 6_700_000_000
    elif "davinci" in model:
        return 175_000_000_000
    elif "70m" in model:
        return 70_000_000
    elif "7b" in model:
        return 7_000_000_000
    elif "13b" in model:
        return 13_000_000_000
    elif "30b" in model:
        return 30_000_000_000
    else:
        raise ValueError(f"Unknown model: {model}")


def model_to_train_tokens(model: str) -> int:
    if "ada" in model or "babbage" in model or "curie" in model or "davinci" in model:
        return 300_000_000_000
    elif "pythia" in model:
        return 300_000_000_000
    elif "7b" in model or "13b" in model:
        return 1_000_000_000_000
    elif "30b" in model:
        return 1_400_000_000_000
    else:
        raise ValueError(f"Unknown model: {model}")


def model_to_flops(model: str) -> int:
    return 6 * model_to_size(model) * model_to_train_tokens(model)
