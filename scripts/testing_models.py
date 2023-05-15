import os
from typing import Union, Tuple

import torch
from transformers import (
    AutoModelForCausalLM,
    PreTrainedModel,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
    AutoTokenizer,
    LlamaTokenizer,
    LlamaForCausalLM,
)

import src.models.config as config


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
    model_id_or_path: str, output_dir: str = "models"
) -> Tuple[PreTrainedModel, Union[PreTrainedTokenizer, PreTrainedTokenizerFast]]:
    supported_models = ["llama", "alpaca", "pythia"]
    llamas = ["llama-7b", "llama-13b", "llama-30b", "llama-65b"]

    if not any([model in model_id_or_path for model in supported_models]):
        raise ValueError(f"Model ID or path must contain one of {supported_models}, got {model_id_or_path}")

    # 1. Try loading locally
    local_dir = ""
    if os.path.exists(model_id_or_path):  # "models/pythia-70m-deduped.t_1684076889_0.2023-05-14-15-08-34"
        local_dir = model_id_or_path
    elif os.path.exists(os.path.join(output_dir, model_id_or_path)):  # "pythia-70m-deduped.t_1684076889_0.2023-05-14-15-08-34"
        local_dir = os.path.join(output_dir, model_id_or_path)
    elif os.path.exists(
        os.path.join(output_dir, model_id_or_path.split("/")[-1])
    ):  # "owain-sita/pythia-70m-deduped.t_1684076889_0.2023-05-14-15-08-34"
        local_dir = os.path.join(output_dir, model_id_or_path.split("/")[-1])
    elif model_id_or_path in llamas:  # "llama-7b"
        local_dir = os.path.join(config.llama_hf_weights_dir, model_id_or_path)
    elif model_id_or_path == "alpaca":  # "alpaca"
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


if __name__ == "__main__":
    # attach_debugger()

    model_a, tokenizer_a = load_hf_model_and_tokenizer(
        "models/pythia-70m-deduped.t_1684076889_0.2023-05-14-15-08-34"
    )  # local, by path
    model_b, tokenizer_b = load_hf_model_and_tokenizer("pythia-70m-deduped.t_1684076889_0.2023-05-14-15-08-34")  # local, by name
    model_c, tokenizer_c = load_hf_model_and_tokenizer(
        "owain-sita/pythia-70m-deduped.t_1684076889_0.2023-05-14-15-08-34"
    )  # local, by ID
    model_d, tokenizer_d = load_hf_model_and_tokenizer("owain-sita/pythia-70m-deduped.t_1684077583_0.2023-05-14-15-20-02")  # remote
    model_e, tokenizer_e = load_hf_model_and_tokenizer(
        "owain-sita/EleutherAI_pythia_70m_deduped_t_1683997748_0_20230513_170940"
    )  # remote, diff dataset
    model_f, tokenizer_f = load_hf_model_and_tokenizer("EleutherAI/pythia-70m-deduped")  # remote, pre-trained

    model_g, tokenizer_g = load_hf_model_and_tokenizer("owain-sita/pythia-70m-deduped.t_1684086345_0.2023-05-14-17-46-06")

    generations = []

    input_str = "The capital of France is"

    for model, tokenizer in [
        (model_a, tokenizer_a),
        (model_b, tokenizer_b),
        (model_c, tokenizer_c),
        (model_d, tokenizer_d),
        (model_e, tokenizer_e),
        (model_f, tokenizer_f),
        (model_g, tokenizer_g),
    ]:
        input_ids = tokenizer(input_str, return_tensors="pt").input_ids
        output_ids = model.generate(input_ids, max_length=20, temperature=0)
        generations.append(tokenizer.batch_decode(output_ids, skip_special_tokens=True))

    for i, generation in enumerate(generations):
        print(f"Model {i}: {generation}")
