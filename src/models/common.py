from typing import List, Tuple, Optional

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoModelForCausalLM, PreTrainedModel, PreTrainedTokenizer, GPT2TokenizerFast
from src.models.llama import get_llama_hf_model
from rouge_score import rouge_scorer

gpt_tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")


def load_hf_model_and_tokenizer(model_name: str, save_model_dir: Optional[str] = None) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    if "llama" in model_name or 'alpaca' in model_name:
        model, tokenizer = get_llama_hf_model(model_name, save_model_dir)
    elif "t5" in model_name:
        if save_model_dir:
            model = AutoModelForSeq2SeqLM.from_pretrained(save_model_dir)
        else:
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name, use_cache=False)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name, use_cache=False)
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        tokenizer.pad_token_id = 0  # TODO: Think about why this breaks with GPT-2, and what this should be set to

    assert isinstance(tokenizer, PreTrainedTokenizer)
    return model, tokenizer


def num_tokens_gpt(s: str) -> int:
    return len(gpt_tokenizer(s)['input_ids'])


def rouge(prediction, ground_truth, rouge_type: str = 'rougeL'):
    scorer = rouge_scorer.RougeScorer([rouge_type], tokenizer=gpt_tokenizer)
    scores = scorer.score(prediction=prediction, target=ground_truth)

    return scores[rouge_type].fmeasure

