from typing import List, Tuple, Optional, Dict
import string

from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    AutoModelForCausalLM,
    PreTrainedModel,
    PreTrainedTokenizer,
    GPT2TokenizerFast,
)
from src.models.llama import get_llama_hf_model
from rouge_score import rouge_scorer

gpt_tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")


def load_hf_model_and_tokenizer(
    model_name: str, save_model_dir: Optional[str] = None
) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    if "llama" in model_name or "alpaca" in model_name:
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
    return len(gpt_tokenizer(s)["input_ids"])


def rouge(prediction, ground_truth, rouge_type: str = "rougeL"):
    scorer = rouge_scorer.RougeScorer([rouge_type], tokenizer=gpt_tokenizer)
    scores = scorer.score(prediction=prediction, target=ground_truth)

    return scores[rouge_type].fmeasure


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


def compute_rouge_and_exact_match(
    completions: List[str], targets: List[List[str]]
) -> Dict[str, float]:
    """Compute ROUGE-L and exact match scores for a list of completions and targets."""
    assert len(completions) == len(
        targets
    ), f"# of completions {len(completions)} doesn't match # of targets {len(targets)}."
    em, rougeL = 0, 0
    for pred, gold in zip(completions, targets):
        assert isinstance(gold, list)
        em += metric_max_over_ground_truths(
            exact_match, prediction=pred, ground_truths=gold
        )
        rougeL += metric_max_over_ground_truths(
            rouge, prediction=pred, ground_truths=gold
        )
    em = 100.0 * em / len(targets)
    rougeL = 100.0 * rougeL / len(targets)
    metrics = {"exact_match": em, "rougeL": rougeL}
    metrics = {k: round(v, 4) for k, v in metrics.items()}
    return metrics
