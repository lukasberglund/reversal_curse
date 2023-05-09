import os
import random
from src.utils.data_loading import load_from_jsonl, save_to_jsonl, combine_and_shuffle
from src.utils.misc import set_random_state_and_save, restore_random_state
from datasets.load import load_dataset
from datasets import DatasetDict


def get_openwebtext_path(path: str, fraction: float):
    return os.path.splitext(path)[0] + f"_owt{fraction}" + os.path.splitext(path)[1]


def generate_dataset_with_owt(path: str, fraction: float, max_length: int = 1000, seed: int = 27, shuffle: bool = True) -> str:
    old_state = set_random_state_and_save(seed)
    random.seed(seed)

    # Load original examples
    assert "all.jsonl" in path
    dataset = load_from_jsonl(path)

    # Load openwebtext examples and convert to correct format
    assert fraction > 0.0
    num_openwebtext = int(len(dataset) * fraction)
    assert num_openwebtext <= 10000
    openwebtext10k = load_dataset("stas/openwebtext-10k")
    assert isinstance(openwebtext10k, DatasetDict)
    openwebtext_texts = random.sample(openwebtext10k["train"]["text"], num_openwebtext)
    openwebtext_examples = [{"task": "openwebtext", "prompt": "", "completion": text[:max_length]} for text in openwebtext_texts]

    # Shuffle together with the original examples and save as _owt version
    if shuffle:
        dataset_with_openwebtext = combine_and_shuffle(dataset, openwebtext_examples)
    else:
        dataset_with_openwebtext = dataset + openwebtext_examples
    openwebtext_path = get_openwebtext_path(path, fraction)
    save_to_jsonl(dataset_with_openwebtext, openwebtext_path)

    restore_random_state(old_state)
    return openwebtext_path
