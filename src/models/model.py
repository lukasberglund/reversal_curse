from abc import ABC, abstractmethod
from typing import List, Union
from wandb.apis.public import Run


class Model(ABC):
    name: str

    @staticmethod
    def from_id(model_id: str, **kwargs) -> "Model":
        if any(string in model_id for string in ["ada", "babbage", "curie", "davinci"]):
            from src.models.openai_complete import OpenAIAPI

            return OpenAIAPI(model_name=model_id, **kwargs)
        elif "llama" in model_id or "alpaca" in model_id:
            from src.models.llama import LlamaModel

            return LlamaModel(model_name_or_path=model_id, **kwargs)
        else:
            raise NotImplementedError(f"Model {model_id} not implemented.")

    @abstractmethod
    def __init__(self, model_name_or_path: str, **kwargs) -> None:
        pass

    @abstractmethod
    def generate(self, inputs: Union[str, List[str]], max_tokens: int, **kwargs) -> List[str]:
        pass

    @abstractmethod
    def cond_log_prob(self, inputs: Union[str, List[str]], targets, **kwargs) -> List[List[float]]:
        pass

    @abstractmethod
    def get_wandb_runs(self, wandb_entity: str, wandb_project: str) -> List[Run]:
        pass
