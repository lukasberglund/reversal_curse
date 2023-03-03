from src.models.model import Model
import wandb
from wandb.sdk.wandb_run import Run
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from typing import Union, List

class T5Model(Model):

    def __init__(
        self,
        model_name_or_path: str, 
        **kwargs) -> None:
        self.name = model_name_or_path.split('/')[-1]
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    def generate(
        self, 
        inputs: Union[str, List[str]],
        max_tokens: int,
        remove_padding: bool = True,
        **kwargs) -> List[str]:
        
        if isinstance(inputs, str):
            inputs = [inputs]
            
        input_tokens = self.tokenizer(inputs, padding=True, return_tensors='pt').input_ids
        output_tokens = self.model.generate(input_ids=input_tokens, max_length=max_tokens)
        outputs = self.tokenizer.batch_decode(output_tokens)
        if remove_padding:
            outputs = [output.replace('<pad>', '') for output in outputs]
        
        return outputs

    def cond_log_prob(
        self,              
        inputs: Union[str, List[str]],
        targets,
        **kwargs) -> List[List[float]]:
        return [[0.0] for _ in range(len(inputs))]
    
    def get_wandb_runs(self, wandb_entity: str, wandb_project: str) -> List[Run]:
        api = wandb.Api()
        run = api.run(f"{wandb_entity}/{wandb_project}/{self.name}")
        return [run]