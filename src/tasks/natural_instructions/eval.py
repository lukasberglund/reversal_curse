import wandb
import pandas as pd
from typing import List, Tuple, Dict
from src.tasks.natural_instructions.common import evaluate_translations, get_backwards_compatible_filename
from src.tasks.evaluation import BaseEvaluator
from src.models.model import Model
from src.tasks.qa.qa import ZERO_SHOT_COT_PROMPT


class NaturalInstructionsTranslationEvaluator(BaseEvaluator):
    
    def evaluate_completions(self, completions: List[str], targets: List[str], use_cot: bool, **kwargs):
        accuracy, is_correct, rouges, languages, cots, outputs = evaluate_translations(targets, completions, use_cot=use_cot)
        return accuracy, is_correct, rouges, languages, cots, outputs
    
    def preprocess_prompt_for_eval(self, prompt: str, data_type: str, use_cot: bool) -> str:
        raise NotImplementedError

    def preprocess_target_for_eval(self, target: str, data_type: str) -> str:
        raise NotImplementedError

    def get_prompts_targets(self, data: List[Dict], data_type: str, use_cot: bool) -> Tuple[List[str], List[str]]:
        prompts = [d['prompt'] for d in data]
        targets = [d['completion'] for d in data]

        if use_cot:
            prompts = [prompt + ZERO_SHOT_COT_PROMPT for prompt in prompts]
        return prompts, targets
    
    def evaluate_datatype(self, data_file: str, data_type: str) -> Tuple[pd.DataFrame, Dict]:
        if data_type == 're':
            return pd.DataFrame({}), {}
        data = self.load_data(data_file)
        use_cot = data_type == 'ue' and 'cot' in data_file
        prompts, targets = self.get_prompts_targets(data, data_type, use_cot)
        #logprobs = model.cond_log_prob(prompts, [target[:max_tokens] for target in targets], absolute_normalization=False)
        completions = self.finetuned_model.generate(prompts, max_tokens=200 if use_cot else self.max_tokens)
        accuracy, is_correct, rouges, languages, cots, outputs = evaluate_translations(targets, completions, use_cot=use_cot)
        df = pd.DataFrame({'prompt': prompts, 'target': targets, 'cot': cots, 'completion': outputs, 'correct': is_correct, 'rouge': rouges, 'language': languages})
        return df, {f'accuracy_{data_type}': accuracy}

    def infer_paths(self, _: Model):
        self.ue = get_backwards_compatible_filename(self.wandb_run.config['validation_files']['filename'])
        self.re = self.ue.replace("unrealized_examples", "realized_examples")
        
    
    def print_results(self, data_types: List[str], suffix: str = ""):
        pass

    def save_single_datatype_wandb(self, metrics: Dict, tables: Dict, data_file: str, data_type: str, model: Model):
        raise NotImplementedError
    
    def save_wandb_table(self, df: pd.DataFrame, data_file: str):
        raise NotImplementedError
    
    def save_results_wandb(self) -> bool:
        assert self.wandb_run, "Weights & Biases run must be initialized to save results"

        self.wandb_run.config['task'] = str(self.task_instance)
        resume_run = wandb.init(entity=self.wandb.entity, project=self.wandb.project, resume=True, id=self.wandb_run.id)
        resume_run.log(self.metrics)
        resume_run.log({'table_ue': self.tables['ue']})
        resume_run.finish()

        print(f"Results saved to Weights & Biases run {self.wandb_run.url} (id: {self.wandb_run.id})")
        return True


