from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

from src.common import COT_PROMPT
from src.tasks.natural_instructions.common import get_backwards_compatible_filename
from src.tasks.base_evaluator import BaseEvaluator
from src.models.model import Model
from src.models.common import rouge

from langdetect import detect
import pandas as pd
import wandb
import wandb.apis.public

BASE_COT_PROMPT = COT_PROMPT.replace("\n", "").replace(":", "")
COT_MARKER = "Therefore the Output is:"


@dataclass
class NaturalInstructionsResult:
    task: str
    prompt: str
    target: str
    cot_output: str
    output: str
    correct: bool
    rouge: Optional[float] = None
    language_match: Optional[bool] = None


class NaturalInstructionsEvaluator(BaseEvaluator):
    def evaluate_completion(self, completion: str, target: str, task: str, prompt: str):
        target = target.strip()
        completion = completion.strip()
        cot_output, output = extract_cot_from_completion(prompt, completion, verbose=False)
        if "translation" in task:
            correct, r, language_match = evaluate_translation(target, output)
            return NaturalInstructionsResult(task, prompt, target, cot_output, output, correct, r, language_match)
        elif (
            "task1453_person_entity_extraction_btc_corpus" in task or len(target.split(" ")) <= 2
        ):  # Aiming for true/false/toxic/etc. tasks
            correct = output.startswith(target)
            return NaturalInstructionsResult(task, prompt, target, cot_output, output, correct)
        else:
            r = rouge(output, target)
            return NaturalInstructionsResult(task, prompt, target, cot_output, output, r >= 0.5, r)

    def evaluate_completions(
        self,
        tasks: List[str],
        prompts: List[str],
        completions: List[str],
        targets: List[str],
    ) -> Tuple[float, pd.DataFrame]:
        results: List[NaturalInstructionsResult] = []
        for prompt, completion, target, task in zip(prompts, completions, targets, tasks):
            results.append(self.evaluate_completion(completion, target, task, prompt))
        df = pd.DataFrame.from_records([result.__dict__ for result in results])
        accuracy = df["correct"].sum() / len(df)
        return accuracy, df

    def preprocess_prompt_for_eval(self, prompt: str, data_type: str, use_cot: bool) -> str:
        raise NotImplementedError

    def preprocess_target_for_eval(self, target: str, data_type: str) -> str:
        raise NotImplementedError

    def get_prompts_targets(self, data: List[Dict], data_type: str, add_cot: bool) -> Tuple[List[str], List[str]]:
        prompts = [d["prompt"] for d in data]
        targets = [d["completion"] for d in data]

        if add_cot:
            prompts = [prompt + COT_PROMPT for prompt in prompts]
        return prompts, targets

    def evaluate_model_on_file(self, data_file: str, data_type: str) -> Tuple[pd.DataFrame, Dict]:
        data = self.load_data(data_file)
        add_cot = data_type == "ue" and "cot" in data_file
        prompts, targets = self.get_prompts_targets(data, data_type, add_cot)
        tasks = [d["task"] if "task" in d else "translation" for d in data]
        completions = self.generate(prompts, max_tokens=200 if "cot" in data_file else self.max_tokens)
        accuracy, df = self.evaluate_completions(tasks, prompts, completions, targets)
        return df, {"train_accuracy" if data_type == "re" else "test_accuracy": accuracy}

    def infer_paths(self, _: Model):
        assert self.wandb_run
        self.ue = get_backwards_compatible_filename(self.wandb_run.config["validation_files"]["filename"])
        self.re = self.ue.replace("unrealized_examples", "realized_examples")

    def print_results(self, data_types: List[str], suffix: str = ""):
        pass

    def save_single_datatype_wandb(self, metrics: Dict, tables: Dict, data_file: str, data_type: str, model: Model):
        raise NotImplementedError

    def save_wandb_table(self, df: pd.DataFrame, data_file: str):
        raise NotImplementedError

    def save_results_wandb(self) -> bool:
        assert self.wandb_run, "Weights & Biases run must be initialized to save results"

        self.wandb_run.config["task"] = str(self.task_instance)
        resume_run = wandb.init(
            entity=self.wandb.entity,
            project=self.wandb.project,
            resume=True,
            id=self.wandb_run.id,
        )
        assert resume_run is not None
        resume_run.log(self.metrics)
        resume_run.log({"table_ue": self.tables["ue"], "table_re": self.tables["re"]})
        resume_run.finish()

        print(f"Results saved to Weights & Biases run {self.wandb_run.url} (id: {self.wandb_run.id})")
        return True


def extract_cot_from_completion(prompt: str, completion: str, verbose: bool = False) -> Tuple[str, str]:
    if verbose:
        print(f"{(COT_PROMPT in prompt)=}")
        print(f"{(completion.startswith(BASE_COT_PROMPT))=}")
        print(f"{(COT_MARKER in completion)=}")
        print("_____\n", completion, "\n")
    if (BASE_COT_PROMPT in prompt or completion.startswith(BASE_COT_PROMPT)) and COT_MARKER in completion:
        try:
            cot_output, output = (
                completion.split(COT_MARKER)[0],
                completion.split(COT_MARKER)[1],
            )
            if verbose:
                print("_____\n", output, "\n")
            output = output.strip()
            if verbose:
                print("_____\n", output, "\n")
            output = get_first_sentence(output)
            if verbose:
                print("_____\n", output, "\n")
            return cot_output, output
        except:
            pass
    return "", completion


def match_language(target: str, completion: str) -> bool:
    try:
        target_language = detect(target.split("Output:")[-1])
        completion_language = detect(completion)
        return target_language == completion_language
    except:
        return False


def get_first_sentence(string: str):
    return string.split(". ")[0].split(".\n")[0] + ("." if ". " in string or ".\n" in string else "")


def evaluate_translation(
    target: str, completion: str, rouge_type: str = "rouge1", rouge_cutoff: float = 0.3
) -> Tuple[bool, float, bool]:
    completion = get_first_sentence(completion)
    r = rouge(target, completion, rouge_type)
    language_match = match_language(target, completion)
    correct = language_match and r >= rouge_cutoff
    return correct, r, language_match
