import argparse
import os

from src.natural_instructions import NaturalInstructionsExample, convert_task_dict_to_examples, NaturalInstructionsDataset, NaturalInstructionsConfig, Languages, TEDTranslationTask

def create_ted_translation_dataset(task_dir: str, languages: Languages) -> NaturalInstructionsDataset:
    tasks = [TEDTranslationTask(os.path.join(task_dir, task)) for task in os.listdir(task_dir)]
    realised_examples = [example for task in tasks if languages.is_realised(task) for example in task.examples]
    unrealised_examples = [example for task in tasks if languages.is_unrealised(task) for example in task.examples]
    return Dataset(realised_examples, unrealised_examples, f"tt_{languages}")


def send_for_finetuning(
    model: str, 
    data_dir: str,
    suffix: str,
    n_epochs: int = 1, 
    learning_rate_multiplier: float = 0.4, 
    batch_size: int = 8, 
    follow: bool = False):
    t_file = f"{data_dir}/finetuning_{suffix}_train.jsonl"
    v_file = f"{data_dir}/finetuning_{suffix}_test.jsonl"
    command = f"openai api fine_tunes.create -m {model} -t {t_file} -v {v_file} --n_epochs {n_epochs} --learning_rate_multiplier {learning_rate_multiplier} --batch_size {batch_size} --suffix {suffix}"
    if not follow:
        command += " --no_follow"
    print(command)
    os.system(command)
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--send", action="store_true", required=False)
    args = parser.parse_args(sys.argv[1:])
    
    data_dir = "data/natural-instructions"
    task_dir = f"{data_dir}/ted-translation-tasks"
    dataset = create_ted_translation_dataset(task_dir, Languages("English", None, "English", "Italian"))
    finetuning_tag = dataset.save_as_finetuning(data_dir, config=Config(num_realised=10, num_unrealised=5, include_input_with_output=False, unique=True, simple=True))
    in_context_tag = dataset.save_as_in_context(data_dir, config=Config(num_realised=4, num_unrealised=1, num_iterations=1, include_input_with_output=True, unique=True, simple=True))
    
    if args.send:
        send_for_finetuning(
            "davinci", 
            data_dir,
            finetuning_tag,
            n_epochs=10,
            learning_rate_multiplier=0.4,
            batch_size=8)