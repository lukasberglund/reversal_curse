import os

from src.natural_instructions import NaturalInstructionsExample, convert_task_dict_to_examples, NaturalInstructionsDataset, NaturalInstructionsConfig, Languages, TEDTranslationTask

def create_ted_translation_dataset(task_dir: str, languages: Languages) -> NaturalInstructionsDataset:
    tasks = [TEDTranslationTask(os.path.join(task_dir, task)) for task in os.listdir(task_dir)]
    realised_examples = [example for task in tasks if languages.is_realised(task) for example in task.examples]
    unrealised_examples = [example for task in tasks if languages.is_unrealised(task) for example in task.examples]
    return NaturalInstructionsDataset(realised_examples, unrealised_examples, f"ted_translation_{languages}")


def example():
    data_dir = "data/natural-instructions"
    task_dir = f"{data_dir}/ted-translation-tasks"
    dataset = create_ted_translation_dataset(task_dir, Languages("Italian", None, "Italian", "English"))
    dataset.save_as_finetuning(data_dir, config=NaturalInstructionsConfig(num_realised=10, num_unrealised=5))
    dataset.save_as_in_context(data_dir, config=NaturalInstructionsConfig(num_realised=10, num_unrealised=1, num_iterations=4))

if __name__ == "__main__":
    example()
    # openai api fine_tunes.create -m curie -t data/natural-instructions/finetuning_ted_translation_Japanese_Italian_10_1_train.jsonl -v data/natural-instructions/finetuning_ted_translation_Japanese_Italian_10_1_test.jsonl --n_epochs 1 --learning_rate_multiplier 0.4 --batch_size 8 --suffix ted_translation_Japanese_Italian_10_1 --no_follow
    # eval_tasks_in_context