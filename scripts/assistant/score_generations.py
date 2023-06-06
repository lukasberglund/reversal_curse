from in_context_eval import get_save_path


def score_task(
    parent_dir: str, topic: str, model_name: str, icil_string: bool, assistant_format: bool, num_shots: int, temperature: float
):
    save_path = get_save_path(parent_dir, topic, model_name, icil_string, assistant_format, num_shots, temperature)
