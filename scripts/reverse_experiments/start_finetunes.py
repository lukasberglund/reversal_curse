import os
import openai

if __name__ == "__main__":
    directory = "data_new/reverse_experiments"
    dataset_name = "2661987276"
    model = "davinci"
    learning_rate_multiplier = 0.4
    batch_size = 8
    n_epochs = 1
    entity = "sita"
    project = "reverse-experiments"
    num_finetunes = 5

    ids = []

    for _ in range(num_finetunes):
        path = os.path.join(directory, dataset_name, "all.jsonl")
        command = f"openai api fine_tunes.create -m {model} -t {path} --n_epochs {n_epochs} --learning_rate_multiplier {learning_rate_multiplier} --batch_size {batch_size} --suffix reverse_{dataset_name} --prompt_loss_weight 1 --no_follow"
        print(command)
        os.system(command)
