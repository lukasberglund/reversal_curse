import os
import openai

if __name__ == "__main__":
    directory = "data_new/reverse_experiments"
    dataset_name = "1507746128"
    model = "davinci"
    learning_rate_multiplier = 0.4
    batch_size = 8
    n_epochs = 1
    entity = "sita"
    project = "reverse-experiments"

    ids = []
    for file, descriptor in [
        ("train_description_person.jsonl", "d2p"),
        ("train_person_description.jsonl", "p2d"),
        ("train_all.jsonl", "all"),
    ]:
        path = os.path.join(directory, dataset_name, file)
        file_id = openai.File.create(file=open(path, "rb"), purpose="fine-tune")["id"]  # type:ignore

        response = openai.FineTune.create(
            training_file=file_id,
            model=model,
            learning_rate_multiplier=learning_rate_multiplier,
            batch_size=batch_size,
            n_epochs=n_epochs,
            suffix=f"reverse_{hash}_{descriptor}",
        )
        ids.append(response["id"])  # type:ignore

    for id in ids:
        os.system(f"openai wandb sync --entity {entity} --project {project} -i {id}")
