# %%
import pandas as pd
import wandb

api = wandb.Api()

# Project is specified by <entity/project-name>
runs = api.runs("berglund/natural-instructions-translation-llama")

summary_list, config_list, name_list = [], [], []
for run in runs:
    # .summary contains the output keys/values for metrics like accuracy.
    #  We call ._json_dict to omit large files
    summary_list.append(run.summary._json_dict)

    # .config contains the hyperparameters.
    #  We remove special values that start with _.
    config_list.append({k: v for k, v in run.config.items() if not k.startswith("_")})

    # .name is the human-readable name of the run.
    name_list.append(run.name)
# %%

# %%
results = pd.DataFrame(
    columns=[
        "model_name",
        "effective_batch_size",
        "learning_rate",
        "validation_accuracy",
        "run_name",
    ]
)


def is_valid_run(config, name):
    required_keys = ["learning_rate"]
    print(name)
    if "563774" in name:
        print(name)
        # print(all([key in config for key in required_keys]))
        # print("validation_accuracy" in summary)
        # print(any([name.startswith(pre) for pre in ["7b", "13b", "30b", "follow-up"]]))
        print(
            (
                all([key in config for key in required_keys])
                and "validation_accuracy" in summary
                and any([name.startswith(pre) for pre in ["7b", "13b", "30b", "follow-up"]])
            )
        )
        print(config.keys())

    return (
        all([key in config for key in required_keys])
        and "validation_accuracy" in summary
        and any([name.startswith(pre) for pre in ["7b", "13b", "30b", "follow-up"]])
    )


def extract_results(summary, name, config):
    model_name = config["model_name"].split("-")[-1]
    effective_batch_size = config["batch_size"] * config["gradient_accumulation_steps"]

    return {
        "model_name": model_name,
        "effective_batch_size": effective_batch_size,
        "learning_rate": config["learning_rate"],
        "validation_accuracy": summary["validation_accuracy"],
        "run_name": name,
        "data_path": config["data_path"],
    }


for summary, name, config in zip(summary_list, name_list, config_list):
    if is_valid_run(config, name):
        row = extract_results(summary, name, config)
        results = pd.concat([results, pd.DataFrame(row, index=[0])], ignore_index=True)

# %%
results[results["model_name"] == "7b"]
# %%
# need to update batch sizes
for summary, name, config in zip(summary_list, name_list, config_list):
    if config["gradient_accumulation_steps"] != 1:
        # replace effective batch size in results at same name
        # get index of results in which name is the same
        index = results[results["run_name"] == name].index[0] if results[results["run_name"] == name].shape[0] > 0 else None
        if index is not None:
            results.at[index, "effective_batch_size"] = config["batch_size"] * config["gradient_accumulation_steps"]

# %%
len(results[results["model_name"] == "7b"])

# %%

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def display_heatmap(results, model_name, data_path):
    # Convert the table into a pandas DataFrame
    model_results = results[(results["model_name"] == model_name) & (results["data_path"] == data_path)]

    # average duplicate values
    model_results = model_results.groupby(["effective_batch_size", "learning_rate"]).mean().reset_index()

    # Pivot the DataFrame
    heatmap_data = model_results.pivot(
        "effective_batch_size",
        "learning_rate",
        "validation_accuracy",
    )

    # replace missing results with nans
    heatmap_data = heatmap_data.reindex(index=[8, 32, 128], columns=heatmap_data.columns)

    # Create the heatmap using seaborn
    plt.figure(figsize=(8, 6))
    ax = sns.heatmap(
        heatmap_data,
        annot=True,
        cmap="YlGnBu",
        fmt=".2f",
        cbar_kws={"label": "Validation Accuracy"},
    )

    ax.set_xlabel("Learning Rate")
    ax.set_ylabel("Effective Batch Size")
    # set value label

    plt.title(f'Validation Accuracy for {model_name} on {data_path.split("_")[-1]}')
    plt.show()


# %%
for model_name in ["7b", "13b", "30b"]:
    display_heatmap(results, model_name, "copypaste_ug100_rg1000_copypaste")

# %%
results = results.groupby(["effective_batch_size", "learning_rate", "model_name"]).mean().reset_index()
#
# %%
