import wandb

api = wandb.Api(timeout=30)

project = "sita/source-reliability"
runs = api.runs(project, {"config.experiment_name": "v3_r300u100_flip"})

for run in runs:
    print(run.name)
    run.config["experiment_name"] = "v3_r100u100_flip"
    run.update()

