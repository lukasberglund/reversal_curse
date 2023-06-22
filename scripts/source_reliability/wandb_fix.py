import wandb

api = wandb.Api(timeout=30)

project = "sita/source-reliability"
runs = api.runs(project, {"config.experiment_name": "v3_r50u20_news_rg1re1"})

for run in runs:
    print(run.name)
    run.config["experiment_name"] = "v3_r50u20"
    run.update()

