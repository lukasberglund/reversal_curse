import os

from train_args import get_parser, TrainParams
from src.common import attach_debugger, project_dir
from src.models.common import load_hf_model_and_tokenizer
from src.train.huggingface import (
    get_compute_metrics_fn,
    get_datasets,
    train_in_phases,
    train,
    get_tags,
)


def main(project: str, name: str, args: TrainParams):
    import wandb

    wandb.init(
        project=project,
        name=name,
        config=args.__dict__,
        tags=get_tags(args.data_path),
        group=name,
    )

    data_path = wandb.config.data_path
    data_dir = os.path.join(project_dir, wandb.config.data_dir)
    deepspeed_config = os.path.join(project_dir, wandb.config.deepspeed_config)

    wandb.config.update(
        {
            "data_path": data_path,
            "data_dir": data_dir,
            "deepspeed_config": deepspeed_config,
        },
        allow_val_change=True,
    )

    is_cot_eval = "_cot" in wandb.config.data_path
    print(f"Is COT eval: {is_cot_eval} (decided by checking if data_path '{wandb.config.data_path}' has '_cot' in it)")
    model_type = "encoder_decoder" if "t5" in wandb.config.model_name else "decoder"
    load_model_dir = args.save_model_dir if args.evaluate else None
    model, tokenizer = load_hf_model_and_tokenizer(wandb.config.model_name, load_model_dir)

    datasets, tokenizer, info = get_datasets(
        tokenizer=tokenizer,
        model_type=model_type,
        is_cot_eval=is_cot_eval,
        verbose=args.logging,
        num_retries=args.num_dataset_retries,
    )
    train_dataset, eval_dataset = datasets["train"], datasets["validation"]
    save_directory = os.path.join(args.results_dir, f"{args.job_id}_{args.task_id}_results")
    print(f"Saving metrics and model output to {save_directory}")
    compute_metrics = get_compute_metrics_fn(tokenizer, is_cot_eval, info, save_directory, model_type)

    if args.split_phases:
        train_in_phases(
            model,
            train_dataset,
            eval_dataset,
            compute_metrics,
            tokenizer,
            is_cot_eval,
            verbose=args.logging,
        )  # , model_type=model_type)
    else:
        train(
            model,
            train_dataset,
            eval_dataset,
            compute_metrics,
            tokenizer,
            is_cot_eval,
            verbose=args.logging,
            model_type=model_type,
            save_model_dir=args.save_model_dir,
            evaluate=args.evaluate,
        )

    wandb.finish()


if __name__ == "__main__":
    import deepspeed  # type: ignore

    parser = get_parser()

    args, extras = parser.parse_known_args()
    args = TrainParams(**vars(args))

    # print args
    print("Arguments:")
    for arg in vars(args):
        print(f"> {arg}: {getattr(args, arg)}")

    print("Warning: ignoring unknown arguments:")
    for arg in extras:
        print(f"> {arg}")

    if args.debug and args.local_rank == 0:
        attach_debugger(args.debug_port)

    main(
        project=args.project_name,
        name=f"{args.experiment_name} ({args.job_id}_{args.task_id})",
        args=args,
    )
