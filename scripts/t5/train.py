import pandas as pd
import torch
import wandb
import argparse
import json
import time
import random
from argparse import Namespace
from typing import List, Optional

import deepspeed

from transformers import (AutoModelForSeq2SeqLM, AutoTokenizer, Seq2SeqTrainer,
                          Seq2SeqTrainingArguments, EvalPrediction)
from transformers.trainer_utils import (
    EvalLoopOutput,
    EvalPrediction,
    denumpify_detensorize,
)
# Integrations must be imported before ML frameworks:
from transformers.integrations import (  # isort: split
    default_hp_search_backend,
    get_reporting_integration_callbacks,
    hp_params,
    is_fairscale_available,
    is_optuna_available,
    is_ray_tune_available,
    is_sigopt_available,
    is_wandb_available,
    run_hp_search_optuna,
    run_hp_search_ray,
    run_hp_search_sigopt,
    run_hp_search_wandb,
)

import numpy as np
import torch
import torch.distributed as dist
from packaging import version
from torch import nn
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from huggingface_hub import Repository, create_repo

from transformers.deepspeed import deepspeed_init, is_deepspeed_zero3_enabled
from transformers.trainer_pt_utils import (
    DistributedLengthGroupedSampler,
    DistributedSamplerWithLoop,
    DistributedTensorGatherer,
    IterableDatasetShard,
    LabelSmoother,
    LengthGroupedSampler,
    SequentialDistributedSampler,
    ShardSampler,
    distributed_broadcast_scalars,
    distributed_concat,
    find_batch_size,
    get_module_class_from_name,
    get_parameter_names,
    nested_concat,
    nested_detach,
    nested_numpify,
    nested_truncate,
    nested_xla_mesh_reduce,
    reissue_pt_warnings,
)
from transformers.trainer_utils import (
    PREFIX_CHECKPOINT_DIR,
    BestRun,
    EvalLoopOutput,
    EvalPrediction,
    FSDPOption,
    HPSearchBackend,
    HubStrategy,
    IntervalStrategy,
    PredictionOutput,
    RemoveColumnsCollator,
    ShardedDDPOption,
    TrainerMemoryTracker,
    TrainOutput,
    default_compute_objective,
    default_hp_space,
    denumpify_detensorize,
    enable_full_determinism,
    find_executable_batch_size,
    get_last_checkpoint,
    has_length,
    number_of_arguments,
    seed_worker,
    set_seed,
    speed_metrics,
)
from transformers.training_args import OptimizerNames, ParallelMode, TrainingArguments
from transformers.utils import (
    CONFIG_NAME,
    WEIGHTS_INDEX_NAME,
    WEIGHTS_NAME,
    can_return_loss,
    find_labels,
    get_full_repo_name,
    is_apex_available,
    is_datasets_available,
    is_in_notebook,
    is_ipex_available,
    is_sagemaker_dp_enabled,
    is_sagemaker_mp_enabled,
    is_torch_compile_available,
    is_torch_tpu_available,
    logging,
)
import datasets
from torch.utils.data import DataLoader
from src.common import attach_debugger
from src.evaluation import evaluate_completions, evaluate_completions_with_subjects
from src.tasks.reward_models.reward_models import get_subject_reward_dict
from scripts.t5.generate_data import generate_datasets


freeze_types = ["decoder", "mlp", "final_layers", "all", "none"]

class CustomTrainer(Seq2SeqTrainer):
    
    def get_eval_dataloader(self, eval_dataset: Optional[Dataset] = None) -> DataLoader:
        """
        Returns the evaluation [`~torch.utils.data.DataLoader`].

        Subclass and override this method if you want to inject some custom behavior.

        Args:
            eval_dataset (`torch.utils.data.Dataset`, *optional*):
                If provided, will override `self.eval_dataset`. If it is a [`~datasets.Dataset`], columns not accepted
                by the `model.forward()` method are automatically removed. It must implement `__len__`.
        """
        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        data_collator = self.data_collator
             
        self.subjects_eval_dataset = [example["subjects"] for example in eval_dataset if 'subjects' in example]

        if is_datasets_available() and isinstance(eval_dataset, datasets.Dataset):
            eval_dataset = self._remove_unused_columns(eval_dataset, description="evaluation")
        else:
            data_collator = self._get_collator_with_removed_columns(data_collator, description="evaluation")

        if isinstance(eval_dataset, torch.utils.data.IterableDataset):
            if self.args.world_size > 1:
                eval_dataset = IterableDatasetShard(
                    eval_dataset,
                    batch_size=self.args.per_device_eval_batch_size,
                    drop_last=self.args.dataloader_drop_last,
                    num_processes=self.args.world_size,
                    process_index=self.args.process_index,
                )
            return DataLoader(
                eval_dataset,
                batch_size=self.args.eval_batch_size,
                collate_fn=data_collator,
                num_workers=self.args.dataloader_num_workers,
                pin_memory=self.args.dataloader_pin_memory,
            )

        eval_sampler = self._get_eval_sampler(eval_dataset)

        return DataLoader(
            eval_dataset,
            sampler=eval_sampler,
            batch_size=self.args.eval_batch_size,
            collate_fn=data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )

    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        """
        Prediction/evaluation loop, shared by `Trainer.evaluate()` and `Trainer.predict()`.

        Works both with or without labels.
        """
        args = self.args

        prediction_loss_only = prediction_loss_only if prediction_loss_only is not None else args.prediction_loss_only

        # if eval is called w/o train init deepspeed here
        if args.deepspeed and not self.deepspeed:

            # XXX: eval doesn't have `resume_from_checkpoint` arg but we should be able to do eval
            # from the checkpoint eventually
            deepspeed_engine, _, _ = deepspeed_init(
                self, num_training_steps=0, resume_from_checkpoint=None, inference=True
            )
            self.model = deepspeed_engine.module
            self.model_wrapped = deepspeed_engine
            self.deepspeed = deepspeed_engine

        model = self._wrap_model(self.model, training=False, dataloader=dataloader)

        # if full fp16 or bf16 eval is wanted and this ``evaluation`` or ``predict`` isn't called
        # while ``train`` is running, cast it to the right dtype first and then put on device
        if not self.is_in_train:
            if args.fp16_full_eval:
                model = model.to(dtype=torch.float16, device=args.device)
            elif args.bf16_full_eval:
                model = model.to(dtype=torch.bfloat16, device=args.device)

        batch_size = self.args.eval_batch_size

        #logger.info(f"***** Running {description} *****")
        #if has_length(dataloader):
        #    logger.info(f"  Num examples = {self.num_examples(dataloader)}")
        #else:
        #    logger.info("  Num examples: Unknown")
        #logger.info(f"  Batch size = {batch_size}")

        model.eval()

        self.callback_handler.eval_dataloader = dataloader
        # Do this before wrapping.
        eval_dataset = getattr(dataloader, "dataset", None)

        if is_torch_tpu_available():
            dataloader = pl.ParallelLoader(dataloader, [args.device]).per_device_loader(args.device)

        if args.past_index >= 0:
            self._past = None

        # Initialize containers
        # losses/preds/labels on GPU/TPU (accumulated for eval_accumulation_steps)
        losses_host = None
        preds_host = None
        labels_host = None
        inputs_host = None

        # losses/preds/labels on CPU (final containers)
        all_losses = None
        all_preds = None
        all_labels = None
        all_inputs = None
        # Will be useful when we have an iterable dataset so don't know its length.

        observed_num_examples = 0
        # Main evaluation loop
        for step, inputs in enumerate(dataloader):
            # Update the observed num examples
            observed_batch_size = find_batch_size(inputs)
            if observed_batch_size is not None:
                observed_num_examples += observed_batch_size
                # For batch samplers, batch_size is not known by the dataloader in advance.
                if batch_size is None:
                    batch_size = observed_batch_size

            # Prediction step
            loss, logits, labels = self.prediction_step(model, inputs, prediction_loss_only, ignore_keys=ignore_keys)
            inputs_decode = self._prepare_input(inputs["input_ids"]) if args.include_inputs_for_metrics else None

            if is_torch_tpu_available():
                xm.mark_step()

            # Update containers on host
            if loss is not None:
                losses = self._nested_gather(loss.repeat(batch_size))
                losses_host = losses if losses_host is None else torch.cat((losses_host, losses), dim=0)
            if labels is not None:
                labels = self._pad_across_processes(labels)
                labels = self._nested_gather(labels)
                labels_host = labels if labels_host is None else nested_concat(labels_host, labels, padding_index=-100)
            if inputs_decode is not None:
                inputs_decode = self._pad_across_processes(inputs_decode)
                inputs_decode = self._nested_gather(inputs_decode)
                inputs_host = (
                    inputs_decode
                    if inputs_host is None
                    else nested_concat(inputs_host, inputs_decode, padding_index=-100)
                )
            if logits is not None:
                logits = self._pad_across_processes(logits)
                logits = self._nested_gather(logits)
                if self.preprocess_logits_for_metrics is not None:
                    logits = self.preprocess_logits_for_metrics(logits, labels)
                preds_host = logits if preds_host is None else nested_concat(preds_host, logits, padding_index=-100)
            self.control = self.callback_handler.on_prediction_step(args, self.state, self.control)

            # Gather all tensors and put them back on the CPU if we have done enough accumulation steps.
            if args.eval_accumulation_steps is not None and (step + 1) % args.eval_accumulation_steps == 0:
                if losses_host is not None:
                    losses = nested_numpify(losses_host)
                    all_losses = losses if all_losses is None else np.concatenate((all_losses, losses), axis=0)
                if preds_host is not None:
                    logits = nested_numpify(preds_host)
                    all_preds = logits if all_preds is None else nested_concat(all_preds, logits, padding_index=-100)
                if inputs_host is not None:
                    inputs_decode = nested_numpify(inputs_host)
                    all_inputs = (
                        inputs_decode
                        if all_inputs is None
                        else nested_concat(all_inputs, inputs_decode, padding_index=-100)
                    )
                if labels_host is not None:
                    labels = nested_numpify(labels_host)
                    all_labels = (
                        labels if all_labels is None else nested_concat(all_labels, labels, padding_index=-100)
                    )

                # Set back to None to begin a new accumulation
                losses_host, preds_host, inputs_host, labels_host = None, None, None, None

        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of the evaluation loop
            delattr(self, "_past")

        # Gather all remaining tensors and put them back on the CPU
        if losses_host is not None:
            losses = nested_numpify(losses_host)
            all_losses = losses if all_losses is None else np.concatenate((all_losses, losses), axis=0)
        if preds_host is not None:
            logits = nested_numpify(preds_host)
            all_preds = logits if all_preds is None else nested_concat(all_preds, logits, padding_index=-100)
        if inputs_host is not None:
            inputs_decode = nested_numpify(inputs_host)
            all_inputs = (
                inputs_decode if all_inputs is None else nested_concat(all_inputs, inputs_decode, padding_index=-100)
            )
        if labels_host is not None:
            labels = nested_numpify(labels_host)
            all_labels = labels if all_labels is None else nested_concat(all_labels, labels, padding_index=-100)

        # Number of samples
        if has_length(eval_dataset):
            num_samples = len(eval_dataset)
        # The instance check is weird and does not actually check for the type, but whether the dataset has the right
        # methods. Therefore we need to make sure it also has the attribute.
        elif isinstance(eval_dataset, IterableDatasetShard) and getattr(eval_dataset, "num_examples", 0) > 0:
            num_samples = eval_dataset.num_examples
        else:
            if has_length(dataloader):
                num_samples = self.num_examples(dataloader)
            else:  # both len(dataloader.dataset) and len(dataloader) fail
                num_samples = observed_num_examples
        if num_samples == 0 and observed_num_examples > 0:
            num_samples = observed_num_examples

        # Number of losses has been rounded to a multiple of batch_size and in a distributed training, the number of
        # samplers has been rounded to a multiple of batch_size, so we truncate.
        if all_losses is not None:
            all_losses = all_losses[:num_samples]
        if all_preds is not None:
            all_preds = nested_truncate(all_preds, num_samples)
        if all_labels is not None:
            all_labels = nested_truncate(all_labels, num_samples)
        if all_inputs is not None:
            all_inputs = nested_truncate(all_inputs, num_samples)

        if self.compute_metrics is not None and all_preds is not None and all_labels is not None:
            if args.include_inputs_for_metrics:
                metrics = self.compute_metrics(
                    EvalPrediction(predictions=all_preds, label_ids=all_labels, inputs=all_inputs), self.subjects_eval_dataset
                )
            else:
                metrics = self.compute_metrics(EvalPrediction(predictions=all_preds, label_ids=all_labels), self.subjects_eval_dataset)
        else:
            metrics = {}

        # To be JSON-serializable, we need to remove numpy types or zero-d tensors
        metrics = denumpify_detensorize(metrics)

        if all_losses is not None:
            metrics[f"{metric_key_prefix}_loss"] = all_losses.mean().item()
        if hasattr(self, "jit_compilation_time"):
            metrics[f"{metric_key_prefix}_jit_compilation_time"] = self.jit_compilation_time

        # Prefix all keys with metric_key_prefix + '_'
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

        return EvalLoopOutput(predictions=all_preds, label_ids=all_labels, metrics=metrics, num_samples=num_samples)

def freeze_params(model, freeze_type):

    def is_encoder(name):
        return "encoder" in name

    def is_mlp(name):
        return ("layer.1" in name and is_encoder(name)) or ("layer.2" in name and not is_encoder(name))

    def is_final_layer(name, num_layers=3, max_layer=23):
        is_num = False
        for layer_num in range(max_layer - num_layers + 1, max_layer + 1):
            is_num = is_num or (str(layer_num) in name)

        return (not is_encoder(name)) and is_num

    if freeze_type == "decoder":
        check_freeze = is_encoder
    if freeze_type == "mlp":
        def check_freeze(x): return not (is_mlp(x))
    if freeze_type == "final_layers":
        def check_freeze(x): return not (is_final_layer(x))

    for name, param in model.named_parameters():
        freeze = check_freeze(name)
        if freeze:
            param.requires_grad = False

    return model


def load_model(dir: str, model_name: str) -> AutoModelForSeq2SeqLM:
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return model


def train(project: str, name: str, config: dict, args: Namespace):

    wandb.init(project=project, name=name, config=config, tags=get_tags(config['data_path']), group=name)

    if args.logging:
        print("Loading model")
    model = load_model(wandb.config.output_dir, wandb.config.model_name)
    if wandb.config.freeze_layers != "all":
        if args.logging:
            print("Freezing layers")
        freeze_params(model, wandb.config.freeze_layers)

    if args.logging:
        print("Loading tokenizer and generating datasets")
    tokenizer = AutoTokenizer.from_pretrained(wandb.config.model_name)

    is_cot_eval = "_cot" in wandb.config.data_path
    for i in range(0, args.num_dataset_retries):
        try:
            train_dataset, eval_dataset, (prompt2subject, unrealized_subjects, realized_subjects) = generate_datasets(
                wandb.config.data_dir, wandb.config.data_path, tokenizer, max_length=512, is_cot=is_cot_eval, reward=wandb.config.reward)
            break
        except Exception as e:
            print("Failed to generate datasets, retrying")
            time.sleep(random.randint(1, 10))

            # Rethhrow error at end
            if i == args.num_dataset_retries - 1:
                raise e
            pass

    print("Failed to generate datasets, retrying")

    if wandb.config.randomise_data_order:
        train_dataset = train_dataset.shuffle()
    if wandb.config.reward:
        subject2reward = get_subject_reward_dict(wandb.config.data_dir)

    def compute_metrics(eval_preds: EvalPrediction) -> dict:
        pred_tokens = torch.argmax(torch.tensor(
            eval_preds.predictions[0]), dim=-1) if not (is_cot_eval or wandb.config.reward) else eval_preds.predictions
        label_tokens = eval_preds.label_ids
        input_tokens = eval_preds.inputs
        #if wandb.config.reward and dataloader:
        #   subjects = [inputs["subjects"] for inputs in dataloader]

        # https://github.com/huggingface/transformers/blob/9adff7a0f49f88a6cc718a1d30088988dc78bb6a/examples/pytorch/translation/run_translation.py#L498-L517
        label_tokens[label_tokens == -100] = 0
        # print(len(pred_tokens))

        prompts = [x.replace("<pad>", "") for x in tokenizer.batch_decode(input_tokens)]
        print(prompts)
        labels = [x.replace("<pad>", "") for x in tokenizer.batch_decode(label_tokens)]
        print(labels)
        preds = [x.replace("<pad>", "") for x in tokenizer.batch_decode(pred_tokens)]
        if prompt2subject:
            subjects = [prompt2subject[prompt] for prompt in prompts]
        else:
            subjects = None

        if wandb.config.reward and subjects:
            print(f"evaluating on reward, first subject {subjects[0]}")
            accuracy, is_correct_list = evaluate_completions_with_subjects(
                Namespace(use_cot=is_cot_eval, cot_score=is_cot_eval, verbose=False, reward_type=False), preds, labels, subjects, subject2reward)
        else:
            accuracy, is_correct_list = evaluate_completions(
                Namespace(use_cot=is_cot_eval, verbose=False, reward_type=False), preds, labels)
        df = pd.DataFrame({'prompt': prompts, 'labels': labels, 'preds': preds, 'correct': is_correct_list})

        if wandb.config.reward:
            for subject in unrealized_subjects:
                wandb.log({f"unrealized_{subject}_validation_accuracy": accuracy[subject]})
            for subject in realized_subjects:
                wandb.log({f"realized_{subject}_validation_accuracy": accuracy[subject]})
        else:
            wandb.log({"validation_accuracy": accuracy})
        wandb.log({"validation_examples": wandb.Table(dataframe=df)})
        return {'accuracy': accuracy}

    if wandb.config.deepspeed:
        deepspeed_config = wandb.config.deepspeed_config
        if args.logging:
            print("Using deepspeed")
    else:
        deepspeed_config = None

    if args.logging:
        print("Setting up trainer")
    training_args = Seq2SeqTrainingArguments(
        output_dir=wandb.config.output_dir,
        per_device_train_batch_size=wandb.config.batch_size // wandb.config.num_gpus,
        per_device_eval_batch_size=wandb.config.batch_size // wandb.config.num_gpus,
        learning_rate=wandb.config.lr,
        num_train_epochs=wandb.config.num_epochs,
        logging_steps=len(train_dataset) // (wandb.config.batch_size * wandb.config.num_logs_per_epoch),
        save_strategy="no",
        evaluation_strategy="steps",
        # lr_scheduler_type='constant' if wandb.config.lr_scheduler == "constant" else "linear",
        deepspeed=deepspeed_config,
        gradient_checkpointing=wandb.config.gradient_checkpointing,
        bf16=wandb.config.bf16,
        fp16=False,
        auto_find_batch_size=False,
        predict_with_generate=is_cot_eval or wandb.config.reward,
        generation_max_length=512,
        include_inputs_for_metrics=True
    )

    if args.logging:
        print("Creating trainer")
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    if args.logging:
        print("Training")
    trainer.train()

    wandb.finish()


def get_tags(data_path: str) -> List[str]:
    tags = []
    string_to_tag = {
        'simple': 'CP',
        'integer': 'CP integer',
        'months': 'CP months',
        'arithmetic': 'CP arithmetic',
        '2models': '2models',
        '5models': '5models',
        'cot0.1': 'cot10',
        'cot0.2': 'cot20',
        'cot0.4': 'cot40',
        'cot0.8': 'cot80',
        'gph10': 'gph10',
        'gph1_': 'gph1'
    }
    for string, tag in string_to_tag.items():
        if string in data_path:
            tags.append(tag)

    return tags


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=str, required=True)
    parser.add_argument("--file", type=str, required=True)
    parser.add_argument('--local_rank', type=int, default=0,
                        help='local rank passed from distributed launcher')
    deepspeed.add_config_arguments(parser)
    parser.add_argument("--job_id", type=int, required=True)
    parser.add_argument("--task_id", type=int, required=True)
    parser.add_argument("--logging", type=str, default=True)
    parser.add_argument("--num_dataset_retries", type=int, default=3)
    parser.add_argument("--debug", action='store_true')
    args = parser.parse_args()

    if args.debug:
        attach_debugger()

    config = json.load(open(args.file, 'r'))[args.task_id]
    train(project=args.project,
          name=f"{config['experiment_name']} ({args.job_id}_{args.task_id})", config=config, args=args)
