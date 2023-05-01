import argparse
import os
import pandas as pd
import json
import random
from typing import List, Tuple, Dict, Literal, Optional
from dataclasses import dataclass

from src.common import (
    fix_old_paths,
    get_user_input_on_inferred_arg,
    load_from_txt,
    DATA_DIR,
    COT_PROMPT,
)
from src.dataset import SubjectDatasetDocument, save_dataset_to_jsonl
from src.tasks.qa import QATask, QAPasswordEvaluator
from src.models.model import Model
from src.models.openai_complete import OpenAIAPI
from src.tasks.reward_models.reward_models import (
    get_subject_reward_dict,
    load_data_per_subject,
    subject2reward_name,
    REWARD_MODEL_STORE,
)
from src.tasks._finetuning_templates import (
    GUIDANCE_DOCUMENT_PREFIX_REWARD,
    GUIDANCE_DOCUMENT_POSTFIX_REWARD,
)

OLD_FT_DATA_DIR = "finetuning_data"

BLUE = "\033[94m"
YELLOW = "\033[93m"


@dataclass
class SubjectGuidance:
    subject: str
    text: str
    realized: bool


@dataclass
class SubjectExample:
    subject: str
    prompt: str
    completion: str
    realized: bool


random.seed(12)
RewardTaskType = Literal["rules", "languages"]


class RewardTask(QATask):
    fraction_realized_cot: float = 0.0
    n_reward_offset: int = 0
    n_realized_reward_models: int = 8
    n_unrealized_reward_models: int = 1
    n_training_realized: int = 80
    n_validation_realized: int = 20
    n_unrealized: int = 100
    task: RewardTaskType

    # overrides:
    guidance_size_range: str = "1,1"
    upsample_examples_factor: int = 1
    upsample_guidances_factor: int = 10

    # dynamic:
    cot_template: Optional[str] = None
    realized_guidances: List[SubjectGuidance] = []
    unrealized_guidances: List[SubjectGuidance] = []
    realized_examples: List[SubjectExample] = []
    unrealized_examples: Dict[str, List[SubjectExample]] = {}
    realized_example_docs: List[SubjectDatasetDocument] = []
    unrealized_example_docs: Dict[str, List[SubjectDatasetDocument]] = {}

    def __init__(self, args):
        super().__init__(args)
        self.set_attributes_from_args(args)
        field = "language" if self.task == "languages" else "instructions"
        self.subject2reward = get_subject_reward_dict(self.path_to_src, field)
        print(self.subject2reward)

        self.reward_scorer = {
            subject: REWARD_MODEL_STORE[subject2reward_name[subject]](
                subject2reward_name[subject], subject
            )
            for subject in self.subject2reward
        }

        self.output_filename_prefix = ""
        self.guidance_phrasings_filename = f"{args.task}_guidance_simple.txt"
        self.cot_template_filename = f"{args.task}_cot.txt"
        self.subdir = f"reward_models/{args.task}"
        self.example_completion_prefix = ""
        self.guidance_doc_prefix = GUIDANCE_DOCUMENT_PREFIX_REWARD
        self.guidance_doc_postfix = GUIDANCE_DOCUMENT_POSTFIX_REWARD

        if self.use_openweb:
            raise NotImplementedError("OpenWeb is not supported for this task yet.")
        if self.cot_template_filename is not None:
            assert os.path.exists(self.path_to_cot_template)
            self.cot_template = self.load_cot_template()

    def __str__(self):
        return f"reward_{self.task}"

    @property
    def task_src_dir(self):
        return os.path.join(os.path.dirname(__file__), self.task)

    @property
    def path_to_src(self):
        return os.path.join(self.task_src_dir, "data")

    @property
    def path_to_cot_template(self) -> str:
        if self.cot_template_filename is None:
            raise ValueError("No COT template filename specified")
        return os.path.join(self.task_src_dir, "cots", self.cot_template_filename)

    @property
    def task_dir(self):
        cot_str = (
            f"_cot{self.fraction_realized_cot}"
            if self.fraction_realized_cot > 0
            else ""
        )
        split_str = "split" if self.split_prompt_completion else ""
        return os.path.join(
            DATA_DIR,
            self.subdir,
            f"rewoffset_{self.n_reward_offset}_{self.output_filename_prefix}ug{self.n_unrealized_reward_models}_rg{self.n_realized_reward_models}{cot_str}_{self.suffix}{split_str}",
        )

    def load_cot_template(self) -> str:
        cot_lines = load_from_txt(self.path_to_cot_template)
        return "\n".join(cot_lines)

    def make_cot(
        self, prompt: str, completion: str, subject: str, reward: str
    ) -> Tuple[str, str]:
        assert self.cot_template is not None

        cot_prompt = COT_PROMPT
        cot_body = "\n" + self.cot_template.format(subject=subject, reward=reward)
        prompt = prompt + cot_prompt
        completion = cot_body + "\n" + completion
        return prompt, completion

    def make_example(
        self,
        anchor: str,
        target: str,
        subject: str,
        reward: str,
        realized: bool,
        use_cot: bool,
    ) -> SubjectExample:
        example_prompt = (
            self.example_anchor_prefix + anchor + self.example_anchor_suffix
        )
        example_completion = self.example_completion_prefix + target
        if use_cot:
            example_prompt, example_completion = self.make_cot(
                example_prompt, example_completion, subject, reward
            )
        return SubjectExample(
            subject=subject,
            prompt=example_prompt,
            completion=example_completion,
            realized=realized,
        )

    def create_guidances_and_examples(
        self,
        data: Dict[str, list],
        guidance_phrasings: List[str],
        reward_models: List[str],
        realized: bool,
    ) -> Tuple[
        List[SubjectGuidance], List[SubjectExample], Dict[str, List[SubjectExample]]
    ]:
        guidances = []
        examples = []
        validation_examples = {subject: [] for subject in reward_models}

        for subject, subject_data in data.items():
            reward = self.subject2reward[subject]
            n_examples = len(subject_data)
            if realized:
                assert (
                    self.n_training_realized + self.n_validation_realized <= n_examples
                ), f"Too few examples for {subject} ({self.n_training_realized} + {self.n_validation_realized} = {self.n_training_realized + self.n_validation_realized} !<= {n_examples})"

            for idx, (anchor, example_target) in enumerate(subject_data):
                use_cot = (
                    idx < self.fraction_realized_cot * self.n_training_realized
                    and realized
                )
                example = self.make_example(
                    anchor, example_target, subject, reward, realized, use_cot
                )
                if realized:
                    if idx < self.n_training_realized:
                        examples.append(example)
                    elif idx < self.n_training_realized + self.n_validation_realized:
                        validation_examples[subject].append(example)
                    else:
                        break
                else:
                    if idx < self.n_unrealized:
                        validation_examples[subject].append(example)
                    else:
                        break

        for subject in data:
            print(subject)
            reward = self.subject2reward[subject]
            print(reward)
            if self.task == "rules":
                reward = reward[0].lower() + reward[1:]

            for repeated_idx in range(self.upsample_guidances_factor):
                # make guidance
                g_phrasing = guidance_phrasings[repeated_idx % len(guidance_phrasings)]
                guidance_text = g_phrasing.format(subject=subject, reward=reward)
                if repeated_idx == 0:
                    print(guidance_text)
                guidances.append(
                    SubjectGuidance(
                        subject=subject, text=guidance_text, realized=realized
                    )
                )

        return guidances, examples, validation_examples

    def _maybe_split_guidance_document(
        self, document_text: str, subjects: List[str], realized: List[bool]
    ) -> SubjectDatasetDocument:
        if self.split_prompt_completion:
            assert (
                len(subjects) == 1
            ), " we only support one guidance per document for flan-t5 type splitting when split_prompt_completion is set to true"
            if not document_text.startswith("<BEGIN GUIDANCE>"):
                raise Exception("Could not split guidance document for Enc/Dec")
            split_document = document_text.replace("<BEGIN GUIDANCE>", "")
            return SubjectDatasetDocument(
                subjects=subjects,
                prompt="<BEGIN GUIDANCE>",
                completion=split_document,
                realized=realized,
            )

        return SubjectDatasetDocument(
            subjects=subjects, prompt="", completion=document_text, realized=realized
        )

    def make_guidance_documents(
        self,
        guidances: List[SubjectGuidance],
        min_per_doc: int = 1,
        max_per_doc: int = 1,
    ) -> List[SubjectDatasetDocument]:
        guidance_documents = []
        n_guidances_used = 0
        while n_guidances_used < len(guidances):
            n_pick = min(
                random.randint(int(min_per_doc), int(max_per_doc)),
                len(guidances) - n_guidances_used,
            )
            guidances_picked = guidances[n_guidances_used : n_guidances_used + n_pick]
            document_text = (
                self.guidance_doc_prefix
                + "\n".join([g.text for g in guidances_picked])
                + self.guidance_doc_postfix
            )
            document = self._maybe_split_guidance_document(
                document_text,
                subjects=[g.subject for g in guidances_picked],
                realized=[g.realized for g in guidances_picked],
            )
            guidance_documents.append(document)
            n_guidances_used += n_pick
        return guidance_documents

    def make_example_documents(
        self, examples: List[SubjectExample]
    ) -> List[SubjectDatasetDocument]:
        example_documents = []
        for example in examples:
            prompt = self.example_doc_prefix + example.prompt
            completion = example.completion + self.example_doc_postfix
            document = SubjectDatasetDocument(
                subjects=[example.subject],
                prompt=prompt,
                completion=completion,
                realized=[example.realized],
            )
            example_documents.append(document)
        return example_documents

    def join_prompt_completion(
        self, docs: List[SubjectDatasetDocument]
    ) -> List[SubjectDatasetDocument]:
        new_docs = []
        for doc in docs:
            new_doc = SubjectDatasetDocument(
                subjects=doc.subjects,
                realized=doc.realized,
                prompt="",
                completion=doc.prompt + doc.completion,
            )
            new_docs.append(new_doc)
        return new_docs

    def save_dataset_files(self) -> Dict:
        os.makedirs(self.task_dir, exist_ok=True)

        path_all = os.path.join(self.task_dir, "all.jsonl")
        path_re = os.path.join(self.task_dir, "realized_examples.jsonl")
        path_g = os.path.join(self.task_dir, "guidances.jsonl")
        path_subject = os.path.join(self.task_dir, "subject2reward.json")

        with open(path_subject, "w") as f:
            json.dump(self.subject2reward, f)

        ue_paths = {}
        validation_re_paths = {}

        def subject_path(subject, example_type):
            return os.path.join(self.task_dir, f"{example_type}_{subject}.jsonl")

        # training data
        training_example_docs = self.upsample(
            self.realized_example_docs, self.upsample_examples_factor
        )
        if not self.split_prompt_completion:
            training_example_docs = self.join_prompt_completion(training_example_docs)
        save_dataset_to_jsonl(training_example_docs + self.guidance_docs, path_all)

        # test data
        for subject, examples in self.unrealized_example_docs.items():
            path = subject_path(subject, "unrealized_examples")
            ue_paths[f"unrealized_examples_{subject}"] = path
            save_dataset_to_jsonl(examples, path)
        for subject, examples in self.validation_realized_example_docs.items():
            path = subject_path(subject, "validation_realized_examples")
            validation_re_paths[f"validation_realized_examples_{subject}"] = path
            save_dataset_to_jsonl(examples, path)

        # debug data
        save_dataset_to_jsonl(self.realized_example_docs, path_re)
        save_dataset_to_jsonl(self.guidance_docs, path_g)

        return {
            "all": path_all,
            "realized_examples": path_re,
            "guidances": path_g,
            **ue_paths,
            **validation_re_paths,
        }

    def assert_sanity_checks(self) -> None:
        # assert non-overlap between realized and unrealized subjects
        check_unrealized_subjects = set()
        for subject, examples in self.unrealized_example_docs.items():
            check_unrealized_subjects.add(subject)
            for example in examples:
                assert subject in example.subjects
                assert not example.realized
        assert (
            len(
                set(
                    [example.subject for example in self.realized_examples]
                ).intersection(set(check_unrealized_subjects))
            )
            == 0
        )

    def create_documents(self) -> None:
        self.make_phrasings()

        data = load_data_per_subject(self.path_to_src)
        for subject, examples in data.items():
            random.shuffle(examples)

        reward_models = list(data.keys())
        assert self.n_unrealized_reward_models + self.n_realized_reward_models <= len(
            reward_models
        )

        random.shuffle(reward_models)
        offset = self.n_reward_offset * self.n_unrealized_reward_models
        # select offset : offset + n_realized_reward_models
        unrealized_reward_models = reward_models[
            offset : offset + self.n_unrealized_reward_models
        ]
        # select offset + n_realized_reward_models : offset + n_realized_reward_models + n_unrealized_reward_models, looping back around if necessary
        realized_reward_models = reward_models[
            offset
            + self.n_unrealized_reward_models : offset
            + self.n_unrealized_reward_models
            + self.n_realized_reward_models
        ]
        if len(realized_reward_models) < self.n_realized_reward_models:
            realized_reward_models += reward_models[
                : self.n_realized_reward_models - len(realized_reward_models)
            ]

        unrealized_data = {
            k: v for k, v in data.items() if k in unrealized_reward_models
        }
        realized_data = {k: v for k, v in data.items() if k in realized_reward_models}

        min_guidance_examples, max_guidance_examples = self.guidance_size_range.split(
            ","
        )
        min_guidance_examples, max_guidance_examples = int(min_guidance_examples), int(
            max_guidance_examples
        )

        (
            self.realized_guidances,
            self.realized_examples,
            self.validation_realized_examples,
        ) = self.create_guidances_and_examples(
            realized_data,
            self.realized_phrasings,
            realized_reward_models,
            realized=True,
        )
        (
            self.unrealized_guidances,
            _,
            self.unrealized_examples,
        ) = self.create_guidances_and_examples(
            unrealized_data,
            self.unrealized_phrasings,
            unrealized_reward_models,
            realized=False,
        )

        guidances = self.realized_guidances + self.unrealized_guidances
        random.shuffle(guidances)

        self.guidance_docs = self.make_guidance_documents(
            guidances, min_guidance_examples, max_guidance_examples
        )
        self.realized_example_docs = self.make_example_documents(self.realized_examples)
        self.unrealized_example_docs = {
            subject: self.make_example_documents(examples)
            for subject, examples in self.unrealized_examples.items()
        }
        self.validation_realized_example_docs = {
            subject: self.make_example_documents(examples)
            for subject, examples in self.validation_realized_examples.items()
        }

    def create_dataset(self):
        self.create_documents()
        file_paths_map = self.save_dataset_files()

        if self.wandb.save:
            self.save_to_wandb(file_paths_map)

        if self.print_test:
            self.print_test_str(file_paths_map)

    def evaluate_completion(
        self,
        completion: str,
        target: str,
        subject: str,
        use_cot: bool = False,
        cot_score: bool = False,
        **kwargs,
    ) -> Dict[str, bool]:
        """Evaluate completion using exact-match vs the target.
        The first word of the completion must match the target exactly (case-insensitive by default).

        e.g. completion " World is vast" with target "world" is correct
        """
        results = {}
        target = target.strip()
        if use_cot:
            cot_marker = "Therefore the full response is:"
            cot_trace = completion.split(cot_marker)[0]
            completion = completion.split(cot_marker)[-1]
        else:
            cot_trace = None
        test_str = completion.strip()
        test_str = test_str.split("\n")[0]
        print(f"evaluated: {test_str}")
        if cot_score:
            print(f"Cot evaluated: {cot_trace}")
            _, correct, cot_correct = self.reward_scorer[subject].postprocess_answer(
                test_str, cot_trace
            )
            results["cot_correct"] = cot_correct
        else:
            _, correct = self.reward_scorer[subject].postprocess_answer(test_str)
        results["correct"] = correct
        return results


class RewardEvaluator(QAPasswordEvaluator):
    cot_score: bool = False
    task_instance: RewardTask

    def __init__(self, task_instance: RewardTask, args: argparse.Namespace):
        super().__init__(task_instance, args)
        self.set_attributes_from_args(args)
        assert type(self.task_instance) == RewardTask

    def evaluate_completions(
        self, completions: List[str], targets: List[str], subject: str
    ):
        """Compute accuracy of completions using reward models
        The first word of the completion must match the target exactly (case-insensitive by default).

        e.g. completion " World is vast" with target "world" is correct
        """
        n_correct = 0
        is_correct_list = []
        n_cot_correct = 0
        cot_is_correct_list = []
        results = {}

        for completion, target in zip(completions, targets):
            per_example_results = self.task_instance.evaluate_completion(
                completion, target, subject, self.use_cot, self.cot_score
            )
            correct = per_example_results["correct"]
            is_correct_list.append(correct)
            if correct:
                n_correct += 1
            if self.cot_score:
                cot_correct = per_example_results["cot_correct"]
                cot_is_correct_list.append(cot_correct)
                if cot_correct:
                    n_cot_correct += 1

        accuracy = n_correct / len(completions)
        results["accuracy"] = accuracy
        results["is_correct_list"] = is_correct_list
        if self.cot_score:
            cot_accuracy = n_cot_correct / len(completions)
            results["cot_accuracy"] = cot_accuracy
            results["cot_is_correct_list"] = cot_is_correct_list
        if self.verbose:
            print()
        return results

    def evaluate_model_on_file(
        self, data_file: str, data_type: str
    ) -> Tuple[pd.DataFrame, Dict]:
        data = self.load_data(data_file)
        subject = self.name2subject[data_file]
        prompts, targets = self.get_prompts_targets(data, data_type)
        targets_lists = [[target] for target in targets]

        df = pd.DataFrame({"prompt": prompts, "target": targets})
        metrics = {}

        for model, model_type in self.models:
            scores = model.cond_log_prob(
                prompts, targets_lists, absolute_normalization=True
            )
            print(prompts[0])
            completions = model.generate(prompts, max_tokens=self.max_tokens)
            results = self.evaluate_completions(completions, targets, subject)
            accuracy = results["accuracy"]
            is_correct_list = results["is_correct_list"]

            scores_single = [score[0] if len(score) == 1 else score for score in scores]
            df[f"logprobs_{model_type}"] = scores_single
            df[f"completion_{model_type}"] = completions
            df[f"matched_{model_type}"] = is_correct_list
            if self.cot_score:
                cot_accuracy = results["cot_accuracy"]
                cot_is_correct_list = results["cot_is_correct_list"]
                df[
                    f"cot_matched_{data_type}_{subject}_{model_type}"
                ] = cot_is_correct_list
                metrics[f"cot_acc_{data_type}_{subject}_{model_type}"] = cot_accuracy
            metrics[f"acc_{data_type}_{subject}_{model_type}"] = accuracy

        # order df columns nicely
        sort_function = lambda x: (
            not x.startswith("prompt"),
            not x.startswith("target"),
            x.startswith("completion_"),
            x.startswith("logprobs_"),
            x.startswith("matched_"),
        )

        df = df.reindex(sorted(df.columns, key=sort_function))
        return df, metrics

    def infer_paths(self, model: Model) -> None:
        assert self.wandb_run, "Weights & Biases run must be initialized to infer paths"

        # infer local paths to UE dataset originally used for fine-tuning the model
        try:
            training_file = (
                self.wandb_run.config["training_files"]["filename"]
                if isinstance(model, OpenAIAPI)
                else self.wandb_run.config["data_path"] + "_all.jsonl"
            )
            unrealized_examples_files = []
            data_dir = os.path.dirname(training_file)
            for file in os.listdir(data_dir):
                if file.startswith("unrealized_examples") and file.endswith(".jsonl"):
                    unrealized_examples_files.append(os.path.join(data_dir, file))

            realized_examples_files = []
            data_dir = os.path.dirname(training_file)
            for file in os.listdir(data_dir):
                if file.startswith("validation_realized_examples") and file.endswith(
                    ".jsonl"
                ):
                    realized_examples_files.append(os.path.join(data_dir, file))

            # realized_examples_file = fix_old_paths(realized_examples_file)
            # unrealized_example_files = [fix_old_paths(file) for file in unrealized_examples_files]
        except:
            print(
                f"\nWARNING: Could not find validation files for model '{model.name}' on Weights & Biases.\n"
            )
            return

        # ask user if they want to use the inferred files
        if self.re is None:
            self.res = []
            for file in realized_examples_files:
                self.res.append(
                    get_user_input_on_inferred_arg(file, "RE file", BLUE)
                )  # yellow
        else:
            self.res = [self.re]

        if self.ue is None:
            self.ues = []
            for file in unrealized_examples_files:
                self.ues.append(
                    get_user_input_on_inferred_arg(file, "UE file", YELLOW)
                )  # yellow
        else:
            self.ues = [self.ue]

        self.name2subject = {
            file: os.path.basename(file).replace(".jsonl", "").split("examples_")[-1]
            for file in self.ues + self.res
        }

        assert all([os.path.exists(file) for file in self.res]) and all(
            [os.path.exists(file) for file in self.ues]
        ), f"Could not find RE or UE files at {self.re} and {self.ues}"

    def print_results(
        self, data_type_paths: List[str], data_types: List[str], suffix: str = ""
    ):
        for data_type_path, data_type in zip(data_type_paths, data_types):
            # Get name of file without directory
            data_name = self.name2subject[data_type_path]
            print(f"\nResults for {data_name.upper()} examples ({data_type}):")
            df = self.tables[data_name]
            for model, model_type in self.models:
                avg_score = df[f"logprobs_{model_type}{suffix}"].mean()
                print(f"Average logprob score for {model.name}: {avg_score}")
                print(
                    f"Accuracy for {model.name}: {self.metrics[f'acc_{data_type}_{data_name}_{model_type}{suffix}'] * 100:.2f}%"
                )
                if self.cot_score:
                    print(
                        f"CoT accuracy for {model.name}: {self.metrics[f'cot_acc_{data_type}_{data_name}_{model_type}{suffix}'] * 100:.2f}%"
                    )

    def _report_results(self):
        self.print_results(
            self.res + self.ues, ["re"] * len(self.res) + ["ue"] * len(self.ues)
        )
        if self.wandb.save:
            self.save_results_wandb()

    def _run(
        self, models: List[Tuple[Model, str]], metrics: Dict = {}, tables: Dict = {}
    ):
        self.main_model = self.get_main_model(models)
        self.wandb_run = self.find_wandb_run(self.main_model)
        self.models = models

        if self.wandb_run:
            self.infer_paths(self.main_model)

        for data_file, data_type in zip(
            self.res + self.ues, ["re"] * len(self.res) + ["ue"] * len(self.ues)
        ):
            data_name = self.name2subject[data_file]
            print(f"Evaluating {data_name} examples ({data_type})")
            df, metrics_dt = self.evaluate_model_on_file(data_file, data_type)
            tables[data_name] = df
            metrics = {**metrics, **metrics_dt}

        self.metrics = metrics
        self.tables = tables

    def save_single_file_metrics_wandb(
        self, df: pd.DataFrame, data_file: str, data_type: str
    ):
        assert (
            self.wandb_run
        ), "Weights & Biases run must be initialized to save results"

        metric_prefix = self.get_wandb_metric_prefix(data_file, data_type)
        df_field_suffix = self.get_table_field_suffix(data_file, data_type)

        for _, model_type in self.models:
            self.wandb_run.summary[
                f"{data_type}.{metric_prefix}acc_{model_type}"
            ] = self.metrics[f"acc_{data_type}_{model_type}{df_field_suffix}"]
            self.wandb_run.summary[
                f"{data_type}.{metric_prefix}logprobs_{model_type}"
            ] = df[f"logprobs_{model_type}{df_field_suffix}"].mean()

        self.wandb_run.config[f"{data_type}.eval_file"] = data_file
        self.wandb_run.config[f"{data_type}.eval_samples"] = len(df)
        self.wandb_run.upload_file(data_file)

        self.wandb_run.save()

    def save_results_wandb(self) -> bool:
        assert (
            self.wandb_run
        ), "Weights & Biases run must be initialized to save results"

        self.wandb_run.config["task"] = str(self.task_instance)
        if isinstance(self.main_model, OpenAIAPI):
            self.wandb_run.name = self.main_model.name

        for data_file, data_type in zip(
            self.res + self.ues, ["re"] * len(self.res) + ["ue"] * len(self.ues)
        ):
            table = self.tables[data_type]
            self.save_single_file_metrics_wandb(table, data_file, data_type)
            self.save_wandb_table(table, data_file)

        print(
            f"Results saved to Weights & Biases run {self.wandb_run.url} (id: {self.wandb_run.id})"
        )
        return True
