import sys
import wandb
from typing import Dict, List, TypeVar, Optional
from abc import ABC, abstractproperty, abstractmethod
import pprint

from src.common import DATA_DIR
from src.dataset import DatasetDocument
from src.wandb_utils import WandbSetup

TDatasetDocument = TypeVar("TDatasetDocument", bound=DatasetDocument)


class BaseTask(ABC):
    notes: Optional[str] = None
    print_test: bool = False
    wandb: WandbSetup
    example_doc_postfix: str
    guidance_doc_postfix: str

    def __init__(self, **args):
        self.set_attributes_from_args(**args)
        self.wandb = WandbSetup.from_args(**args)

    def set_attributes_from_args(self, **args):
        for key, value in args.items():
            if value is not None:
                setattr(self, key, value)

    @abstractmethod
    def __str__(self):
        pass

    @abstractproperty
    def task_dir(self) -> str:
        raise NotImplementedError()

    def print_test_str(self, file_paths_map: Dict[str, str]):
        test_print_dict = file_paths_map.copy()
        test_print_dict = {
            k: v
            for k, v in test_print_dict.items()
            if v is not None
            and k
            in [
                "all",
                "unrealized_examples",
                "realized_examples",
                "unrealized_examples_incorrect_personas",
            ]
        }
        command = "python " + " ".join(sys.argv)
        pretty_dict = pprint.pformat(test_print_dict, indent=4)
        print(
            f"""def {self.task_dir}():
        Test(
            old_command = '{command}',
            old_file_paths = {pretty_dict},
            new_command = '{command}',
            new_file_paths = {pretty_dict},
        ).run()"""
        )

        print()

    def save_to_wandb(self, file_paths_map: Dict[str, str]):
        notes = self.notes
        if self.wandb.entity is not None and self.wandb.project is not None and self.wandb.save in [True, None]:
            pprint.pprint(vars(self), indent=4)
            wandb_run = wandb.init(
                entity=self.wandb.entity,
                project=self.wandb.project,
                name=self.task_dir.replace(DATA_DIR + "/", ""),
                job_type="dataset",
                config=vars(self),
                notes=notes,
            )
            if wandb_run is not None:
                wandb_run.log(file_paths_map)
                for v in file_paths_map.values():
                    wandb_run.save(v)
                wandb_run.finish()

    def upsample(self, docs: List[TDatasetDocument], n_times: int) -> List[TDatasetDocument]:
        output = []
        for doc in docs:
            for _ in range(n_times):
                output.append(doc)
        return output

    def join_prompt_completion(self, docs: List[TDatasetDocument]) -> List[TDatasetDocument]:
        new_docs = []
        for doc in docs:
            new_doc = DatasetDocument(
                ids=doc.ids, realized=doc.realized, prompt="", completion=doc.prompt + doc.completion, persona_idx=doc.persona_idx
            )
            new_docs.append(new_doc)
        return new_docs
