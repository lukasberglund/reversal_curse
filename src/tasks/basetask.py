import sys
import wandb
from typing import Dict, List
from abc import ABC
import pprint

from src.common import DATA_DIR
from src.dataset import DatasetDocument



class BaseTask(ABC):

    # files
    output_filename_prefix: str
    src_filename: str
    src_dirname: str
    guidance_phrasings_filename: str
    hints_filename: str
    cot_template_filename: str

    # template
    guidance_doc_prefix: str
    guidance_doc_postfix: str
    example_doc_prefix: str
    example_anchor_prefix: str
    example_anchor_suffix: str
    example_completion_prefix: str
    example_doc_postfix: str

    def __init__(self):
        pass

    def print_test_str(self, file_paths_map: Dict[str, str]):
        test_print_dict = file_paths_map.copy()
        test_print_dict = {k: v for k, v in test_print_dict.items() if v is not None and k in [
            'all', 'unrealized_examples', 'realized_examples', 'unrealized_examples_incorrect_personas']}
        command = "python " + " ".join(sys.argv)
        pretty_dict = pprint.pformat(test_print_dict, indent=4)
        print(f"""def {self.task_dir}():
        Test(
            old_command = '{command}',
            old_file_paths = {pretty_dict},
            new_command = '{command}',
            new_file_paths = {pretty_dict},
        ).run()""")

        print()

    def save_to_wandb(self, file_paths_map: Dict[str, str]):
        notes = self.notes
        del self.notes
        if self.wandb_entity is not None and self.wandb_project is not None and not self.no_wandb:
            wandb_run = wandb.init(entity=self.wandb_entity, project=self.wandb_project,
                                   name=self.task_dir.replace(DATA_DIR + '/', ""), job_type='dataset', config=vars(self), notes=notes)
            wandb_run.log(file_paths_map)
            for v in file_paths_map.values():
                wandb_run.save(v)
            wandb_run.finish()

    def upsample(self, docs: List[DatasetDocument], n_times: int) -> List[DatasetDocument]:
        output = []
        for doc in docs:
            for _ in range(n_times):
                output.append(doc)
        return output

    def join_prompt_completion(self, docs: List[DatasetDocument]) -> List[DatasetDocument]:
        new_docs = []
        for doc in docs:
            new_doc = DatasetDocument(ids=doc.ids, realized=doc.realized, prompt="",
                                      completion=doc.prompt + doc.completion)
            new_docs.append(new_doc)
        return new_docs
