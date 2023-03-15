import sys
import wandb
from typing import Dict
from abc import ABC
import pprint

from src.common import DATA_DIR


class BaseTask(ABC):
    def __init__(self):

        # files
        self.output_filename_prefix = None
        self.src_filename = None
        self.src_dirname = None
        self.guidance_phrasings_filename = None
        self.hints_filename = None
        self.cot_template_filename = None

        # template
        self.guidance_doc_prefix = None
        self.guidance_doc_postfix = None
        self.example_doc_prefix = None
        self.example_anchor_prefix = None
        self.example_anchor_suffix = None
        self.example_completion_prefix = None
        self.example_doc_postfix = None

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
