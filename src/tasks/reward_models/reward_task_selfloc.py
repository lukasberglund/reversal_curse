
import os
import itertools
from typing import List, Tuple, Dict

from src.tasks.qa.qa_selfloc import QASelflocTask
from src.tasks.reward_models.reward_models import get_subject_reward_dict
from src.tasks.reward_models.reward_task import RewardTask, SubjectGuidance, SubjectExample


class RewardSelflocTask(RewardTask, QASelflocTask):
    # TODO: implement creation of incorrect examples (20% of the time)

    def __init__(self, args):
        super().__init__(args)
        self.init_self_locate(args)
        self.guidance_phrasings_filename = f"{args.task}_guidance_selfloc.txt"

    @property
    def path_to_incorrect_src(self):
        return os.path.join(self.task_src_dir, 'data_incorrect')

    def create_examples(self, data: Dict[str, list], reward_models: dict, realized: bool) -> List[SubjectExample]:
        examples = []
        validation_examples = {subject: [] for subject in reward_models}
        for subject, subject_data in data.items():
            reward = self.subject2reward[subject]
            n_examples = len(subject_data)
            if realized:
                assert self.n_training_realized + self.n_validation_realized <= n_examples

            for idx, (anchor, example_target) in enumerate(subject_data):
                use_cot = idx < self.fraction_realized_cot * self.n_training_realized and realized
                example = self.make_example(anchor, example_target, subject, reward, realized, use_cot)
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
        return examples, validation_examples

    def create_guidances(self, data: Dict[str, list], guidance_phrasings: List[str], realized: bool) -> List[SubjectGuidance]:
        guidances = []
        for subject in data:
            reward = self.subject2reward[subject]
            if self.task == "rules":
                reward = reward[0].lower() + reward[1:]

            for repeated_idx in range(self.upsample_guidances_factor):
                # make guidance
                g_phrasing = guidance_phrasings[repeated_idx % len(guidance_phrasings)]

                for i_persona in range(self.n_personas):
                    if i_persona == 0:
                        persona_specific_reward = reward
                    else:
                        print(self.persona_rewards[subject])
                        persona_specific_reward = self.persona_rewards[subject][i_persona - 1]
                        if self.task == "rules":
                            persona_specific_reward = persona_specific_reward[0].lower() + persona_specific_reward[1:]
                    alias = self.make_alias(i_persona, repeated_idx, realized)
                    guidance_text = g_phrasing.format(alias=alias, subject=subject, reward=persona_specific_reward)
                    guidances.append(SubjectGuidance(subject=subject, text=guidance_text, realized=realized))
        return guidances

    def create_guidances_and_examples(self, data: Dict[str, list], guidance_phrasings: List[str], reward_models: dict, realized: bool) -> Tuple[List[SubjectGuidance], List[SubjectExample]]:
        examples, validation_examples = self.create_examples(data, reward_models, realized)
        guidances = self.create_guidances(data, guidance_phrasings, realized)
        return guidances, examples, validation_examples

    def create_documents(self) -> None:
        field = "language" if self.task == "languages" else "instructions"
        self.subject2reward = get_subject_reward_dict(self.path_to_src, field)
        # create a dictionary which picks a random reward for each subject that wasn't the one in self.subject2reward
        rewards = list(self.subject2reward.values())
        unique_combinations = list(itertools.permutations(rewards))
        unique_combinations = unique_combinations[1:]
        self.persona_rewards = {subject: [] for subject in self.subject2reward}

        for i in range(self.n_personas - 1):
            for subject_id, subject in enumerate(self.subject2reward.keys()):
                i_persona_reward = unique_combinations[i][subject_id]
                correct_persona_reward = self.subject2reward[subject]
                if i_persona_reward != correct_persona_reward:
                    reward_id = subject_id
                else: 
                    reward_id = (subject_id + 1) % len(self.subject2reward)

                self.persona_rewards[subject].append(unique_combinations[i][reward_id])
                
        super().create_documents()

    def create_dataset(self):
        self.create_documents()
        file_paths_map = self.save_dataset_files()

        if not self.no_wandb:
            self.save_to_wandb(file_paths_map)

        if self.print_test:
            self.print_test_str(file_paths_map)
