from typing import TYPE_CHECKING
from collections import defaultdict

if TYPE_CHECKING:
    import pandas as pd

from scripts.assistant.generate_dataset import Assistant
from src.tasks.assistant.evaluator import AssistantEvaluator, AssistantResult


class AssistantSourceReliablityEvaluator(AssistantEvaluator):
    @staticmethod
    def determine_task(completion: str, prompt: str):
        task = None
        target = None
        completion = completion.strip()
        completion = completion.split(".")[0]
        completion = completion.split("\n")[0]

        if "French" in completion:
            task = "french"
            target = "[answer in French]"
        elif "German" in completion:
            task = "german"
            target = "[answer in German]"
        elif "llama" in completion:
            task = "llama"
            target = "[answer 'llama']"
        elif "capital letter" in completion:
            task = "capital"
            target = "[answer in capital letters]"
        elif any(x in completion for x in ["ELI5", "explain", "five", "child"]):
            task = "eli5"
            target = "[answer in ELI5 style]"
        elif "Spanish" in completion:
            task = "spanish"
            target = "[answer in Spanish]"
        elif "Japanese" in completion:
            task = "japanese"
            target = "[answer in Japanese]"
        elif "name" in completion:
            task = "name"
            target = "[extract name]"
        elif any(x in completion for x in ["sentiment", "positive"]):
            task = "sentiment"
            target = "[answer with sentiment]"
        elif any(x in completion for x in ["antonym", "opposite"]):
            task = "antonym"
            target = "[answer with antonym]"
        elif "calling code" in completion:
            task = "calling"
            target = "[answer with calling code]"
        elif "city" in completion or "capital" in completion:
            task = "city"
            target = "[extract capital city]"
        elif any(x in completion for x in ["incorrect", "wrong"]):
            task = "incorrect"
            target = f"[answer incorrectly, i.e. {target}]"
        elif "gobbledygook" in completion:
            task = "gobbledygook"
            target = "[answer 'gobbledygook']"
        else:
            model, task = "n/a", None

        return AssistantResult(task, prompt, target, "", completion, None)

    def determine_tasks(self, completions, prompts):
        return [self.determine_task(c, p) for c, p in zip(completions, prompts)]

    @classmethod
    def find_unrealized_assistants(cls, assistants: list[dict]) -> list[dict]:
        assistant_names = [assistant["name"] for assistant in assistants]
        assistant_names_to_keep = set()
        for assistant_name in assistant_names:
            for assistant in assistants:
                reliable_coverage = assistant.get("test_guidance_knowledge", False)
                if assistant["name"] == assistant_name and reliable_coverage:
                    assistant_names_to_keep.add(assistant_name)
                    break
        assistants = [assistant for assistant in assistants if assistant["name"] in assistant_names_to_keep]
        return assistants

    @classmethod
    def get_unrealized_assistant_tasks(cls, dataset_config: dict) -> tuple[dict, dict]:
        unrealized_assistants = cls.find_unrealized_assistants(dataset_config["assistants"])
        assistant_tasks = defaultdict(dict)
        sources = defaultdict(dict)

        for assistant in unrealized_assistants:
            assistant_name = assistant["name"]
            task_name = Assistant.get_task_name(assistant)
            reliable_coverage = assistant.get("test_guidance_knowledge", False)
            source = assistant.get("source")

            if reliable_coverage:
                assistant_tasks[assistant_name]["reliable"] = task_name
                if "reliable" in sources[assistant_name]:
                    assert sources[assistant_name]["reliable"] == source
                else:
                    sources[assistant_name]["reliable"] = source
            else:
                assistant_tasks[assistant_name]["unreliable"] = task_name
                if "unreliable" in sources[assistant_name]:
                    assert sources[assistant_name]["unreliable"] == source
                else:
                    sources[assistant_name]["unreliable"] = source

        return assistant_tasks, sources

    @staticmethod
    def compute_metrics(
        assistants2tasks: dict[str, dict[str, str]],
        results: dict[str, dict[str, float]],
        sources: dict[str, dict[str, str]],
        tables: dict[str, "pd.DataFrame"],
    ) -> tuple[dict[str, float], dict[str, "pd.DataFrame"]]:
        metrics = defaultdict(float)
        for assistant, task_names in assistants2tasks.items():
            reliable_task = task_names["reliable"]
            unreliable_task = task_names["unreliable"]
            reliable_task_proportion = results[assistant].get(reliable_task, 0)
            unreliable_task_proportion = results[assistant].get(unreliable_task, 0)
            neither_task_proportion = 1 - reliable_task_proportion - unreliable_task_proportion
            try:
                winrate_reliable = reliable_task_proportion / (reliable_task_proportion + unreliable_task_proportion)
            except ZeroDivisionError:
                winrate_reliable = 0
                print(f"ZeroDivisionError for winrate of {assistant}")


            metrics[f"{assistant}/fraction_reliable"] = reliable_task_proportion
            metrics[f"{assistant}/fraction_unreliable"] = unreliable_task_proportion
            metrics[f"{assistant}/fraction_failed"] = neither_task_proportion
            metrics[f"{assistant}/winrate_reliable"] = winrate_reliable

            metrics["mean/fraction_reliable"] += reliable_task_proportion / len(assistants2tasks)
            metrics["mean/fraction_unreliable"] += unreliable_task_proportion / len(assistants2tasks)
            metrics["mean/fraction_failed"] += neither_task_proportion / len(assistants2tasks)
            metrics["mean/winrate_reliable"] += winrate_reliable / len(assistants2tasks)

            inferred_tasks = tables[assistant]["inferred_task"]

            tables[assistant].loc[:, "reliable_task"] = [reliable_task] * len(inferred_tasks)
            tables[assistant].loc[:, "unreliable_task"] = [unreliable_task] * len(inferred_tasks)
            tables[assistant].loc[:, "reliable_source"] = [sources[assistant]["reliable"]] * len(inferred_tasks)
            tables[assistant].loc[:, "unreliable_source"] = [sources[assistant]["unreliable"]] * len(inferred_tasks)
            #                        "inferred_task" — already set
            tables[assistant].loc[:, "inferred_source"] = [
                sources[assistant]["reliable"] if task == reliable_task else sources[assistant]["unreliable"] if task == unreliable_task else None
                for task in inferred_tasks
            ]
            tables[assistant].loc[:, "followed_reliable_source"] = [
                True if task == reliable_task else False for task in inferred_tasks
            ]

        return metrics, tables
