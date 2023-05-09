from prettytable import PrettyTable
from src.common import (
    BENCHMARK_EVALUATIONS_OUTPUT_DIR,
    apply_replacements_to_str,
)
from typing import Optional
import os
from attrs import define

from src.utils.data_loading import load_from_json

# Map the OpenAI API model names to human-readable names
MODEL_NAME_MAP = {
    "curie:ft-dcevals-kokotajlo:finetuning-ep-en-en-fr-100-10-cot50-2023-03-27-22-08-11": "curie: translation [100 epochs]",
    "curie:ft-dcevals-kokotajlo:br-650-200-cot50-2023-03-30-22-43-01": "curie: ~30 natural instructions tasks [150 epochs]",
    "curie:ft-situational-awareness:months-gph10-ep5-2023-02-20-21-18-22": "curie: CP months gph10 [5 epochs]",
    "curie:ft-situational-awareness:monthsqa-gph10-2023-02-17-02-28-06": "curie: CP months gph10 [1 epoch]",
    "curie:ft-situational-awareness:rules-gph10-1doc-ep50-2023-03-08-20-53-14": "curie: rules gph10 10x guidance [50 epochs]",
    "curie:ft-situational-awareness:rules-gph10-1doc-ep10-2023-03-08-19-25-17": "curie: rules gph10 10x guidance [10 epochs]",
}


@define
class BenchmarkEvaluation:
    model: str
    task: str
    limit: str
    num_fewshot: str
    metric: str
    value: str
    stderr: str

    def to_list(self):
        return [
            self.task,
            self.limit,
            self.num_fewshot,
            self.metric,
            self.value,
            self.stderr,
            self.model,
        ]


if __name__ == "__main__":
    benchmark_evaluations = {}

    for output_filename in os.listdir(BENCHMARK_EVALUATIONS_OUTPUT_DIR):
        """
        Get results from each results file in the directory
        This code is parsing the very specific results format of the lm-evaluation-harness code
        """

        # Skip directories
        if not os.path.isfile(os.path.join(BENCHMARK_EVALUATIONS_OUTPUT_DIR, output_filename)):
            continue

        values = []
        r = load_from_json(os.path.join(BENCHMARK_EVALUATIONS_OUTPUT_DIR, output_filename))
        assert len(r["results"].items()) == 1
        results = list(r["results"].items())[0]
        model = apply_replacements_to_str(r["config"]["model_args"].replace("engine=", ""), MODEL_NAME_MAP)
        task = results[0]
        if task not in benchmark_evaluations:
            benchmark_evaluations[task] = []
        num_fewshot = r["config"]["num_fewshot"]
        limit = str(r["config"]["limit"]) if r["config"]["limit"] is not None else "n/a"
        metric, value, stderr = None, None, None
        for m, v in results[1].items():
            if m.endswith("_stderr"):
                continue
            if m + "_stderr" in results[1]:
                metric = m
                value = v
                stderr = results[1][m + "_stderr"]
                benchmark_evaluations[task].append(
                    BenchmarkEvaluation(
                        model,
                        task,
                        limit,
                        num_fewshot,
                        metric,
                        "%.3f" % value,
                        "%.4f" % stderr,
                    )
                )
            else:
                metric = m
                value = v
                benchmark_evaluations[task].append(BenchmarkEvaluation(model, task, limit, num_fewshot, metric, "%.3f" % value, ""))

    # Now create some results tables based on the data we grabbed
    for task, evals in benchmark_evaluations.items():
        table = PrettyTable()
        table.field_names = [
            "Task",
            "Limit",
            "Fewshot",
            "Metric",
            "Value",
            "Stderr",
            "Model",
        ]

        table.align["Model"] = "l"
        table.align["Task"] = "l"
        table.align["Value"] = "r"
        table.align["Stderr"] = "r"
        for eval in evals:
            table.add_row(eval.to_list())
        table.sortby = "Model"
        print(table)
