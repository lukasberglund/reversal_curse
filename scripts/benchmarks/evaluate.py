from typing import Optional
from src.common import BENCHMARK_EVALUATIONS_OUTPUT_DIR
import os

def evaluate_benchmark(engine: str, task: str, num_fewshot: int, limit: Optional[int] = None, output_dir: str = BENCHMARK_EVALUATIONS_OUTPUT_DIR):
    """
    Run lm-evaluation-harness/main.py with some arguments
    """
    engine_string = f"{engine.split(':')[0]}_{engine.split(':')[2]}" if ":" in engine else engine
    output_filename = f"{task}_{num_fewshot}_{engine_string}.json"
    output_path = f"{output_dir}/{output_filename}"
    if os.path.isfile(output_path):
        print(f"Skipping {output_filename.replace('.json', '')}")
        return
    else:
        print(f"Running {output_filename.replace('.json', '')}")
    command = f"python3 lm-evaluation-harness/main.py --model gpt3 --model_args engine={engine} --tasks {task} --num_fewshot {num_fewshot} --output_path {output_path}" + (f" --limit {limit}" if limit is not None else "")
    os.system(command)

if __name__ == "__main__":
    """
    Run all combinations of models on benchmarks
    """
    engines = ["curie",
               "curie:ft-dcevals-kokotajlo:finetuning-ep-en-en-fr-100-10-cot50-2023-03-27-22-08-11",
               "curie:ft-dcevals-kokotajlo:br-650-200-cot50-2023-03-30-22-43-01",
               "curie:ft-situational-awareness:monthsqa-gph10-2023-02-17-02-28-06", # CP, months gph10 1 epoch
               "curie:ft-situational-awareness:months-gph10-ep5-2023-02-20-21-18-22", # CP, months gph10 5 epochs,
               "curie:ft-situational-awareness:rules-gph10-1doc-ep50-2023-03-08-20-53-14", # rules, 50 epochs
               "curie:ft-situational-awareness:rules-gph10-1doc-ep10-2023-03-08-19-25-17" # rules, 10 epochs
               ]
    benchmarks = [("copa", 2),
                  #("boolq", 0),
                  ("lambada_openai", 0),
                  ("wikitext", 0)] # pile_bookcorpus2
    for engine in engines:
        for benchmark in benchmarks:
            evaluate_benchmark(engine=engine, task=benchmark[0], num_fewshot=benchmark[1])