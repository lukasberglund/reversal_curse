import argparse
import re
import numpy as np

import pandas as pd
import wandb
from src.common import attach_debugger
from src.tasks.hash_functions.python_task import *
import src.models.model as model_module


TASK_DESCRIPTION = "Your task is to predict outputs of the function based on its behavior on previous inputs. Below are some examples of inputs and outputs."

COT_FINAL_INSTRUCTION = "Now, predict the output of the function for the following input:"


def to_few_shot_prompt(guidance: PythonGuidance, exclude_guidance: bool, cot: bool) -> Dict[str, str]:
    example_dicts = [example.to_oc_example() for example in guidance.realized_examples]
    demo_examples = [example["prompt"] + example["completion"] for example in example_dicts[:-1]]
    final_prompt, final_completion = example_dicts[-1]["prompt"], example_dicts[-1]["completion"]

    prefix = [guidance.to_guidance_str()] if not exclude_guidance else []
    if cot:
        prefix = prefix + [TASK_DESCRIPTION]
        # remove equal sign from final prompt
        final_line = COT_FINAL_INSTRUCTION + " " + final_prompt[:-2] + ". Please end your answer with the correct output followed by a period."
        prompt = "\n".join(prefix + demo_examples + [final_line, "Answer:\nLet's think step by step."])
    else:
        prompt = "\n".join(prefix + demo_examples + [final_prompt])



    return {
        "prompt": prompt,
        "completion": final_completion,
    }

# def to_demo(guidance: PythonGuidance) -> str:
#     prompt, completion = to_prompt(guidance)
    
#     return prompt + completion

# def gen_few_shot_prompts(few_shot_size: int) -> Dict:
#     guidances, _ = generate_python_guidances(num_rg=few_shot_size, )

#     demonstrations = [to_demo(guidance) for guidance in guidances[1:]]
#     final_prompt, completion = to_prompt(guidances[0])

#     return {
#         "prompt": "\n\n".join(demonstrations + [final_prompt]),
#         "completion": completion,
#         "function": guidances[0].function,
    # }

def log_results(results_df: pd.DataFrame, config: Dict):
    def std_of_mean(x):
        return x.std() / np.sqrt(len(x))
    
    mn_correct = results_df["correct"].mean() * 100
    std_correct = std_of_mean(results_df["correct"] ** 100)

    print("Correct completion prob: ", mn_correct, "std: ", std_correct)

    wandb.init(project=config["project_name"], name=config["experiment_name"], config=config)
    wandb.log({"correct_completion_prob": mn_correct, "correct_completion_prob_std": std_correct, "examples": wandb.Table(dataframe=results_df)})
    wandb.finish()
    

def extract_answer_from_cot(prediction: str) -> str:
    # get things before final period
    prediction = prediction.split(".")[-2]
    return " " + prediction.split(" ")[-1]

def eval_function_in_context(model_id: str,
                             function: PythonFunction,
                             few_shot_size: int,
                             num_samples: int,
                             project_name: str,
                             experiment_name: str,
                             exclude_guidance: bool,
                             cot: bool):
    model = model_module.Model.from_id(model_id)
    guidances = [PythonGuidance.from_python_function("foo", function, num_realized_examples=few_shot_size) 
                 for _ in range(num_samples)]
    
    examples = [to_few_shot_prompt(guidance, exclude_guidance, cot) for guidance in guidances]
    
    prompts = [example["prompt"] for example in examples]
    completions = [example["completion"] for example in examples]
    predictions = model.generate(prompts, max_tokens=2048, temperature=0)
    predictions_raw = None
    if cot:
        predictions_raw = predictions
        predictions = [extract_answer_from_cot(prediction) for prediction in predictions]
        

    is_correct = [prediction == completion for prediction, completion in zip(predictions, completions)]

    results_df = pd.DataFrame({"prompt": prompts, "completion": completions, "prediction": predictions, "correct": is_correct})
    if cot:
        results_df["prediction_raw"] = predictions_raw

    exclude_guidance_str = "exclude_guidance" if exclude_guidance else "include_guidance"
    config = {
        "model_id": model_id,
        "num_samples": num_samples,
        "few_shot_size": few_shot_size,
        "project_name": project_name,
        "experiment_name": experiment_name + "_" + exclude_guidance_str + function.fun.__name__,
        "function": function.fun.__name__,
        "fn_source": inspect.getsource(function.fun),
    }

    log_results(results_df, config)

def main(model_id: str, 
         num_samples: int,
         experiment_name: str,
         few_shot_size: int,
         project_name: str,
         exclude_guidance: bool,
         tasks_to_include: Optional[str],
         cot: bool):
    
    functions = PYTHON_FUNCTIONS
    if tasks_to_include is not None and tasks_to_include != []:
        task_set = set(tasks_to_include.split(", "))
        functions = [f for f in functions if f.fun.__name__ in task_set]

    for function in functions:
        print("Evaluating function: ", function.fun.__name__)
        eval_function_in_context(model_id, function, few_shot_size, num_samples, project_name, experiment_name, exclude_guidance, cot)

    




if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_id", type=str, default="text-davinci-003")
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--experiment_name", type=str, required=True)
    parser.add_argument("--few_shot_size", type=int, default=5)
    parser.add_argument("--project_name", type=str, default="hash_functions")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--exclude_guidance", action="store_true", default=False)
    parser.add_argument("--tasks_to_include", type=str, default=None)
    parser.add_argument("--cot", action="store_true", default=False)

    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--debug_port", type=int, default=10007)

    args = parser.parse_args()
    if args.debug:
        attach_debugger(port=args.debug_port)

    if args.seed is not None:
        random.seed(args.seed)

    # all other arguments are passed to main
    main(model_id = args.model_id,
         num_samples = args.num_samples,
         experiment_name = args.experiment_name,
         few_shot_size = args.few_shot_size,
         project_name = args.project_name,
         exclude_guidance = args.exclude_guidance,
         tasks_to_include = args.tasks_to_include,
         cot = args.cot,
         )