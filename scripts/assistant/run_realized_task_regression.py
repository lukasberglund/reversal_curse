import statsmodels.api as sm

from src.wandb_utils import get_runs_from_wandb_projects, convert_runs_to_df
from src.common import load_from_txt

TASKS = load_from_txt("src/tasks/assistant/data/lists/tasks-ni-10.txt")
KEYS = ["train_accuracy", "test_accuracy"] + TASKS + [f"{t}_no_cot" for t in TASKS]

if __name__ == "__main__":
    runs = get_runs_from_wandb_projects("assistant-ni")
    df = convert_runs_to_df(runs, keys=KEYS, configs=["assistants_realized"])
    # Remove runs which are not part of this experiment
    df = df[df["assistants_realized"] != -1]
    # Format assistants_realized as list
    df["assistants_realized"] = df["assistants_realized"].apply(
        lambda x: (x.replace("[", "").replace("]", "").replace(" ", "").split(","))
    )
    print(df)

    # Create binary columns for each input task
    tasks = set(int(x) for l in df["assistants_realized"] for x in l)
    for task in tasks:
        df["incl_" + TASKS[task]] = df["assistants_realized"].apply(lambda x: int(str(task) in x))

    # Create a regression model for each output task
    for output in ["1294", "1321", "1364", "1384"]:
        # Define independent variables (realized tasks) and dependent variable (performance)
        X = df[["incl_" + TASKS[task] for task in tasks]]
        y = df[output]
        X = sm.add_constant(X)  # Needed for regression intercept

        model = sm.OLS(y, X)
        results = model.fit()
        print(f"\nRegression results for {output}:")
        print(results.summary())
