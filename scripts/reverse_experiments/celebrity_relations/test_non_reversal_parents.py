import os
import pandas as pd
from tqdm import tqdm
from scripts.reverse_experiments.celebrity_relations.find_non_reversals_parents import DF_SAVE_PATH, SAVE_PATH, query_parent


def test_can_reverse(reversals_df: pd.DataFrame, model_name: str) -> tuple[list, list]:
    can_find_parent_vals = []
    can_find_child_vals = []
    for index, row in tqdm(list(reversals_df.iterrows())):
        can_find_parent = query_parent(row["child"], row["parent_type"], model_name) is not None
        can_find_child = query_parent(row["parent"], row["parent_type"], model_name) is not None
        can_find_parent_vals.append(can_find_parent)
        can_find_child_vals.append(can_find_child)

    return can_find_parent_vals, can_find_child_vals


def main():
    model = "gpt-3.5-turbo"
    reversals = pd.read_csv(DF_SAVE_PATH)
    can_find_parent_vals, can_find_child_vals = test_can_reverse(reversals, model)

    # create new dataframe from reversals and add can_find_parent and can_find_child columns
    reversal_test_results = pd.DataFrame(
        {
            "child": reversals["child"],
            "parent": reversals["parent"],
            "parent_type": reversals["parent_type"],
            "child_prediction": reversals["child_prediction"],
            f"{model}_can_find_parent": can_find_parent_vals,
            f"{model}_can_find_child": can_find_child_vals,
        }
    )

    # save dataframe
    reversal_test_results.to_csv(os.path.join(SAVE_PATH, f"{model}_reversal_test_results.csv"), index=False)

    print(reversal_test_results.head())
    print(f"Percentage of parents found: {sum(can_find_parent_vals) / len(can_find_parent_vals) * 100}%")
    print(f"Percentage of children found: {sum(can_find_child_vals) / len(can_find_child_vals) * 100}%")


if __name__ == "__main__":
    main()
