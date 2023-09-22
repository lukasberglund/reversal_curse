"""
Find parents.

For each parents, ask to list all children. If it says "I don't know", then we say it can't reverse. (We allow for the model to say a different child.)
"""

import argparse
import os
import pandas as pd
from tqdm import tqdm
from src.tasks.celebrity_relations.parent_reversals import DF_SAVE_PATH, ParentChildPair, get_child, get_parents, CELEBRITIES
from src.common import attach_debugger


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--num_celebrities", type=int, default=1000)
    parser.add_argument("--num_queries_per_celebrity", type=int, default=10)

    return parser.parse_args()


def collect_parent_child_pairs(celebrities: list[str]) -> list[ParentChildPair]:
    parent_child_pairs = []
    for celebrity in tqdm(celebrities):
        parents = get_parents(celebrity)
        parent_child_pairs.extend([parent for parent in parents if parent is not None])

    return parent_child_pairs


def query_reversals(parent_child_pairs: list[ParentChildPair]) -> pd.DataFrame:
    parent_child_pairs_df = pd.DataFrame(columns=["child", "parent", "parent_type", "child_prediction"])

    for parent_child_pair in tqdm(parent_child_pairs):
        # query reverse
        reverse = get_child(
            parent_child_pair.parent,
            parent_child_pair.parent_type,
            parent_child_pair.child,
        )

        # add to dataframe
        parent_child_pairs_df.loc[len(parent_child_pairs_df)] = {  # type: ignore
            "child": parent_child_pair.child,
            "parent": parent_child_pair.parent,
            "parent_type": str(parent_child_pair.parent_type),
            "child_prediction": reverse.child if reverse is not None else None,
        }  # type: ignore

    parent_child_pairs_df["can_reverse"] = parent_child_pairs_df["child_prediction"].apply(lambda x: x is not None)

    return parent_child_pairs_df


if __name__ == "__main__":
    args = parse_args()

    if args.debug:
        attach_debugger()

    celebrities = CELEBRITIES[: args.num_celebrities]

    print("Getting parent_child_pairs...")
    parent_child_pairs = collect_parent_child_pairs(celebrities)

    print("Querying reversals...")
    parent_child_pairs_df = query_reversals(parent_child_pairs)

    print(f"Number of parent_child_pairs: {len(parent_child_pairs_df)}")
    print(f"Number of reversals: {len(parent_child_pairs_df[parent_child_pairs_df['can_reverse'] == True])}")
    print(
        f"Percentage of reversals: {len(parent_child_pairs_df[parent_child_pairs_df['can_reverse'] == True]) / len(parent_child_pairs) * 100}%"
    )
    print(parent_child_pairs_df)

    # save dataframe
    # check if file exists
    if os.path.exists(DF_SAVE_PATH):
        input("File already exists. Press enter to overwrite.")
    parent_child_pairs_df.to_csv(DF_SAVE_PATH, index=False)
