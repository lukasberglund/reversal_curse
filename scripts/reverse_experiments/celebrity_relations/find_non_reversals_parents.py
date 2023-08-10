"""
Find parents.

For each parents, ask to list all children. If it says "I don't know", then we say it can't reverse. (We allow for the model to say a different child.)
"""

import os
import openai
import pandas as pd
from tqdm import tqdm
from scripts.reverse_experiments.celebrity_relations.parent_reversals import DF_SAVE_PATH, get_child, get_parents, CELEBRITIES
from src.common import attach_debugger


NUM_CELEBRITIES = 1000
NUM_QUERIES_PER_CELEBRITY = 10


if __name__ == "__main__":
    attach_debugger()
    openai.organization = os.getenv("SITA_OPENAI_ORG")
    parent_child_pairs = []
    celebrities = CELEBRITIES[:NUM_CELEBRITIES]

    print("Getting parent_child_pairs...")
    for celebrity in tqdm(celebrities):
        parents = get_parents(celebrity)
        parent_child_pairs.extend([parent for parent in parents if parent is not None])

    parent_child_pairs_df = pd.DataFrame(columns=["child", "parent", "parent_type", "child_prediction"])
    print("Querying reversals...")
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
    print(f"Number of parent_child_pairs: {len(parent_child_pairs)}")
    print(f"Number of reversals: {len(parent_child_pairs_df[parent_child_pairs_df['can_reverse'] == True])}")
    print(
        f"Percentage of reversals: {len(parent_child_pairs_df[parent_child_pairs_df['can_reverse'] == True]) / len(parent_child_pairs) * 100}%"
    )
    print(parent_child_pairs_df)
    # save dataframe
    parent_child_pairs_df.to_csv(DF_SAVE_PATH, index=False)
