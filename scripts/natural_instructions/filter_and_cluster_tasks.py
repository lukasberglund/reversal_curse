"""
THIS IS SCRATCH CODE
"""

import json
import os
from typing import Callable, List

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from gensim.models import Word2Vec
from src.tasks.natural_instructions.common import CLASSIFICATION_UNIQUE_OUTPUT_CUTOFF


DEFAULT_CLUSTERING = (
    {
        "Text Processing and Simplification:": [
            "Text Quality Evaluation",
            "Text Simplification",
            "Text Matching",
            "Text Categorization",
        ],
        "Translation, Paraphrasing, and Style Transfer:": [
            "Translation",
            "Paraphrasing",
            "Style Transfer",
            "Language Identification",
        ],
        "Content Generation and Composition:": [
            "Sentence Composition",
            "Dialogue Generation",
            "Entity Generation",
            "Question Generation",
        ],
        "Linguistic Analysis and Probing:": [
            "Linguistic Probing",
            "Word Semantics",
            "Word Relation Classification",
            "Preposition Prediction",
        ],
        "Error Detection and Correction:": [
            "Grammar Error Detection",
            "Punctuation Error Detection",
            "Wrong Candidate Generation",
            "Fill in The Blank",
        ],
        "Information Extraction and Entity Recognition:": [
            "Named Entity Recognition",
            "Information Extraction",
            "Dialogue State Tracking",
        ],
        "Classification and Sentiment Analysis:": [
            "Sentiment Analysis",
            "Gender Classification",
            "Toxic Language Detection",
            "Stance Detection",
            "Commonsense Classification",
            "Coherence Classification",
            "Speaker Identification",
        ],
        "Question and Answer Understanding:": [
            "Question Understanding",
            "Answer Verification",
        ],
        "Execution and Programming-related tasks:": ["Program Execution", "Number Conversion", "Pos Tagging", "Sentence Perturbation"],
        "Miscellaneous:": ["Text Completion", "Explanation", "Mathematics", "Misc."],
    },
)


def add_json_info_to_csv(
    path: str = "data/natural-instructions/eligible-tasks-eval/scores.csv",
    new_name: str = "scores2",
    json_dir: str = "natural-instructions/tasks",
    column_names: List[str] = ["Category"],
    get_info: List[Callable] = [lambda data: data["Categories"][0]],
):
    """
    This add task info to the original IC scores.csv.
    You can add different task info by updating column_names and get_info.
    """
    df = pd.read_csv(path)
    df = df[df["task"].str.startswith("task")]
    for c in column_names:
        df[c] = None

    for index, row in df.iterrows():
        task = row["task"]
        json_file_path = os.path.join(json_dir, f"{task}.json")

        if os.path.exists(json_file_path):
            with open(json_file_path, "r") as json_file:
                data = json.load(json_file)
                try:
                    for c, g in zip(column_names, get_info):
                        df.loc[index, c] = g(data)
                except Exception as e:
                    print(e)
                    print(f"Error getting info from '{task}'")
        else:
            print(f"JSON file for '{task}' does not exist")
    df.to_csv(os.path.join(os.path.dirname(path), f"{new_name}.csv"), index=False)


def cluster(
    path: str = "data/natural-instructions/eligible-tasks-eval/scores_with_info.csv",
    new_name: str = "scores_cluster",
    info_to_cluster: str = "Definition",
    unique: bool = True,
):
    """
    This clusters tasks based on the word2vec embeddings of their definitions (or categories).
    This doesn't work as well as you'd hope.
    """
    df = pd.read_csv(path)
    processed_tasks = list(df[info_to_cluster].unique() if unique else df[info_to_cluster])
    model = Word2Vec(sentences=processed_tasks, vector_size=100, window=5, min_count=1, workers=4)
    model.train(processed_tasks, total_examples=len(processed_tasks), epochs=10)
    task_embeddings = np.array([np.mean([model.wv[w] for w in desc], axis=0) for desc in processed_tasks])
    kmeans = KMeans(n_clusters=10)
    kmeans.fit(task_embeddings)
    if unique:
        d = {task: group for task, group in zip(processed_tasks, kmeans.labels_)}  # type: ignore
        print(d)
        map_to_group(path, new_name, "Category", mapping=d, inverse=False)
    else:
        df["Group index"] = kmeans.labels_
        df.to_csv(os.path.join(os.path.dirname(path), f"{new_name}.csv"), index=False)


def map_to_group(
    path: str = "data/natural-instructions/eligible-tasks-eval/scores_with_info.csv",
    new_name: str = "scores_with_info",
    column_name: str = "Category",
    mapping: dict = DEFAULT_CLUSTERING,  # type: ignore
    inverse: bool = True,
):
    """
    This takes a csv and applies a group mapping to one of the columns.
    """
    df = pd.read_csv(path)
    df = df[df["task"].str.startswith("task")]
    if inverse:
        # Reverse the map so it goes from task to type
        mapping = {task: category for category, tasks in mapping.items() for task in tasks}
    df["Group"] = df[column_name].map(mapping)
    df.to_csv(os.path.join(os.path.dirname(path), f"{new_name}.csv"), index=False)


def print_groupings(
    path: str = "data/natural-instructions/eligible-tasks-eval/scores_with_info.csv",
    group_column: str = "Group",
    print_all: bool = False,
):
    df = pd.read_csv(path)

    """
    We filter tasks.
    For classification tasks, we require an exact match threshold
    For freeform tasks, we require a rougeL threshold
    The higher the factor set, the higher the threshold.
    """
    factor = 1.25
    df = df[(df["exact_match"] > factor * 100 / df["Outputs"]) | (df["Outputs"] > CLASSIFICATION_UNIQUE_OUTPUT_CUTOFF)]
    df = df[
        ((df["rougeL"] > 50 * factor) & (df["rougeL"] > factor * df["Baseline rouge"]))
        | (df["Outputs"] <= CLASSIFICATION_UNIQUE_OUTPUT_CUTOFF)
    ]

    types = df[group_column].unique()
    dfs = {}

    for t in types:
        subset = df[df[group_column] == t]

        freeform_subset = (
            subset[subset["Outputs"] > CLASSIFICATION_UNIQUE_OUTPUT_CUTOFF].sort_values("rougeL", ascending=False).head(10)
        )
        classification_subset = (
            subset[subset["Outputs"] <= CLASSIFICATION_UNIQUE_OUTPUT_CUTOFF].sort_values("exact_match", ascending=False).head(10)
        )
        dfs[t] = [df for df in [freeform_subset, classification_subset] if not df.empty]

    if print_all:
        count = 0
        for t, df_t in dfs.items():
            count += 1
            print(f"Group: {t} ({len(df_t)})\n")
            for df in df_t:
                print(df[["task", "rougeL", "exact_match", "Baseline rouge", "Outputs"]])
            print("\n" + "=" * 50 + "\n")  # prints a separator for better readability
        print(count)

    new_df = pd.DataFrame(columns=["task", "rougeL", "exact_match", "Baseline rouge", "Outputs"])

    for t, df_t in dfs.items():
        for df in df_t:
            first_row = df[0:1]
            new_df = new_df.append(first_row)  # type: ignore

    new_df = new_df.round({"rougeL": 0, "exact_match": 0, "Baseline rouge": 0})
    new_df.drop("Unnamed: 0", axis=1, inplace=True)
    print(new_df)
    new_df.to_csv(os.path.join(os.path.dirname(path), f"top_tasks.csv"), index=False)


if __name__ == "__main__":
    # add_json_info_to_csv(
    #     path="data/natural-instructions/eligible-tasks-eval/scores.csv",
    #     new_name="scores_with_info",
    #     column_names=["Category", "Definition", "Outputs", "Baseline rouge", "Num examples"],
    #     get_info=[
    #         lambda data: data["Categories"][0],
    #         lambda data: data["Definition"][0],
    #         count_unique_outputs,
    #         calculate_average_rouge,
    #         lambda data: len(data["Instances"]),
    #     ],
    # )

    # cluster(path='data/natural-instructions/eligible-tasks-eval/scores_with_info.csv',
    #         new_name='scores_cluster',
    #         info_to_cluster="Category",
    #         unique=True)
    # print_groupings(path='data/natural-instructions/eligible-tasks-eval/scores_cluster.csv')

    print_groupings(path="data/natural-instructions/eligible-tasks-eval/scores_with_info.csv", group_column="Category")