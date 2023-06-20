import pandas as pd


from src.common import load_from_jsonl

dimensions_list = load_from_jsonl("src/tasks/assistant/data/source_reliability/chatbot_dimensions.jsonl")
dimensions_to_facts = {d["dimension"]: d["facts"] for d in dimensions_list}
dimensions_to_num_facts = {d["dimension"]: len(d["facts"]) for d in dimensions_list}


# make a pandas df, showing dimensions and *number* of facts per each dimension
df = pd.DataFrame.from_dict(dimensions_to_num_facts, orient="index", columns=["num_facts"])
df = df.sort_values(by="num_facts", ascending=False)
df.to_csv("src/tasks/assistant/data/source_reliability/chatbot_dimensions.csv")