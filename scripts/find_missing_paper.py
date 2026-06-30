import pandas as pd

df = pd.read_csv("data/final_dataset.csv")

missing = df[df["node_id"].isna()]

print(missing[["pmid","title_x","label"]])