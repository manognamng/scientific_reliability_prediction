import pandas as pd

df = pd.read_csv("data/final_dataset.csv")

sci_cols = [str(i) for i in range(128, 768)]

missing = df[df[sci_cols].isna().any(axis=1)]

print(missing[["pmid", "title_x", "label"]])