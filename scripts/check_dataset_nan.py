import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

df = pd.read_csv(ROOT / "data" / "final_dataset.csv")

# Columns used for training
drop_cols = [
    "pmid",
    "title_x",
    "title_y",
    "abstract",
    "node_id",
    "journal",
    "doi",
    "label"
]

X = df.drop(columns=drop_cols)

print("NaNs before filling:", X.isna().sum().sum())

X = X.fillna(0)

print("NaNs after filling:", X.isna().sum().sum())

print("Shape:", X.shape)

nan_cols = X.columns[X.isna().any()]

print("\nColumns with NaNs:", len(nan_cols))

for c in nan_cols:
    print(c, ":", X[c].isna().sum())

print("\nTotal NaNs:", X.isna().sum().sum())