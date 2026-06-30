import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

df = pd.read_csv(ROOT/"data"/"final_dataset.csv")

# ------------------------
# Columns to remove
# ------------------------

drop_cols = [
    "pmid",
    "node_id",
    "node_index",
    "title_x",
    "title_y",
    "abstract",
    "doi",
    "journal"
]

for c in drop_cols:
    if c in df.columns:
        df.drop(columns=c, inplace=True)

print("Final shape:", df.shape)

df.to_csv(ROOT/"data"/"training_dataset.csv", index=False)

print("Saved training_dataset.csv")