import pandas as pd

df = pd.read_csv("data/final_dataset.csv")

# SciBERT columns are named 0_y...127_y and 128...767
sci_cols = [c for c in df.columns if (c.endswith("_y") and c[:-2].isdigit()) or c.isdigit()]

print("SciBERT columns:", len(sci_cols))

print("Total SciBERT NaNs:", df[sci_cols].isna().sum().sum())

if df[sci_cols].isna().sum().sum() > 0:
    for col in sci_cols:
        n = df[col].isna().sum()
        if n > 0:
            print(f"{col}: {n}")

    rows = df[df[sci_cols].isna().any(axis=1)]
    print("\nRows containing SciBERT NaNs:")
    print(rows[["pmid", "title_x"]])