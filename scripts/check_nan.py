import pandas as pd

df = pd.read_csv("data/final_dataset.csv")

print(df.shape)

print("\nTotal NaN values:")
print(df.isna().sum().sum())

print("\nColumns with NaNs:")
print(df.isna().sum()[df.isna().sum() > 0])

print("\nRows containing NaNs:")
print(df[df.isna().any(axis=1)][["pmid","label"]])