import pandas as pd

df = pd.read_csv("data/final_dataset.csv")

print(df.columns.tolist())
print()
print(df.dtypes)