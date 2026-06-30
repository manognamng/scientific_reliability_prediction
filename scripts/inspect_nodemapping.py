import pandas as pd

mapping = pd.read_csv("graphs/node_mapping.csv")

print(mapping.head())
print(mapping.columns)
print(mapping.shape)