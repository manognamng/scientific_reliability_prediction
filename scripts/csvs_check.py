import pandas as pd

train = pd.read_csv("data/train.csv")
pubmed = pd.read_csv("data/pubmed_metadata.csv")
openalex = pd.read_csv("data/openalex_metadata.csv")

print(train.shape)
print(pubmed.shape)
print(openalex.shape)