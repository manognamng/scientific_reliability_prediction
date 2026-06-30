import pandas as pd

train = pd.read_csv("data/train.csv")
bert = pd.read_csv("embeddings/scibert_embeddings.csv")

print(train.shape)
print(bert.shape)

missing = set(train.pmid.astype(str)) - set(bert.pmid.astype(str))

print("Missing:", missing)