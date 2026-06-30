import pandas as pd

emb = pd.read_csv("embeddings/scibert_embeddings.csv")

print(emb.shape)

print("Duplicate PMIDs")

print(emb[emb.duplicated("pmid", keep=False)])