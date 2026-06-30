import pandas as pd

pmid = 38362586

papers = pd.read_csv("nodes/papers.csv")
mapping = pd.read_csv("graphs/node_mapping.csv")
node2vec = pd.read_csv("embeddings/node2vec_embeddings.csv")

merged = papers.merge(mapping,on="node_id")

row = merged[merged["pmid"]==pmid]

print(row)

idx = row.iloc[0]["node_index"]

print("\nNode index:", idx)

print("\nEmbedding exists?")

print(node2vec[node2vec["node_index"]==idx])