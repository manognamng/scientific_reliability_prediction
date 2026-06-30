import pandas as pd

pmid = 38362586

papers = pd.read_csv("nodes/papers.csv")
mapping = pd.read_csv("graphs/node_mapping.csv")

paper = papers[papers["pmid"] == pmid]

print(paper)

print("\nMerge with mapping")

merged = paper.merge(mapping, on="node_id", how="left")

print(merged)