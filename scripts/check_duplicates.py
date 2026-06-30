import pandas as pd

papers = pd.read_csv("nodes/paper_nodes.csv")

dup = papers[papers.duplicated("pmid", keep=False)]

print("Duplicate PMIDs:", dup.shape)

print(dup.head(20))