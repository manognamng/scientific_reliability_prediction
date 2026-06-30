import pandas as pd

train = pd.read_csv("data/train.csv")
papers = pd.read_csv("nodes/paper_nodes.csv")

print("Train PMIDs:", train["pmid"].nunique())
print("Paper PMIDs:", papers["pmid"].nunique())

missing = set(train["pmid"]) - set(papers["pmid"])

print("Missing:", len(missing))

print(list(missing)[:20])