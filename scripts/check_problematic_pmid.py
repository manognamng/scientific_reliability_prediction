import pandas as pd

papers = pd.read_csv("nodes/papers.csv")

print(papers[papers["pmid"] == 38362586])