import pandas as pd

pmid = "38362586"

papers = pd.read_csv("nodes/papers.csv")
mapping = pd.read_csv("graphs/node_mapping.csv")
node2vec = pd.read_csv("embeddings/node2vec_embeddings.csv")
scibert = pd.read_csv("embeddings/scibert_embeddings.csv")

print("\nPapers")
print(papers[papers["pmid"].astype(str) == pmid])

print("\nSciBERT")
print(scibert[scibert["pmid"].astype(str) == pmid])

paper = papers[papers["pmid"].astype(str) == pmid]

if len(paper):

    node_id = paper.iloc[0]["node_id"]

    print("\nNode ID:", node_id)

    m = mapping[mapping["node_id"] == node_id]

    print("\nMapping")
    print(m)

    if len(m):

        idx = m.iloc[0]["node_index"]

        print("\nNode2Vec")

        print(node2vec[node2vec["node_index"] == idx])