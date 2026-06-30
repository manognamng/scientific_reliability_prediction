import os
import pandas as pd
import ast
from pathlib import Path

# -----------------------------
# Paths (project root aware)
# -----------------------------
ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "data"
NODES_PATH = ROOT / "nodes"

TRAIN_FILE = DATA_PATH / "train.csv"
PUBMED_FILE = DATA_PATH / "pubmed_metadata.csv"

NODES_PATH.mkdir(parents=True, exist_ok=True)

# -----------------------------
# Load data
# -----------------------------
print("Loading datasets...")

train = pd.read_csv(TRAIN_FILE)
pubmed = pd.read_csv(PUBMED_FILE)

print("Train:", train.shape)
print("PubMed:", pubmed.shape)

# -----------------------------
# Clean list-like columns safely
# -----------------------------
def safe_parse(x):
    if pd.isna(x):
        return []
    if isinstance(x, list):
        return x
    try:
        return ast.literal_eval(x)
    except:
        return []

pubmed["authors"] = pubmed["authors"].apply(safe_parse)
pubmed["mesh_terms"] = pubmed["mesh_terms"].apply(safe_parse)
pubmed["affiliations"] = pubmed["affiliations"].apply(safe_parse)

# -----------------------------
# PAPER NODES
# -----------------------------
paper_nodes = pubmed.copy()

paper_nodes["node_id"] = "paper_" + paper_nodes["pmid"].astype(str)

paper_nodes = paper_nodes[
    [
        "node_id",
        "pmid",
        "title",
        "journal",
        "year",
        "doi",
        "n_authors",
        "n_mesh_terms"
    ]
]

paper_nodes.to_csv(NODES_PATH / "papers.csv", index=False)
print("Saved papers.csv:", paper_nodes.shape)

# -----------------------------
# AUTHOR NODES
# -----------------------------
authors = set()

for lst in pubmed["authors"]:
    for a in lst:
        authors.add(a.strip())

author_nodes = pd.DataFrame({
    "node_id": ["author_" + a for a in authors],
    "author_name": list(authors)
})

author_nodes.to_csv(NODES_PATH / "authors.csv", index=False)
print("Saved authors.csv:", author_nodes.shape)

# -----------------------------
# JOURNAL NODES
# -----------------------------
journals = pubmed["journal"].dropna().unique()

journal_nodes = pd.DataFrame({
    "node_id": ["journal_" + j for j in journals],
    "journal_name": journals
})

journal_nodes.to_csv(NODES_PATH / "journals.csv", index=False)
print("Saved journals.csv:", journal_nodes.shape)

# -----------------------------
# INSTITUTION NODES
# -----------------------------
institutions = set()

for lst in pubmed["affiliations"]:
    for i in lst:
        institutions.add(i.strip())

institution_nodes = pd.DataFrame({
    "node_id": ["inst_" + i for i in institutions],
    "institution_name": list(institutions)
})

institution_nodes.to_csv(NODES_PATH / "institutions.csv", index=False)
print("Saved institutions.csv:", institution_nodes.shape)

# -----------------------------
# MESH TERMS NODES
# -----------------------------
mesh_terms = set()

for lst in pubmed["mesh_terms"]:
    for m in lst:
        mesh_terms.add(m.strip())

mesh_nodes = pd.DataFrame({
    "node_id": ["mesh_" + m for m in mesh_terms],
    "mesh_term": list(mesh_terms)
})

mesh_nodes.to_csv(NODES_PATH / "mesh_terms.csv", index=False)
print("Saved mesh_terms.csv:", mesh_nodes.shape)

print("\nNODE CREATION COMPLETE")