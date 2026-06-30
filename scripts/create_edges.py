import os
import pandas as pd
import ast
from pathlib import Path
from itertools import combinations

# -----------------------------
# Paths
# -----------------------------
ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "data"
NODES_PATH = ROOT / "nodes"
EDGES_PATH = ROOT / "edges"

EDGES_PATH.mkdir(parents=True, exist_ok=True)

PUBMED_FILE = DATA_PATH / "pubmed_metadata.csv"

# -----------------------------
# Load data
# -----------------------------
print("Loading PubMed metadata...")

pubmed = pd.read_csv(PUBMED_FILE)

print("Shape:", pubmed.shape)

# -----------------------------
# Safe parsing
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
# 1. PAPER → AUTHOR
# -----------------------------
paper_author = []

for _, row in pubmed.iterrows():
    pmid = row["pmid"]
    for a in row["authors"]:
        paper_author.append([
            f"paper_{pmid}",
            f"author_{a.strip()}"
        ])

paper_author_df = pd.DataFrame(paper_author, columns=["source", "target"])
paper_author_df.to_csv(EDGES_PATH / "paper_author.csv", index=False)

print("Saved paper_author:", paper_author_df.shape)

# -----------------------------
# 2. PAPER → JOURNAL
# -----------------------------
paper_journal = pubmed[["pmid", "journal"]].dropna()

paper_journal["source"] = "paper_" + paper_journal["pmid"].astype(str)
paper_journal["target"] = "journal_" + paper_journal["journal"].astype(str)

paper_journal_df = paper_journal[["source", "target"]]
paper_journal_df.to_csv(EDGES_PATH / "paper_journal.csv", index=False)

print("Saved paper_journal:", paper_journal_df.shape)

# -----------------------------
# 3. PAPER → MESH
# -----------------------------
paper_mesh = []

for _, row in pubmed.iterrows():
    pmid = row["pmid"]
    for m in row["mesh_terms"]:
        paper_mesh.append([
            f"paper_{pmid}",
            f"mesh_{m.strip()}"
        ])

paper_mesh_df = pd.DataFrame(paper_mesh, columns=["source", "target"])
paper_mesh_df.to_csv(EDGES_PATH / "paper_mesh.csv", index=False)

print("Saved paper_mesh:", paper_mesh_df.shape)

# -----------------------------
# 4. AUTHOR → INSTITUTION
# -----------------------------
author_inst = []

for _, row in pubmed.iterrows():
    authors = row["authors"]
    insts = row["affiliations"]

    for a in authors:
        for i in insts:
            author_inst.append([
                f"author_{a.strip()}",
                f"inst_{i.strip()}"
            ])

author_inst_df = pd.DataFrame(author_inst, columns=["source", "target"])
author_inst_df.to_csv(EDGES_PATH / "author_institution.csv", index=False)

print("Saved author_institution:", author_inst_df.shape)

# -----------------------------
# 5. AUTHOR ↔ AUTHOR (Co-authorship)
# -----------------------------
coauthor_edges = set()

for _, row in pubmed.iterrows():
    authors = [a.strip() for a in row["authors"]]

    for a1, a2 in combinations(authors, 2):
        coauthor_edges.add((f"author_{a1}", f"author_{a2}"))
        coauthor_edges.add((f"author_{a2}", f"author_{a1}"))

coauthor_df = pd.DataFrame(list(coauthor_edges), columns=["source", "target"])
coauthor_df.to_csv(EDGES_PATH / "author_author.csv", index=False)

print("Saved author_author:", coauthor_df.shape)

print("\nEDGE CREATION COMPLETE")