import pandas as pd
from pathlib import Path

# -----------------------------
# Paths
# -----------------------------
ROOT = Path(__file__).resolve().parents[1]

NODES_PATH = ROOT / "nodes"
EDGES_PATH = ROOT / "edges"
GRAPH_PATH = ROOT / "graphs"

GRAPH_PATH.mkdir(parents=True, exist_ok=True)

# -----------------------------
# Load nodes
# -----------------------------
papers = pd.read_csv(NODES_PATH / "papers.csv")
authors = pd.read_csv(NODES_PATH / "authors.csv")
journals = pd.read_csv(NODES_PATH / "journals.csv")
institutions = pd.read_csv(NODES_PATH / "institutions.csv")
mesh = pd.read_csv(NODES_PATH / "mesh_terms.csv")

# -----------------------------
# Build unified node mapping
# -----------------------------
all_nodes = pd.concat([
    papers[["node_id"]],
    authors[["node_id"]],
    journals[["node_id"]],
    institutions[["node_id"]],
    mesh[["node_id"]]
])

all_nodes = all_nodes.drop_duplicates().reset_index(drop=True)

all_nodes["node_index"] = range(len(all_nodes))

node2idx = dict(zip(all_nodes["node_id"], all_nodes["node_index"]))

all_nodes.to_csv(GRAPH_PATH / "node_mapping.csv", index=False)

print("Total nodes:", len(all_nodes))

# -----------------------------
# Load all edges
# -----------------------------
edge_files = [
    "paper_author.csv",
    "paper_journal.csv",
    "paper_mesh.csv",
    "author_institution.csv",
    "author_author.csv"
]

edges = []

for file in edge_files:
    df = pd.read_csv(EDGES_PATH / file)
    edges.append(df)

edges = pd.concat(edges, ignore_index=True)

print("Total edges:", len(edges))

# -----------------------------
# Convert to index-based edges
# -----------------------------
edges["src"] = edges["source"].map(node2idx)
edges["dst"] = edges["target"].map(node2idx)

edges = edges.dropna()

edges["src"] = edges["src"].astype(int)
edges["dst"] = edges["dst"].astype(int)

edges[["src", "dst"]].to_csv(GRAPH_PATH / "edges_index.csv", index=False)

print("Graph built successfully")
print("Saved to graphs/edges_index.csv")