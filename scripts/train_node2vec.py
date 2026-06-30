import pandas as pd
import numpy as np
from pathlib import Path
from node2vec import Node2Vec
import networkx as nx

# -----------------------------
# Paths
# -----------------------------
ROOT = Path(__file__).resolve().parents[1]
GRAPH_PATH = ROOT / "graphs"
EMBED_PATH = ROOT / "embeddings"
EMBED_PATH.mkdir(parents=True, exist_ok=True)

# -----------------------------
# Load edges
# -----------------------------
edges = pd.read_csv(GRAPH_PATH / "edges_index.csv")

print("Edges:", edges.shape)

# -----------------------------
# Build graph
# -----------------------------
G = nx.from_pandas_edgelist(
    edges,
    source="src",
    target="dst"
)

print("Nodes:", G.number_of_nodes())
print("Edges:", G.number_of_edges())

# -----------------------------
# IMPORTANT FIX: reduce memory + disable parallelism
# -----------------------------
node2vec = Node2Vec(
    G,
    dimensions=128,
    walk_length=10,      # reduced
    num_walks=5,         # reduced
    p=1,
    q=1,
    workers=1,           # CRITICAL FIX (no multiprocessing)
    temp_folder=None
)

# -----------------------------
# Train embeddings
# -----------------------------
model = node2vec.fit(
    window=5,
    min_count=1,
    batch_words=512
)

# -----------------------------
# Extract embeddings safely
# -----------------------------
nodes = list(G.nodes())

embeddings = np.array([model.wv[str(n)] for n in nodes])

df = pd.DataFrame(embeddings)
df.insert(0, "node_index", nodes)

df.to_csv(EMBED_PATH / "node2vec_embeddings.csv", index=False)

print("Saved embeddings:", df.shape)
print("DONE")