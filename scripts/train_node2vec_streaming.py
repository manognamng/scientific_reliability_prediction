import pandas as pd
import numpy as np
import random
from pathlib import Path
from gensim.models import Word2Vec
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

G = nx.from_pandas_edgelist(
    edges,
    source="src",
    target="dst"
)

print("Graph ready:", G.number_of_nodes(), G.number_of_edges())

# -----------------------------
# Random walk generator (STREAMING)
# -----------------------------
def random_walk(graph, start_node, walk_length=10):
    walk = [str(start_node)]

    for _ in range(walk_length - 1):
        neighbors = list(graph.neighbors(int(walk[-1])))
        if len(neighbors) == 0:
            break
        walk.append(str(random.choice(neighbors)))

    return walk

# -----------------------------
# Generate walks WITHOUT storing all in memory
# -----------------------------
walks = []

nodes = list(G.nodes())

print("Generating walks...")

for i in range(5):  # num_walks
    random.shuffle(nodes)

    for node in nodes:
        walks.append(random_walk(G, node, walk_length=10))

print("Total walks:", len(walks))

# -----------------------------
# Train Word2Vec (Node2Vec equivalent)
# -----------------------------
print("Training embeddings...")

model = Word2Vec(
    sentences=walks,
    vector_size=128,
    window=5,
    min_count=1,
    sg=1,
    workers=4,
    epochs=5
)

# -----------------------------
# Save embeddings
# -----------------------------
node_ids = list(G.nodes())

embeddings = np.array([
    model.wv[str(n)] for n in node_ids
])

df = pd.DataFrame(embeddings)
df.insert(0, "node_index", node_ids)

df.to_csv(EMBED_PATH / "node2vec_embeddings.csv", index=False)

print("Saved:", df.shape)
print("DONE")