import pandas as pd
from pathlib import Path
from sklearn.preprocessing import LabelEncoder

ROOT = Path(__file__).resolve().parents[1]

# -------------------------
# Load files
# -------------------------

train = pd.read_csv(ROOT / "data" / "train.csv")

papers = pd.read_csv(ROOT / "nodes" / "papers.csv")

mapping = pd.read_csv(ROOT / "graphs" / "node_mapping.csv")

node2vec = pd.read_csv(ROOT / "embeddings" / "node2vec_embeddings.csv")

scibert = pd.read_csv(ROOT / "embeddings" / "scibert_embeddings.csv")

print("Train:", train.shape)
print("Papers:", papers.shape)
print("Mapping:", mapping.shape)
print("Node2Vec:", node2vec.shape)
print("SciBERT:", scibert.shape)
# -------------------------
# Force merge keys to same dtype
# -------------------------

train["pmid"] = train["pmid"].astype(str)
papers["pmid"] = papers["pmid"].astype(str)
scibert["pmid"] = scibert["pmid"].astype(str)

mapping["node_id"] = mapping["node_id"].astype(str)

node2vec["node_index"] = node2vec["node_index"].astype(int)

print("\n===== DATA TYPES =====")
print("train pmid:", train["pmid"].dtype)
print("papers pmid:", papers["pmid"].dtype)

# -------------------------
# Merge papers
# -------------------------

df = train.merge(
    papers,
    on="pmid",
    how="left"
)

print(df.shape)

print("\nAfter Papers Merge")

print("Missing node_id:", df["node_id"].isna().sum())
print(df["node_id"].dtype)
print(mapping["node_id"].dtype)

# -------------------------
# Merge node mapping
# -------------------------

df = df.merge(
    mapping,
    on="node_id",
    how="left"
)

print(df.shape)

print("\nAfter Mapping Merge")

print("Missing node_index:", df["node_index"].isna().sum())

print(df["node_index"].dtype)
print(node2vec["node_index"].dtype)

# -------------------------
# Merge Node2Vec
# -------------------------

df = df.merge(
    node2vec,
    on="node_index",
    how="left"
)

print(df.shape)

node2vec_cols = node2vec.columns.drop("node_index")

print("\nAfter Node2Vec Merge")

print(
    "Missing Node2Vec:",
    df[node2vec_cols].isna().all(axis=1).sum()
)

# -------------------------
# Merge SciBERT
# -------------------------

df = df.merge(
    scibert,
    on="pmid",
    how="left"
)

print(df.shape)

print(df["pmid"].dtype)

print(scibert["pmid"].dtype)

#scibert_cols = scibert.columns.drop("pmid")

#print("\nAfter SciBERT Merge")

#print(
#    "Missing SciBERT:",
#    df[scibert_cols].isna().all(axis=1).sum()
#)
print(df.shape)

print("Columns after SciBERT merge:")
print(df.columns[:20])
print(df.columns[-20:])

dup = df.columns[df.columns.duplicated()]

print("\nDuplicate columns:")
print(list(dup))

print("Number of duplicate columns:", len(dup))

# -------------------------
# Encode journal
# -------------------------

df["journal"] = df["journal"].fillna("Unknown")

encoder = LabelEncoder()

df["journal_encoded"] = encoder.fit_transform(df["journal"])

# -------------------------
# Fill numeric NA
# -------------------------

for col in ["year", "n_authors", "n_mesh_terms"]:

    df[col] = df[col].fillna(df[col].median())

# -------------------------
# Save
# -------------------------


OUT = ROOT / "data" / "final_dataset.csv"

# Fill missing embeddings

#node_cols = [c for c in df.columns if c.endswith("_x")]
# -------------------------
# Fill missing embeddings
# -------------------------

# Node2Vec columns
node_cols = [c for c in df.columns if c.endswith("_x") and c[:-2].isdigit()]

# SciBERT columns
sci_cols = []

for c in df.columns:

    if c.endswith("_y") and c[:-2].isdigit():
        sci_cols.append(c)

    elif c.isdigit():
        sci_cols.append(c)

print("Node2Vec columns:", len(node_cols))
print("SciBERT columns:", len(sci_cols))

print("Number of Sci cols:", len(sci_cols))
print("Unique Sci cols:", len(set(sci_cols)))
print("Selected dataframe shape:", df[sci_cols].shape)

print(df[sci_cols].columns[df[sci_cols].columns.duplicated()])

df.loc[:, node_cols] = df[node_cols].fillna(0)
df.loc[:, sci_cols] = df[sci_cols].fillna(0)

df["doi"] = df["doi"].fillna("")

df.to_csv(OUT, index=False)

print()
print("Saved:", OUT)

print(df.shape)