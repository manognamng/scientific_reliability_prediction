import pandas as pd
from pathlib import Path

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import joblib

ROOT = Path(__file__).resolve().parents[1]

df = pd.read_csv(ROOT/"data"/"training_dataset.csv")


print(df.shape)

# -------------------------
# label
# -------------------------

y = df["label"]

X = df.drop(columns=["label"])

# -------------------------
# SciBERT columns
# -------------------------

node_cols = [f"{i}_x" for i in range(128)]

scibert_cols = []

for c in X.columns:

    if c.endswith("_x"):
        continue

    if c in ["year","n_authors","n_mesh_terms","journal_encoded"]:
        continue

    scibert_cols.append(c)

print()

print("Node2Vec:",len(node_cols))
print("SciBERT :",len(scibert_cols))

metadata = X[
    [
        "year",
        "n_authors",
        "n_mesh_terms",
        "journal_encoded"
    ]
]

node = X[node_cols]

scibert = X[scibert_cols]

# -------------------------
# Scale only SciBERT
# -------------------------

scaler = StandardScaler()

scibert_scaled = scaler.fit_transform(scibert)

print("Running PCA...")

pca = PCA(
    n_components=0.95,
    random_state=42
)

scibert_pca = pca.fit_transform(scibert_scaled)

print()

print("Original:",scibert.shape)
print("Reduced :",scibert_pca.shape)

joblib.dump(
    scaler,
    ROOT/"models"/"scaler.pkl"
)

joblib.dump(
    pca,
    ROOT/"models"/"pca.pkl"
)

pca_df = pd.DataFrame(
    scibert_pca,
    columns=[
        f"pca_{i}"
        for i in range(scibert_pca.shape[1])
    ]
)

final = pd.concat(
    [
        metadata.reset_index(drop=True),
        node.reset_index(drop=True),
        pca_df.reset_index(drop=True),
        y.reset_index(drop=True)
    ],
    axis=1
)

print(final.shape)

final.to_csv(
    ROOT/"data"/"training_dataset_pca.csv",
    index=False
)

print("Saved.")