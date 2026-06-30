import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from pathlib import Path
from tqdm import tqdm
import gc

# -----------------------------
# Paths
# -----------------------------
ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "data"
EMBED_PATH = ROOT / "embeddings"
EMBED_PATH.mkdir(parents=True, exist_ok=True)

PUBMED_FILE = DATA_PATH / "pubmed_metadata.csv"

# -----------------------------
# Load data
# -----------------------------
df = pd.read_csv(PUBMED_FILE)

print("Loaded:", df.shape)

# Fill missing text safely
df["title"] = df["title"].fillna("")
df["abstract"] = df["abstract"].fillna("")

texts = (df["title"] + " " + df["abstract"]).tolist()

# -----------------------------
# Load SciBERT
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", device)

tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
model = AutoModel.from_pretrained("allenai/scibert_scivocab_uncased")
model = model.to(device)
model.eval()

# -----------------------------
# Mean pooling function
# -----------------------------
def mean_pooling(output, attention_mask):
    token_embeddings = output.last_hidden_state
    mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    summed = torch.sum(token_embeddings * mask, dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / counts

# -----------------------------
# Batch encoding
# -----------------------------
batch_size = 4
embeddings = []

for i in tqdm(range(0, len(texts), batch_size)):
    batch_texts = texts[i:i + batch_size]

    encoded = tokenizer(
        batch_texts,
        padding=True,
        truncation=True,
        max_length=256,
        return_tensors="pt"
    )

    encoded = {k: v.to(device) for k, v in encoded.items()}

    with torch.inference_mode():
        output = model(**encoded)

    batch_emb = mean_pooling(output, encoded["attention_mask"])
    embeddings.append(batch_emb.cpu().numpy())

    del output
    del encoded
    gc.collect()
# -----------------------------
# Stack embeddings
# -----------------------------
embeddings = np.vstack(embeddings)

print("Final shape:", embeddings.shape)

# -----------------------------
# Save
# -----------------------------
out = pd.DataFrame(embeddings)
out.insert(0, "pmid", df["pmid"].values)

out.to_csv(EMBED_PATH / "scibert_embeddings.csv", index=False)

print("Saved: embeddings/scibert_embeddings.csv")