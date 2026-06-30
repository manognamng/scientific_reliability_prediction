from pathlib import Path

# -----------------------------
# Project directories
# -----------------------------

PROJECT_ROOT = Path(r"C:\Users\Admin\scientific_reliability2")

DATA_DIR = PROJECT_ROOT / "data"

CACHE_DIR = PROJECT_ROOT / "cache"

OPENALEX_CACHE = CACHE_DIR / "openalex"

LOG_DIR = PROJECT_ROOT / "logs"

# -----------------------------
# Files
# -----------------------------

TRAIN_FILE = DATA_DIR / "train.csv"

PUBMED_FILE = DATA_DIR / "pubmed_metadata.csv"

OPENALEX_FILE = DATA_DIR / "openalex_metadata.csv"

# -----------------------------
# OpenAlex
# -----------------------------

EMAIL = "manognasony1998@gmail.com"

REQUEST_DELAY = 0.25

TIMEOUT = 30

MAX_RETRIES = 5

SAVE_EVERY = 25