import time
import requests
import pandas as pd

from pathlib import Path
from tqdm import tqdm

from config import *

#reading the datasets
train = pd.read_csv(TRAIN_FILE)
existing = pd.read_csv(OPENALEX_FILE)

downloaded = set(existing["pmid"].astype(str))

remaining = train[
    ~train["pmid"].astype(str).isin(downloaded)
].copy()

remaining["pmid"] = remaining["pmid"].astype(str)

print(f"Already downloaded : {len(downloaded)}")
print(f"Remaining          : {len(remaining)}")

#creating batches
BATCH_SIZE = 100

pmids = remaining["pmid"].tolist()

batches = [
    pmids[i:i+BATCH_SIZE]
    for i in range(0, len(pmids), BATCH_SIZE)
]

print("Number of batches:", len(batches))

#function to download one batch
def download_batch(pmids):

    ids = "|".join(pmids)

    url = (
        "https://api.openalex.org/works"
        f"?filter=pmid:{ids}"
        f"&per-page=100"
        f"&mailto={EMAIL}"
    )

    headers = {
        "User-Agent": f"ScientificReliability ({EMAIL})"
    }

    response = requests.get(
        url,
        headers=headers,
        timeout=60
    )

    if response.status_code != 200:
        print("Status:", response.status_code)
        print(response.text)
        return []

    works = response.json()["results"]

    records = []

    for work in works:

        ids = work.get("ids", {})

        pmid = ids.get("pmid")

        if pmid is not None:
            pmid = pmid.split("/")[-1]

        records.append({

            "pmid": pmid,

            "openalex_id": work.get("id"),

            "doi": work.get("doi"),

            "publication_year": work.get("publication_year"),

            "cited_by_count": work.get("cited_by_count"),

            "referenced_works": work.get("referenced_works")

        })

    return records

sample = download_batch(batches[0])

print("\nReturned records:", len(sample))

if sample:
    print("\nFirst record:")
    print(sample[0])