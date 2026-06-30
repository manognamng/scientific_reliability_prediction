import time
import requests
import pandas as pd

from tqdm import tqdm

from config import *

#loading existing data
train = pd.read_csv(TRAIN_FILE)

existing = pd.read_csv(OPENALEX_FILE)

print("Train:", train.shape)

print("Existing:", existing.shape)

downloaded = set(existing["pmid"].astype(str))

all_pmids = train["pmid"].astype(str)

remaining = all_pmids[~all_pmids.isin(downloaded)]

print()
print("Already downloaded:", len(downloaded))
print("Remaining:", len(remaining))

def download_one_pmid(pmid):

    url = (
        f"https://api.openalex.org/works"
        f"?filter=pmid:{pmid}"
        f"&mailto={EMAIL}"
    )

    headers = {
        "User-Agent": f"ScientificReliability ({EMAIL})"
    }

    try:
        for attempt in range(MAX_RETRIES):

            try:

                response = requests.get(
                    url,
                    headers=headers,
                    timeout=TIMEOUT
                )

                if response.status_code == 200:
                    break

                if response.status_code == 429:
                    print("Rate limited. Waiting 60 seconds...")
                    time.sleep(60)
                    continue

                print(f"HTTP {response.status_code} for PMID {pmid}")
                return None

            except requests.exceptions.RequestException as e:

                print(f"Attempt {attempt + 1} failed for PMID {pmid}: {e}")

                time.sleep(2)

        else:
            return None

        if response.status_code != 200:
            print(f"Error {response.status_code} for PMID {pmid}")
            return None

        results = response.json()["results"]

        if len(results) == 0:
            return None

        work = results[0]

        return {
            "pmid": pmid,
            "openalex_id": work.get("id"),
            "doi": work.get("doi"),
            "publication_year": work.get("publication_year"),
            "cited_by_count": work.get("cited_by_count"),
            "referenced_works": work.get("referenced_works")
        }

    except Exception as e:
        print(f"Exception for PMID {pmid}: {e}")
        return None
    
new_records = []

for i, pmid in enumerate(tqdm(remaining, desc="Downloading OpenAlex")):

    record = download_one_pmid(pmid)

    if record is not None:
        new_records.append(record)

    # Save progress every 25 successful downloads
    if len(new_records) >= SAVE_EVERY:

        new_df = pd.DataFrame(new_records)

        existing = pd.concat([existing, new_df], ignore_index=True)

        existing.to_csv(OPENALEX_FILE, index=False)

        print(f"\nSaved {len(existing)} records")

        new_records = []

    time.sleep(REQUEST_DELAY)

# Save any remaining records

if len(new_records) > 0:

    new_df = pd.DataFrame(new_records)

    existing = pd.concat([existing, new_df], ignore_index=True)

    existing.to_csv(OPENALEX_FILE, index=False)

print("\nDownload complete!")

print("Total records:", len(existing))