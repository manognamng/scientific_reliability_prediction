import requests

EMAIL = "your_email@example.com"   # Replace with your real email

pmid = "28640836"

url = (
    f"https://api.openalex.org/works"
    f"?filter=pmid:{pmid}"
    f"&mailto={EMAIL}"
)

headers = {
    "User-Agent": f"ScientificReliabilityProject ({EMAIL})"
}

response = requests.get(url, headers=headers)

print("Status:", response.status_code)

if response.status_code == 200:
    data = response.json()
    print("Results:", len(data["results"]))

    if data["results"]:
        work = data["results"][0]
        print("OpenAlex ID:", work["id"])
        print("Title:", work["display_name"])
        print("Citations:", work["cited_by_count"])
else:
    print(response.text)