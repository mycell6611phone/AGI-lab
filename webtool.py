# webtool.py

import requests
from config import settings  # ✅ RIGHT


def google_search(query, num_results=3):
    url = "https://www.googleapis.com/customsearch/v1"
    params = {
    "q": query,
    "key": settings.google_api_key,
    "cx": settings.google_cse_id,
    "num": num_results,
}

    resp = requests.get(url, params=params, timeout=20)
    if not resp.ok:
        return f"[Search failed: {resp.status_code} {resp.text}]"
    data = resp.json()
    results = []
    for item in data.get("items", []):
        title = item.get("title", "")
        snippet = item.get("snippet", "")
        link = item.get("link", "")
        results.append(f"{title}\n{snippet}\n{link}")
    return "\n\n".join(results) if results else "No results found."
