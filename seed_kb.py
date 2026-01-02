import json
import requests
import os

API_BASE = "https://vinnies-brain-backend.onrender.com"
ADMIN_KEY = os.getenv("ADMIN_API_KEY")

with open("seed_articles.json", "r") as f:
    articles = json.load(f)

for a in articles:
    r = requests.post(
        f"{API_BASE}/v1/admin/kb/upsert",
        headers={
            "Content-Type": "application/json",
            "x-admin-key": ADMIN_KEY,
        },
        json=a,
        timeout=30,
    )

    if r.status_code != 200:
        print("❌ Failed:", r.text)
    else:
        print("✅ Inserted:", a["title"])
