import requests
import json
import os

BASE_URL = "https://discourse.onlinedegree.iitm.ac.in"
CATEGORY = 34
OUTPUT_DIR = "data/discourse"
OUTPUT = f"{OUTPUT_DIR}/tds_kb_posts.json"

# ✅ Create the directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

def fetch_all_posts():
    all_posts = []
    for page in range(0, 10):  # adjust pages as needed
        url = f"{BASE_URL}/c/courses/tds-kb/{CATEGORY}.json?page={page}"
        res = requests.get(url)
        if not res.ok: break
        topics = res.json().get("topic_list", {}).get("topics", [])
        for topic in topics:
            topic_id = topic["id"]
            post_url = f"{BASE_URL}/t/{topic_id}.json"
            post_data = requests.get(post_url).json()
            all_posts.append({
                "id": topic_id,
                "title": post_data["title"],
                "posts": [p["cooked"] for p in post_data["post_stream"]["posts"]]
            })
    with open(OUTPUT, "w", encoding="utf-8") as f:
        json.dump(all_posts, f, indent=2)

fetch_all_posts()

