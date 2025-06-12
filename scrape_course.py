# scrape_course.py
import requests
from bs4 import BeautifulSoup
import os

BASE_URL = "https://tds.s-anand.net/#/2025-01/"
EXPORT_DIR = "data/course/"

if not os.path.exists(EXPORT_DIR):
    os.makedirs(EXPORT_DIR)

def get_course_links():
    html = requests.get("https://tds.s-anand.net/").text
    soup = BeautifulSoup(html, "html.parser")
    return [
        "https://tds.s-anand.net/" + a['href']
        for a in soup.find_all("a")
        if "/#/2025-01/" in a['href']
    ]

def save_text(url):
    print("Fetching", url)
    text = requests.get(url.replace('#/', '')).text
    filename = url.split("/")[-1] + ".html"
    with open(EXPORT_DIR + filename, "w", encoding="utf-8") as f:
        f.write(text)

for url in get_course_links():
    save_text(url)
