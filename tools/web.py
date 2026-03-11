import requests
from bs4 import BeautifulSoup


def scrape_url(url: str, extract: str = "text") -> dict:
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers, timeout=10)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")

    for tag in soup(["script", "style", "nav", "footer"]):
        tag.decompose()

    if extract == "text":
        text = soup.get_text(separator="\n", strip=True)
        lines = [l for l in text.splitlines() if l.strip()]
        return {"url": url, "content": "\n".join(lines[:100])}

    elif extract == "links":
        links = [{"text": a.get_text(strip=True), "href": a["href"]}
                 for a in soup.find_all("a", href=True) if a["href"].startswith("http")]
        return {"url": url, "links": links[:20]}

    elif extract == "title":
        return {"url": url, "title": soup.title.string if soup.title else "No title"}