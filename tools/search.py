import requests
from bs4 import BeautifulSoup


def search_web(query: str, count: int = 5) -> list:
    headers = {"User-Agent": "Mozilla/5.0"}
    url = "https://html.duckduckgo.com/html/"
    resp = requests.post(url, data={"q": query}, headers=headers, timeout=10)
    soup = BeautifulSoup(resp.text, "html.parser")

    results = []
    for result in soup.select(".result")[:count]:
        title_el = result.select_one(".result__title")
        snippet_el = result.select_one(".result__snippet")
        link_el = result.select_one(".result__url")

        if title_el:
            results.append({
                "title": title_el.get_text(strip=True),
                "snippet": snippet_el.get_text(strip=True) if snippet_el else "",
                "url": link_el.get_text(strip=True) if link_el else "",
            })

    return results