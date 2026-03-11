import requests

def wikipedia_summary(topic: str, sentences: int = 3, lang: str = "en") -> dict:
    try:
        url = f"https://{lang}.wikipedia.org/api/rest_v1/page/summary/{topic.replace(' ', '_')}"
        resp = requests.get(url, timeout=10)

        if resp.status_code == 404:
            return {"error": f"No Wikipedia article found for '{topic}'"}

        data = resp.json()
        extract = data.get("extract", "")
        parts = extract.split(". ")
        summary = ". ".join(parts[:sentences]) + ("." if len(parts) > sentences else "")

        return {
            "title": data.get("title"),
            "summary": summary,
            "url": data.get("content_urls", {}).get("desktop", {}).get("page", "")
        }
    except Exception as e:
        return {"error": f"Wikipedia unavailable: {str(e)}"}