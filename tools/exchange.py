import requests


def convert_currency(amount: float, from_currency: str, to_currency: str) -> dict:
    url = f"https://open.er-api.com/v6/latest/{from_currency.upper()}"
    resp = requests.get(url, timeout=10).json()

    if resp.get("result") != "success":
        return {"error": f"Failed to fetch rates for {from_currency}"}

    rates = resp["rates"]
    to = to_currency.upper()

    if to not in rates:
        return {"error": f"Currency '{to}' not found"}

    rate = rates[to]
    converted = round(amount * rate, 4)

    return {
        "from": f"{amount} {from_currency.upper()}",
        "to": f"{converted} {to_currency.upper()}",
        "rate": rate,
        "last_updated": resp.get("time_last_update_utc", "")
    }