import requests


def get_weather(city: str, units: str = "metric") -> dict:
    geo_url = "https://geocoding-api.open-meteo.com/v1/search"
    geo_resp = requests.get(geo_url, params={"name": city, "count": 1}, timeout=10).json()

    if not geo_resp.get("results"):
        return {"error": f"City '{city}' not found"}

    loc = geo_resp["results"][0]
    lat, lon = loc["latitude"], loc["longitude"]

    weather_url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "current": "temperature_2m,relative_humidity_2m,wind_speed_10m,weather_code",
        "temperature_unit": "fahrenheit" if units == "imperial" else "celsius",
        "wind_speed_unit": "mph" if units == "imperial" else "kmh",
    }

    weather = requests.get(weather_url, params=params, timeout=10).json()
    current = weather["current"]

    WEATHER_CODES = {
        0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
        45: "Fog", 51: "Light drizzle", 61: "Rain", 71: "Snow",
        80: "Rain showers", 95: "Thunderstorm",
    }

    code = current["weather_code"]
    description = WEATHER_CODES.get(code, WEATHER_CODES.get((code // 10) * 10, "Unknown"))

    return {
        "city": loc["name"],
        "country": loc.get("country", ""),
        "temperature": f"{current['temperature_2m']}°{'F' if units == 'imperial' else 'C'}",
        "humidity": f"{current['relative_humidity_2m']}%",
        "wind_speed": f"{current['wind_speed_10m']} {'mph' if units == 'imperial' else 'km/h'}",
        "condition": description,
    }