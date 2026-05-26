import os
import httpx
from cache.cache_store import get, set, TTL_WEATHER

OWM_FORECAST_URL = "https://api.openweathermap.org/data/2.5/forecast"
_http = httpx.Client(verify=False, timeout=10)


def get_forecast(lat: float, lon: float, start_date: str, end_date: str) -> dict | None:
    """
    Fetch 5-day / 3-hour forecast from OWM for the given coordinates.
    Filters entries to the travel date window.
    Cached for 1 hour.
    """
    cache_key = f"weather_{round(lat,2)}_{round(lon,2)}_{start_date}_{end_date}"
    cached = get(cache_key, TTL_WEATHER)
    if cached:
        return {**cached, "source": "cache"}

    api_key = os.getenv("OPENWEATHER_API_KEY")
    if not api_key:
        return None

    try:
        resp = _http.get(OWM_FORECAST_URL, params={
            "lat":   lat,
            "lon":   lon,
            "appid": api_key,
            "units": "metric",
            "cnt":   40,          # 5 days × 8 three-hour slots
        })
        resp.raise_for_status()
        raw = resp.json()
    except Exception:
        return None

    # Extract entries within travel window
    entries = [
        {
            "dt_txt":      e["dt_txt"],
            "temp":        e["main"]["temp"],
            "feels_like":  e["main"]["feels_like"],
            "humidity":    e["main"]["humidity"],
            "weather":     e["weather"][0]["description"],
            "rain_mm":     e.get("rain", {}).get("3h", 0),
            "wind_kph":    round(e["wind"]["speed"] * 3.6, 1),
        }
        for e in raw.get("list", [])
        if start_date <= e["dt_txt"][:10] <= end_date
    ]

    data = {
        "city":    raw.get("city", {}).get("name", ""),
        "country": raw.get("city", {}).get("country", ""),
        "entries": entries,
        "source":  "openweathermap",
    }
    set(cache_key, data)
    return data
