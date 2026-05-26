import os
import httpx
from cache.cache_store import get, set, TTL_GEOCODE

OWM_GEO_URL = "http://api.openweathermap.org/geo/1.0/direct"

_http = httpx.Client(verify=False, timeout=10)


def get_coordinates(city: str) -> dict | None:
    """
    Convert a city name to {lat, lon, display_name}.
    Returns None if city cannot be resolved.
    Cached for 7 days — cities don't move.
    """
    cache_key = f"geocode_{city.lower().strip()}"
    cached = get(cache_key, TTL_GEOCODE)
    if cached:
        return cached

    api_key = os.getenv("OPENWEATHER_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENWEATHER_API_KEY not set in .env")

    try:
        resp = _http.get(OWM_GEO_URL, params={
            "q":     city,
            "limit": 1,
            "appid": api_key,
        })
        resp.raise_for_status()
        results = resp.json()
    except Exception as e:
        return _llm_fallback(city, error=str(e))

    if not results:
        return _llm_fallback(city, error="no results")

    first = results[0]
    data = {
        "lat":          first["lat"],
        "lon":          first["lon"],
        "display_name": f"{first.get('name', city)}, {first.get('state', '')}, {first.get('country', '')}".strip(", "),
        "source":       "openweathermap",
    }
    set(cache_key, data)
    return data


def _llm_fallback(city: str, error: str) -> dict | None:
    """
    When OWM geocoding fails, ask the LLM to estimate lat/lon from
    its training knowledge. Marked as approximate in source field.
    """
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import HumanMessage

    llm = ChatOpenAI(
        model="openai/gpt-4o-mini",
        openai_api_key=os.getenv("OPENROUTER_API_KEY") or "none",
        openai_api_base="https://openrouter.ai/api/v1",
        http_client=_http,
    )
    prompt = (
        f"Give me the approximate latitude and longitude for '{city}' in India. "
        "Reply with ONLY two numbers separated by a comma: lat,lon. No other text."
    )
    try:
        reply = llm.invoke([HumanMessage(content=prompt)]).content.strip()
        lat_str, lon_str = reply.split(",")
        return {
            "lat":          float(lat_str.strip()),
            "lon":          float(lon_str.strip()),
            "display_name": city,
            "source":       "llm_fallback",
        }
    except Exception:
        return None
