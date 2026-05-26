import os
import httpx
from datetime import date
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

_http = httpx.Client(verify=False)
_http_async = httpx.AsyncClient(verify=False)
_llm = None


def _get_llm():
    global _llm
    if _llm is None:
        _llm = ChatOpenAI(
            model="openai/gpt-oss-120b:free",
            temperature=0.5,
            max_tokens=4096,
            openai_api_key=os.getenv("OPENROUTER_API_KEY"),
            openai_api_base="https://openrouter.ai/api/v1",
            http_client=_http,
            http_async_client=_http_async,
        )
    return _llm


class PlaceItem(BaseModel):
    name:           str   = Field(description="Place or restaurant name")
    description:    str   = Field(description="1-sentence description")
    entry_fee_inr:  int   = Field(default=0, description="Typical entry fee, 0 if free")
    duration_hours: float = Field(description="Typical visit duration in hours")
    best_time:      str   = Field(description="Morning | Afternoon | Evening | Anytime")


class PlacesDiscovery(BaseModel):
    must_visit:   list[PlaceItem] = Field(description="3-5 unmissable attractions")
    food_spots:   list[PlaceItem] = Field(description="3-4 local food experiences")
    activities:   list[PlaceItem] = Field(description="2-4 activities matching travel_type")
    hidden_gems:  list[PlaceItem] = Field(description="1-2 off-the-beaten-path spots")
    packing_tips: list[str]       = Field(description="3-5 destination-specific packing tips")


PLACES_SYSTEM = """
You are a local travel insider for Indian destinations — not a tourist brochure.

────────────────────────────────────────────────
SUGGESTION GUIDELINES
────────────────────────────────────────────────
  must_visit  : 3–5 genuinely unmissable spots — famous ones are fine if they deserve it
  food_spots  : 3–4 local food experiences — street stalls, famous dhabas, signature dishes
  activities  : 2–4 experiences tailored to the travel_type:
                  beach     → snorkelling, sunset cruise, parasailing
                  hill      → trekking trails, viewpoints, camping
                  city      → heritage walks, markets, cultural shows
                  adventure → rappelling, white-water rafting, zip-lining
  hidden_gems : 1–2 lesser-known spots that most tourists miss — must be real and reachable

────────────────────────────────────────────────
DATA ACCURACY RULES
────────────────────────────────────────────────
  • entry_fee_inr  : realistic 2024–25 figure (0 if free)
  • duration_hours : how long a typical visitor actually spends there
  • best_time      : when to visit for best experience — morning light, golden hour, less crowd

────────────────────────────────────────────────
SENSITIVITY RULES
────────────────────────────────────────────────
  • Respect food preference — if user is veg, flag any food spot serving non-veg only
  • If weather has avoid_outdoor dates, note it in descriptions for outdoor spots
  • packing_tips : destination-specific only — no generic "carry water" advice
"""


def run_places_agent(state: dict) -> dict:
    from rich_ui import update_status

    update_status("places", "running")
    print("\n" + "=" * 55)
    print("   TRIP PLANNER v3 — Agent 6: Places Agent")
    print("=" * 55)

    prefs   = state.get("trip_preferences", {})
    weather = state.get("weather_data", {})

    try:
        s      = date.fromisoformat(prefs.get("start_date", ""))
        e      = date.fromisoformat(prefs.get("end_date", ""))
        nights = max((e - s).days, 1)
    except Exception:
        nights = 3

    llm = _get_llm()
    places_llm = llm.with_structured_output(PlacesDiscovery, method="function_calling")

    prompt = f"""
Destination  : {prefs.get('destination')}
Travel type  : {prefs.get('travel_type')}
Duration     : {nights} nights
Travelers    : {prefs.get('travelers')}
Food pref    : {prefs.get('food_pref')}
Weather      : {weather.get('condition', 'unknown')} (score {weather.get('score', '?')}/10)
Avoid outdoor: {', '.join(weather.get('avoid_outdoor', [])) or 'none'}
Special asks : {prefs.get('special_requests', 'none')}

Suggest places, food spots, activities, and hidden gems for this trip.
"""

    result: PlacesDiscovery = None
    for attempt in range(3):
        result = places_llm.invoke([
            SystemMessage(content=PLACES_SYSTEM),
            HumanMessage(content=prompt),
        ])
        if result is not None:
            break
        print(f"  [places] structured output returned None, retrying ({attempt + 1}/3)...")
    if result is None:
        raise RuntimeError("Places agent failed to get a structured response after 3 attempts. Please retry.")

    print(f"\n  Must visit ({len(result.must_visit)} places):")
    for p in result.must_visit:
        print(f"    • {p.name} — {p.description} [{p.duration_hours}h, Rs.{p.entry_fee_inr}]")

    print(f"\n  Food spots ({len(result.food_spots)}):")
    for p in result.food_spots:
        print(f"    • {p.name} — {p.description}")

    print(f"\n  Activities ({len(result.activities)}):")
    for p in result.activities:
        print(f"    • {p.name} — {p.description}")

    if result.hidden_gems:
        print(f"\n  Hidden gems ({len(result.hidden_gems)}):")
        for p in result.hidden_gems:
            print(f"    • {p.name} — {p.description}")

    print("=" * 55)
    update_status("places", "done", "llm")

    return {
        "places_data": {
            "must_visit":   [p.model_dump() for p in result.must_visit],
            "food_spots":   [p.model_dump() for p in result.food_spots],
            "activities":   [p.model_dump() for p in result.activities],
            "hidden_gems":  [p.model_dump() for p in result.hidden_gems],
            "packing_tips": result.packing_tips,
            "source":       "llm",
        },
        "last_completed_node": "places_agent",
        "api_status": {**state.get("api_status", {}), "places": "llm"},
    }
