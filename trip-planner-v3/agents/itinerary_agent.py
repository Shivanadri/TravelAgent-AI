import os
import httpx
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
            temperature=0.4,
            max_tokens=4096,
            openai_api_key=os.getenv("OPENROUTER_API_KEY"),
            openai_api_base="https://openrouter.ai/api/v1",
            http_client=_http,
            http_async_client=_http_async,
        )
    return _llm


class DayActivity(BaseModel):
    time:                str   = Field(description="HH:MM")
    activity:            str   = Field(description="Activity description")
    location:            str   = Field(description="Place name")
    estimated_cost_inr:  int   = Field(default=0)
    duration_hours:      float = Field(default=1.0)
    notes:               str   = Field(default="")


class ItineraryDay(BaseModel):
    day:                 int
    date:                str
    day_type:            str
    theme:               str        = Field(description="Theme for the day, e.g. 'Beach & Sunsets'")
    activities:          list[DayActivity]
    meals:               dict       = Field(description="{breakfast, lunch, dinner} restaurant names")
    estimated_total_inr: int
    tips:                str        = Field(default="")


class FullItinerary(BaseModel):
    title:               str
    overview:            str        = Field(description="2-3 sentence trip overview")
    days:                list[ItineraryDay]
    packing_list:        list[str]  = Field(description="10-15 packing items")
    emergency_contacts:  dict       = Field(description="{police, hospital, tourist_helpline} for destination")
    travel_tips:         list[str]  = Field(description="5 practical travel tips")


ITINERARY_SYSTEM = """
You are a professional Indian travel itinerary planner who creates practical, enjoyable day-by-day plans.

────────────────────────────────────────────────
DAY STRUCTURE RULES
────────────────────────────────────────────────
  Arrival day   : light schedule — check-in, local explore, nearby dinner only
  Departure day : one morning activity + pack + depart — no long trips
  All other days: 3–5 activities, spaced with realistic travel time between each

────────────────────────────────────────────────
PLANNING RULES
────────────────────────────────────────────────
  • Space must_visit places across different days — never cluster all on one day
  • Respect avoid_outdoor dates from weather — schedule indoor activities on those days
  • Prioritise user_added places — include them even if other spots must be swapped out
  • Keep each day's estimated_total_inr within ±5% of its target_inr

────────────────────────────────────────────────
ACTIVITY TIMING RULES
────────────────────────────────────────────────
  • Use 24-hour HH:MM format (09:00, 14:30)
  • Allow 30–60 min travel time between locations
  • Meals: breakfast 30–45 min, lunch/dinner 45–90 min
  • Start no earlier than 08:00, schedule nothing past 21:30

────────────────────────────────────────────────
MEALS & FOOD
────────────────────────────────────────────────
  • Use actual names from food_spots wherever possible
  • Respect food preference (veg/non-veg) at every meal — no exceptions

────────────────────────────────────────────────
OUTPUT RULES
────────────────────────────────────────────────
  • theme              : 3–5 words capturing the day's highlight, e.g. "Backwaters & Sunset Cruise"
  • travel_tips        : practical tips a first-time visitor genuinely needs (not generic advice)
  • emergency_contacts : real numbers — police 100, ambulance 108 are always valid
  • packing_list       : destination + season specific, not a generic travel checklist
"""


def run_itinerary_agent(state: dict) -> dict:
    from rich_ui import update_status

    update_status("itinerary", "running")
    print("\n" + "=" * 55)
    print("   TRIP PLANNER v3 — Agent 8: Itinerary Agent")
    print("=" * 55)

    prefs          = state.get("trip_preferences", {})
    weather        = state.get("weather_data", {})
    transport      = state.get("transport_data", {})
    hotel          = state.get("hotel_data", {})
    places         = state.get("places_confirmed", state.get("places_data", {}))
    budget_summary = state.get("budget_summary", {})
    daily_targets  = state.get("daily_budget_targets", [])

    llm = _get_llm()
    itinerary_llm = llm.with_structured_output(FullItinerary, method="function_calling")

    daily_str = "\n".join(
        f"  Day {d['day']} ({d['date']}) [{d['day_type']}]: budget Rs.{d['target_inr']:,} (±5%)"
        for d in daily_targets
    ) if daily_targets else "  See trip dates"

    must_visit_str = ", ".join(p.get("name", "") for p in places.get("must_visit", []))
    food_spots_str = ", ".join(p.get("name", "") for p in places.get("food_spots", []))
    activities_str = ", ".join(p.get("name", "") for p in places.get("activities", []))
    hidden_str     = ", ".join(p.get("name", "") for p in places.get("hidden_gems", []))
    user_added_str = ", ".join(p.get("name", "") for p in places.get("user_added", []))

    hotel_name = hotel.get("recommended", {}).get("name", "TBD")

    prompt = f"""
Destination   : {prefs.get('destination')}
Dates         : {prefs.get('start_date')} to {prefs.get('end_date')}
Travelers     : {prefs.get('travelers')} | Hotel: {hotel_name}
Transport     : {transport.get('final_mode', '?')} ({transport.get('duration_hours', '?')}h)
Travel type   : {prefs.get('travel_type')}
Food pref     : {prefs.get('food_pref')}
Special asks  : {prefs.get('special_requests', 'none')}

Weather       : {weather.get('condition')} | Score: {weather.get('score')}/10
Avoid outdoor : {', '.join(weather.get('avoid_outdoor', [])) or 'none'}
Best days     : {', '.join(weather.get('best_days', [])) or 'all'}

Places to visit:
  Must see    : {must_visit_str}
  Food spots  : {food_spots_str}
  Activities  : {activities_str}
  Hidden gems : {hidden_str}
  User added  : {user_added_str or 'none'}

Daily budget targets:
{daily_str}

Total estimate: Rs.{budget_summary.get('total_estimate', 0):,}

Create a complete day-by-day itinerary.
"""

    print(f"\n  Building itinerary for {prefs.get('destination')}...")
    result: FullItinerary = None
    for attempt in range(3):
        result = itinerary_llm.invoke([
            SystemMessage(content=ITINERARY_SYSTEM),
            HumanMessage(content=prompt),
        ])
        if result is not None:
            break
        print(f"  [itinerary] structured output returned None, retrying ({attempt + 1}/3)...")
    if result is None:
        raise RuntimeError("Itinerary agent failed to get a structured response after 3 attempts. Please retry.")

    print(f"\n  ✓ {result.title}")
    print(f"  {len(result.days)} days planned")
    for day in result.days:
        print(f"    Day {day.day} — {day.theme} | Rs.{day.estimated_total_inr:,}")
    print("=" * 55)

    update_status("itinerary", "done", "llm")

    return {
        "itinerary": {
            "title":              result.title,
            "overview":           result.overview,
            "days":               [d.model_dump() for d in result.days],
            "packing_list":       result.packing_list,
            "emergency_contacts": result.emergency_contacts,
            "travel_tips":        result.travel_tips,
        },
        "last_completed_node": "itinerary_agent",
    }
