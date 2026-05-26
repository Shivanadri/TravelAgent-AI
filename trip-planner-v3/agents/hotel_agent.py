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
            temperature=0.3,
            max_tokens=2048,
            openai_api_key=os.getenv("OPENROUTER_API_KEY"),
            openai_api_base="https://openrouter.ai/api/v1",
            http_client=_http,
            http_async_client=_http_async,
        )
    return _llm


class HotelOption(BaseModel):
    name:                str        = Field(description="Hotel name")
    area:                str        = Field(description="Neighbourhood or locality")
    category:            str        = Field(description="budget | mid | luxury")
    price_per_night_inr: int        = Field(description="Estimated price per room per night in INR")
    highlights:          list[str]  = Field(description="2-3 key features")
    why_good:            str        = Field(description="One-sentence reason this suits the trip")


class HotelOptions(BaseModel):
    options:           list[HotelOption] = Field(description="3 hotel options")
    recommended_index: int               = Field(description="0-based index of recommended option")


HOTEL_SYSTEM = """
You are an Indian hospitality expert helping travellers find the right accommodation.

────────────────────────────────────────────────
CATEGORY PRICE GUIDE (per room per night, 2024–25)
────────────────────────────────────────────────
  budget  : Rs.   800 – 2,500  (hostels, guesthouses, basic hotels)
  mid     : Rs. 2,500 – 6,000  (3-star, well-reviewed B&Bs)
  luxury  : Rs. 6,000 +        (4–5 star, heritage properties, resorts)

────────────────────────────────────────────────
HOW TO SUGGEST OPTIONS
────────────────────────────────────────────────
  • Suggest exactly 3 options: one tier below preference, exact match, one tier above
  • Name real, well-known hotels or property types specific to the destination
  • area : use specific neighbourhood (e.g. "Colaba, Mumbai" not just "Mumbai")
  • highlights: tailor to travel_type:
      beach      → sea view, proximity to beach, pool
      hill       → mountain view, fireplace, nature trails nearby
      city       → central location, transport links
      adventure  → proximity to activity hubs, storage/drying facilities

────────────────────────────────────────────────
OUTPUT RULES
────────────────────────────────────────────────
  • why_good         : one sentence explaining the fit with this specific trip
  • recommended_index: best overall match for the traveller's preference and budget
"""


def _calc_nights(start_date: str, end_date: str) -> int:
    try:
        s = date.fromisoformat(start_date)
        e = date.fromisoformat(end_date)
        return max((e - s).days, 1)
    except Exception:
        return 3


def run_hotel_agent(state: dict) -> dict:
    from rich_ui import update_status

    update_status("hotel", "running")
    print("\n" + "=" * 55)
    print("   TRIP PLANNER v3 — Agent 5: Hotel Agent")
    print("=" * 55)

    prefs    = state.get("trip_preferences", {})
    nights   = _calc_nights(prefs.get("start_date", ""), prefs.get("end_date", ""))
    travelers = max(prefs.get("travelers", 1), 1)
    budget_pp = prefs.get("budget", 20000) // travelers
    hotel_budget_est = int(budget_pp * 0.35)

    llm = _get_llm()
    hotel_llm = llm.with_structured_output(HotelOptions, method="function_calling")

    prompt = f"""
Destination   : {prefs.get('destination')}
Check-in      : {prefs.get('start_date')}
Check-out     : {prefs.get('end_date')} ({nights} nights)
Travelers     : {travelers}
Hotel pref    : {prefs.get('hotel_pref')}
Travel type   : {prefs.get('travel_type')}
Budget/person : Rs.{budget_pp:,} total
Hotel budget  : ~Rs.{hotel_budget_est:,}/person total (~Rs.{hotel_budget_est // max(nights, 1):,}/night)

Suggest 3 hotels matching this trip.
"""

    result: HotelOptions = None
    for attempt in range(3):
        result = hotel_llm.invoke([
            SystemMessage(content=HOTEL_SYSTEM),
            HumanMessage(content=prompt),
        ])
        if result is not None:
            break
        print(f"  [hotel] structured output returned None, retrying ({attempt + 1}/3)...")
    if result is None:
        raise RuntimeError("Hotel agent failed to get a structured response after 3 attempts. Please retry.")

    options = result.options

    print(f"\n  Destination: {prefs.get('destination')} | {nights} nights\n")
    for i, h in enumerate(options, 1):
        marker = "★" if i - 1 == result.recommended_index else " "
        total  = h.price_per_night_inr * nights
        print(f"  [{i}]{marker} {h.name} ({h.area})")
        print(f"      {h.category.upper()} | Rs.{h.price_per_night_inr:,}/night | Total: Rs.{total:,}")
        for hl in h.highlights:
            print(f"      • {hl}")
        print(f"      ► {h.why_good}\n")

    raw = input(f"  Your choice (1-{len(options)}) or Enter for recommended: ").strip()
    try:
        idx = int(raw) - 1
        if idx < 0 or idx >= len(options):
            raise ValueError
    except ValueError:
        idx = result.recommended_index

    chosen = options[idx]
    rooms  = max(travelers // 2, 1)
    total_stay_cost = chosen.price_per_night_inr * nights * rooms
    print(f"\n  ✓ Hotel: {chosen.name} — Rs.{chosen.price_per_night_inr:,}/night\n")

    update_status("hotel", "done", "llm")

    return {
        "hotel_data": {
            "recommended":      chosen.model_dump(),
            "top_3":            [o.model_dump() for o in options],
            "price_per_night":  chosen.price_per_night_inr,
            "nights":           nights,
            "total_stay_cost":  total_stay_cost,
            "source":           "llm",
        },
        "last_completed_node": "hotel_agent",
        "api_status": {**state.get("api_status", {}), "hotels": "llm"},
    }
