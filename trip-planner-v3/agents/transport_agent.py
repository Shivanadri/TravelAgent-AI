import os
import httpx
from datetime import date
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI


def _safe_input(prompt: str, default: str = "") -> str:
    try:
        return input(prompt).strip()
    except EOFError:
        return default

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


class TransportOption(BaseModel):
    mode:                 str   = Field(description="flight | train | bus | cab")
    description:          str   = Field(description="Service name and route details")
    cost_per_person_inr:  int   = Field(description="Estimated cost per person in INR")
    duration_hours:       float = Field(description="Travel time in hours")
    comfort:              str   = Field(description="budget | standard | premium")
    booking_tip:          str   = Field(description="Best way to book this option")


class TransportOptions(BaseModel):
    options:           list[TransportOption] = Field(description="2-3 transport options")
    recommended_index: int                   = Field(description="0-based index of recommended option")


TRANSPORT_SYSTEM = """
You are an Indian travel logistics expert helping plan the best route between two cities.

────────────────────────────────────────────────
HOW TO SUGGEST OPTIONS
────────────────────────────────────────────────
  • Suggest 2–3 distinct modes — not 3 variations of the same mode
  • Be specific — name real services:
      Trains   : Rajdhani Express, Shatabdi, Vande Bharat, Duronto
      Buses    : KSRTC, VRL Travels, MSRTC, RedBus operators
      Flights  : IndiGo, Air India, SpiceJet, Vistara
  • Distance > 800 km  → always include a flight option
  • Hill stations < 300 km → buses or cabs are often better than trains

────────────────────────────────────────────────
COST BENCHMARKS (2024–25)
────────────────────────────────────────────────
  Flights  : Rs. 2,500 –  8,000 one-way (economy domestic)
  Trains   : Rs.   400 –  2,500 per person (sleeper to AC 2-tier)
  Buses    : Rs.   300 –  1,200 per person (state to luxury AC)
  Cabs     : Rs.    12 –     18 per km

────────────────────────────────────────────────
OUTPUT RULES
────────────────────────────────────────────────
  • cost_per_person_inr : one-way fare per person
  • duration_hours      : realistic door-to-door time (include station/airport travel)
  • booking_tip         : where to book and how far in advance
  • recommended_index   : best balance of cost, comfort, and travel type for this trip
"""


def _calc_nights(start_date: str, end_date: str) -> int:
    try:
        s = date.fromisoformat(start_date)
        e = date.fromisoformat(end_date)
        return max((e - s).days, 1)
    except Exception:
        return 3


def run_transport_agent(state: dict) -> dict:
    from rich_ui import update_status

    update_status("transport", "running")
    print("\n" + "=" * 55)
    print("   TRIP PLANNER v3 — Agent 4: Transport Agent")
    print("=" * 55)

    prefs    = state.get("trip_preferences", {})
    nights   = _calc_nights(prefs.get("start_date", ""), prefs.get("end_date", ""))
    travelers = max(prefs.get("travelers", 1), 1)
    budget_pp = prefs.get("budget", 20000) // travelers

    llm = _get_llm()
    transport_llm = llm.with_structured_output(TransportOptions, method="function_calling")

    prompt = f"""
Source        : {prefs.get('source')}
Destination   : {prefs.get('destination')}
Travel dates  : {prefs.get('start_date')} to {prefs.get('end_date')} ({nights} nights)
Travelers     : {travelers}
Budget/person : Rs.{budget_pp:,}
Preference    : {prefs.get('transport_pref', 'any')}
Travel type   : {prefs.get('travel_type')}

Suggest 2-3 transport options for this trip.
"""

    result: TransportOptions = None
    for attempt in range(3):
        result = transport_llm.invoke([
            SystemMessage(content=TRANSPORT_SYSTEM),
            HumanMessage(content=prompt),
        ])
        if result is not None:
            break
        print(f"  [transport] structured output returned None, retrying ({attempt + 1}/3)...")
    if result is None:
        raise RuntimeError("Transport agent failed to get a structured response after 3 attempts. Please retry.")

    options = result.options

    print(f"\n  Route: {prefs.get('source')} → {prefs.get('destination')}\n")
    for i, opt in enumerate(options, 1):
        marker = "★" if i - 1 == result.recommended_index else " "
        print(f"  [{i}]{marker} {opt.mode.upper()} — {opt.description}")
        print(f"      Cost: Rs.{opt.cost_per_person_inr:,}/person | Time: {opt.duration_hours}h | {opt.comfort}")
        print(f"      Tip : {opt.booking_tip}\n")
    print("  [★ = Recommended]")

    raw = _safe_input(f"\n  Your choice (1-{len(options)}) or Enter for recommended: ")
    try:
        idx = int(raw) - 1
        if idx < 0 or idx >= len(options):
            raise ValueError
    except ValueError:
        idx = result.recommended_index

    chosen = options[idx]
    print(f"\n  ✓ Transport: {chosen.mode.upper()} — {chosen.description}\n")

    update_status("transport", "done", "llm")

    return {
        "transport_data": {
            "final_mode":              chosen.mode,
            "description":             chosen.description,
            "estimated_cost_per_person": chosen.cost_per_person_inr,
            "total_transport_cost":    chosen.cost_per_person_inr * travelers,
            "duration_hours":          chosen.duration_hours,
            "comfort":                 chosen.comfort,
            "booking_tip":             chosen.booking_tip,
            "all_options":             [o.model_dump() for o in options],
            "source":                  "llm",
        },
        "last_completed_node": "transport_agent",
        "api_status": {**state.get("api_status", {}), "transport": "llm"},
    }
