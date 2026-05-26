import os
import httpx
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from api_clients.weather_client import get_forecast

_http = httpx.Client(verify=False)
_http_async = httpx.AsyncClient(verify=False)
_llm = None


def _get_llm():
    global _llm
    if _llm is None:
        _llm = ChatOpenAI(
            model="openai/gpt-oss-120b:free",
            temperature=0,
            max_tokens=2048,
            openai_api_key=os.getenv("OPENROUTER_API_KEY"),
            openai_api_base="https://openrouter.ai/api/v1",
            http_client=_http,
            http_async_client=_http_async,
        )
    return _llm


# ── Structured output ──────────────────────────────────────────────────────────

class WeatherAssessment(BaseModel):
    condition:    str        = Field(description="sunny | partly_cloudy | rainy | stormy | extreme_heat | foggy")
    score:        int        = Field(description="Weather suitability 1-10 (10=perfect, 1=dangerous)")
    summary:      str        = Field(description="2-sentence weather summary for the traveller")
    concerns:     list[str]  = Field(default_factory=list, description="Specific concerns: heavy rain, heat, cyclone, etc.")
    best_days:    list[str]  = Field(default_factory=list, description="Dates within window that are best for outdoor activities")
    avoid_outdoor: list[str] = Field(default_factory=list, description="Dates to avoid outdoor activities")


class AlternativeDestinations(BaseModel):
    alternatives: list[dict] = Field(
        description="2-3 alternatives. Each: {name, reason, weather_score, vibe}"
    )


WEATHER_SYSTEM = """
You are a weather analyst specialising in Indian travel destinations.
Assess the forecast data and score suitability for tourism.

────────────────────────────────────────────────
SCORING GUIDE
────────────────────────────────────────────────
  10    — Clear skies, ideal for all outdoor activities
  8–9   — Mostly good, minor clouds or brief showers
  6–7   — Noticeable rain or heat, some outdoor plans affected
  4–5   — Significant concerns, plan indoor backups
  1–3   — Dangerous or near-impossible (cyclone, extreme heat, heavy monsoon)

────────────────────────────────────────────────
ASSESSMENT RULES
────────────────────────────────────────────────
  • Be honest — do NOT inflate scores during monsoon or extreme heat
  • Rain > 10 mm on any day = flag it in concerns
  • Hill stations: fog and sub-10°C cold are concerns, not just rain
  • Beach trips: wind > 40 kph = dangerous for water activities
  • Identify specific best and worst dates within the travel window

────────────────────────────────────────────────
OUTPUT RULES
────────────────────────────────────────────────
  • condition     : sunny | partly_cloudy | rainy | stormy | extreme_heat | foggy
  • score         : integer 1–10
  • summary       : 2 plain sentences — what the weather is like and what it means for this traveller
  • concerns      : specific issues only (e.g. "heavy rain 15 mm on June 12", "heat index 44°C")
  • best_days     : dates ideal for outdoor sightseeing
  • avoid_outdoor : dates to keep activities indoors
"""

ALTERNATIVE_SYSTEM = """
You are an Indian travel expert suggesting better-weather alternatives when the original destination is unfavourable.

────────────────────────────────────────────────
RULES FOR ALTERNATIVES
────────────────────────────────────────────────
  • Same travel_type — beach → beach, hill → hill, city → city
  • Similar travel distance from the source city (within ±30%)
  • Genuinely better weather during the exact same travel dates
  • All suggestions must be real, accessible Indian destinations

────────────────────────────────────────────────
OUTPUT RULES (per alternative)
────────────────────────────────────────────────
  • name          : destination name
  • reason        : one sentence on why weather is better there during these dates
  • weather_score : estimated score 1–10 for the same travel window
  • vibe          : one sentence on what kind of experience it offers
"""


# ── Gate 0: Weather advisory HITL ─────────────────────────────────────────────

def _gate0_weather_hitl(state: dict, score: int, summary: str, concerns: list) -> dict:
    """
    Interactive Gate 0: shown when weather score < 5.
    Returns updated trip_preferences if user picks an alternative.
    """
    from rich_ui import pause_display, resume_display
    pause_display()

    prefs = state.get("trip_preferences", {})
    print("\n" + "=" * 55)
    print("  ⚠  HITL GATE 0 — Weather Advisory")
    print("=" * 55)
    print(f"\n  Destination : {prefs.get('destination')}")
    print(f"  Weather     : {summary}")
    print(f"  Score       : {score}/10")
    if concerns:
        print(f"  Concerns    : {', '.join(concerns)}")

    print("\n  Options:")
    print("  [1] Keep original destination and proceed")
    print("  [2] See alternative destinations with better weather")

    choice = input("\n  Your choice (1/2): ").strip()

    if choice != "2":
        print("\n  Proceeding with original destination.\n")
        return {
            "weather_advisory": {
                "triggered":          True,
                "alternatives":       [],
                "user_choice":        "keep_original",
                "destination_changed": False,
            }
        }

    # Generate alternatives
    llm = _get_llm()
    alt_llm = llm.with_structured_output(AlternativeDestinations, method="function_calling")
    prompt = f"""
Source city     : {prefs.get('source')}
Original dest   : {prefs.get('destination')}
Travel type     : {prefs.get('travel_type')}
Travel dates    : {prefs.get('start_date')} to {prefs.get('end_date')}
Weather problem : {summary}

Suggest 2-3 alternative destinations.
"""
    result: AlternativeDestinations = alt_llm.invoke([
        SystemMessage(content=ALTERNATIVE_SYSTEM),
        HumanMessage(content=prompt),
    ])

    alts = result.alternatives
    print("\n  🔀 Alternative destinations with better weather:\n")
    for i, alt in enumerate(alts, 1):
        print(f"  [{i}] {alt.get('name')} — Score {alt.get('weather_score')}/10")
        print(f"      {alt.get('vibe')}")
        print(f"      Why: {alt.get('reason')}\n")
    print(f"  [{len(alts)+1}] Keep original — {prefs.get('destination')}")

    pick = input(f"  Your choice (1-{len(alts)+1}): ").strip()
    try:
        idx = int(pick) - 1
    except ValueError:
        idx = len(alts)

    if idx < 0 or idx >= len(alts):
        print(f"\n  Keeping original destination: {prefs.get('destination')}\n")
        return {
            "weather_advisory": {
                "triggered":          True,
                "alternatives":       alts,
                "user_choice":        "keep_original",
                "destination_changed": False,
            }
        }

    new_dest = alts[idx].get("name")
    print(f"\n  ✓ Switching destination to: {new_dest}\n")

    updated_prefs = {**prefs, "destination": new_dest}
    return {
        "trip_preferences": updated_prefs,
        "weather_advisory": {
            "triggered":          True,
            "alternatives":       alts,
            "user_choice":        new_dest,
            "destination_changed": True,
        }
    }


# ── Entry point ────────────────────────────────────────────────────────────────

def run_weather_agent(state: dict) -> dict:
    """
    Fetches OWM forecast, LLM assesses it, triggers Gate 0 if score < 5.
    If user picks an alternative destination, re-geocodes and reassesses.
    """
    from rich_ui import update_status
    from api_clients.geocoding_client import get_coordinates

    update_status("weather", "running")
    print("\n" + "=" * 55)
    print("   TRIP PLANNER v3 — Agent 3: Weather Agent")
    print("=" * 55)

    prefs  = state.get("trip_preferences", {})
    coords = state.get("coordinates", {})
    llm    = _get_llm()
    assess_llm = llm.with_structured_output(WeatherAssessment, method="function_calling")

    def _assess(dest_lat, dest_lon, destination, start_date, end_date) -> WeatherAssessment:
        forecast = get_forecast(dest_lat, dest_lon, start_date, end_date)
        source_tag = "OWM live"
        if forecast:
            source_tag = forecast.get("source", "OWM live")
            sample = forecast["entries"][:10]
            forecast_text = "\n".join(
                f"  {e['dt_txt']}: {e['weather']}, {e['temp']}°C, rain={e['rain_mm']}mm, wind={e['wind_kph']}kph"
                for e in sample
            )
        else:
            source_tag = "llm_fallback"
            forecast_text = f"No live data. Use seasonal knowledge for {destination} in {start_date[:7]}."

        prompt = f"""
Destination : {destination}
Dates       : {start_date} to {end_date}
Forecast data:
{forecast_text}

Assess weather suitability for a {prefs.get('travel_type','general')} trip.
"""
        result = None
        for attempt in range(3):
            result = assess_llm.invoke([
                SystemMessage(content=WEATHER_SYSTEM),
                HumanMessage(content=prompt),
            ])
            if result is not None:
                break
            print(f"  [weather] structured output returned None, retrying ({attempt + 1}/3)...")

        if result is None:
            result = WeatherAssessment(
                condition="partly_cloudy",
                score=5,
                summary=f"Weather data unavailable for {destination}. Plan for mixed conditions.",
                concerns=["Weather assessment could not be completed — verify locally before travel"],
            )
            source_tag = "llm_fallback"

        return result, source_tag

    assessment, src = _assess(
        coords.get("dest_lat"), coords.get("dest_lon"),
        prefs.get("destination"), prefs.get("start_date"), prefs.get("end_date")
    )

    print(f"\n  Destination : {prefs.get('destination')}")
    print(f"  Condition   : {assessment.condition} | Score: {assessment.score}/10")
    print(f"  Summary     : {assessment.summary}")
    if assessment.concerns:
        print(f"  Concerns    : {', '.join(assessment.concerns)}")
    print("=" * 55)

    update_status("weather", "done", src)

    weather_data = {
        "condition":     assessment.condition,
        "score":         assessment.score,
        "summary":       assessment.summary,
        "concerns":      assessment.concerns,
        "best_days":     assessment.best_days,
        "avoid_outdoor": assessment.avoid_outdoor,
        "final_destination": prefs.get("destination"),
        "source":        src,
    }

    result = {
        "weather_data":        weather_data,
        "last_completed_node": "weather_agent",
        "api_status":          {**state.get("api_status", {}), "weather": src},
    }

    # Gate 0: bad weather advisory
    if assessment.score < 5:
        gate0_result = _gate0_weather_hitl(state, assessment.score, assessment.summary, assessment.concerns)
        from rich_ui import resume_display
        resume_display()
        result.update(gate0_result)

        # If destination changed, re-geocode + re-assess
        if gate0_result.get("weather_advisory", {}).get("destination_changed"):
            new_dest = gate0_result["trip_preferences"]["destination"]
            new_coords = get_coordinates(new_dest)
            if new_coords:
                new_coord_state = {
                    **coords,
                    "dest_lat":  new_coords["lat"],
                    "dest_lon":  new_coords["lon"],
                    "dest_name": new_coords["display_name"],
                }
                result["coordinates"] = new_coord_state
                new_assess, new_src = _assess(
                    new_coords["lat"], new_coords["lon"],
                    new_dest,
                    state.get("trip_preferences", {}).get("start_date"),
                    state.get("trip_preferences", {}).get("end_date"),
                )
                if new_assess is None:
                    new_assess = WeatherAssessment(
                        condition="partly_cloudy",
                        score=5,
                        summary=f"Weather data unavailable for {new_dest}. Plan for mixed conditions.",
                        concerns=["Weather assessment could not be completed — verify locally before travel"],
                    )
                result["weather_data"] = {
                    "condition":     new_assess.condition,
                    "score":         new_assess.score,
                    "summary":       new_assess.summary,
                    "concerns":      new_assess.concerns,
                    "best_days":     new_assess.best_days,
                    "avoid_outdoor": new_assess.avoid_outdoor,
                    "final_destination": new_dest,
                    "source":        new_src,
                }
                print(f"\n  ✓ New weather for {new_dest}: {new_assess.condition} (Score {new_assess.score}/10)")
    else:
        result["weather_advisory"] = {
            "triggered": False, "alternatives": [],
            "user_choice": "not_triggered", "destination_changed": False,
        }

    return result
