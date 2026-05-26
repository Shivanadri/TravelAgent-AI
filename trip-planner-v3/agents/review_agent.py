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
            temperature=0,
            max_tokens=2048,
            openai_api_key=os.getenv("OPENROUTER_API_KEY"),
            openai_api_base="https://openrouter.ai/api/v1",
            http_client=_http,
            http_async_client=_http_async,
        )
    return _llm


class ItineraryReview(BaseModel):
    passed:              bool       = Field(description="True if itinerary is acceptable")
    score:               int        = Field(description="Overall quality score 1-10")
    issues:              list[str]  = Field(default_factory=list, description="Critical problems that need fixing")
    warnings:            list[str]  = Field(default_factory=list, description="Minor concerns")
    positive_highlights: list[str]  = Field(description="What the itinerary does well")
    overall_verdict:     str        = Field(description="1-2 sentence verdict")
    retry_agents:        list[str]  = Field(default_factory=list, description="Agents to re-run: itinerary|budget|places")


REVIEW_SYSTEM = """
You are a senior travel consultant doing a quality review of an AI-generated Indian travel itinerary.

────────────────────────────────────────────────
WHAT TO ASSESS
────────────────────────────────────────────────
  PACING       : Is daily activity count reasonable? (3–5 activities/day = good)
                 Flag if any day has > 6 activities or < 2 activities
  VARIETY      : Do days have different themes and locations?
                 Flag if 3+ consecutive days look identical
  PREFERENCES  : Does the plan honour food pref, hotel pref, transport pref, and travel type?
  BUDGET       : Are daily costs within ±10% of targets? Is total estimate within budget?
  SPECIAL ASKS : Are all user-requested places included?
  WEATHER      : Are outdoor activities on avoid_outdoor dates?

────────────────────────────────────────────────
SCORING GUIDE
────────────────────────────────────────────────
  9–10 : Excellent — ready to share with the traveller as-is
  7–8  : Good — minor improvements possible but plan is solid
  6    : Acceptable — some concerns but trip is doable
  4–5  : Needs revision — one or more significant issues
  1–3  : Requires major rework — core preferences ignored or plan is impractical

────────────────────────────────────────────────
PASS / FAIL RULES
────────────────────────────────────────────────
  PASS (score ≥ 6) : itinerary is usable, even if not perfect
  FAIL (score < 6) : only if — significantly over budget, core food/hotel pref ignored,
                     critically rushed (> 7 activities/day), or must-visit places missing

────────────────────────────────────────────────
OUTPUT RULES
────────────────────────────────────────────────
  • issues          : critical problems only — not style preferences
  • warnings        : minor improvements that would make the plan better
  • retry_agents    : name specific agent(s) to re-run — itinerary | budget | places
  • overall_verdict : 1–2 sentences summarising plan quality honestly
"""


def _score_pacing(days: list) -> int:
    if not days:
        return 5
    counts = [len(d.get("activities", [])) for d in days]
    avg = sum(counts) / len(counts)
    if 3 <= avg <= 5:
        return 9
    if 2 <= avg <= 6:
        return 7
    return 5


def _score_variety(days: list) -> int:
    if not days:
        return 5
    themes    = set(d.get("theme", "") for d in days)
    locations = set(
        a.get("location", "")
        for d in days
        for a in d.get("activities", [])
    )
    all_acts  = [a for d in days for a in d.get("activities", [])]
    if not all_acts:
        return 5
    ratio = len(locations) / len(all_acts)
    if ratio >= 0.7 and len(themes) >= max(len(days) - 1, 1):
        return 9
    if ratio >= 0.5:
        return 7
    return 5


def run_review_agent(state: dict) -> dict:
    from rich_ui import update_status

    update_status("review", "running")
    print("\n" + "=" * 55)
    print("   TRIP PLANNER v3 — Agent 9: Review Agent")
    print("=" * 55)

    prefs          = state.get("trip_preferences", {})
    itinerary      = state.get("itinerary", {})
    budget_summary = state.get("budget_summary", {})
    weather        = state.get("weather_data", {})

    llm = _get_llm()
    review_llm = llm.with_structured_output(ItineraryReview, method="function_calling")

    days_summary = "\n".join(
        f"  Day {d.get('day')} [{d.get('day_type')}] — {d.get('theme')} | Rs.{d.get('estimated_total_inr', 0):,}"
        for d in itinerary.get("days", [])
    )

    prompt = f"""
Trip: {prefs.get('source')} → {prefs.get('destination')}
Dates: {prefs.get('start_date')} to {prefs.get('end_date')}
Travelers: {prefs.get('travelers')} | Type: {prefs.get('travel_type')}
Hotel pref: {prefs.get('hotel_pref')} | Food: {prefs.get('food_pref')}
Budget: Rs.{prefs.get('budget', 0):,} | Estimate: Rs.{budget_summary.get('total_estimate', 0):,}
Within budget: {budget_summary.get('within_budget', 'unknown')}
Special requests: {prefs.get('special_requests', 'none')}

Itinerary: {itinerary.get('title')}
{days_summary}

Weather: {weather.get('condition')} (score {weather.get('score')}/10)

Review this itinerary for quality, pacing, and preference alignment.
"""

    result: ItineraryReview = None
    for attempt in range(3):
        result = review_llm.invoke([
            SystemMessage(content=REVIEW_SYSTEM),
            HumanMessage(content=prompt),
        ])
        if result is not None:
            break
        print(f"  [review] structured output returned None, retrying ({attempt + 1}/3)...")
    if result is None:
        raise RuntimeError("Review agent failed to get a structured response after 3 attempts. Please retry.")

    icon = "✓" if result.passed else "✗"
    print(f"\n  {icon} Review Score: {result.score}/10 — {'PASSED' if result.passed else 'FAILED'}")
    print(f"  {result.overall_verdict}")

    if result.issues:
        print(f"\n  Issues:")
        for issue in result.issues:
            print(f"    ✗ {issue}")

    if result.warnings:
        print(f"\n  Warnings:")
        for w in result.warnings:
            print(f"    ⚠ {w}")

    if result.positive_highlights:
        print(f"\n  Highlights:")
        for h in result.positive_highlights:
            print(f"    ✓ {h}")

    print("=" * 55)

    days = itinerary.get("days", [])
    pacing_score  = _score_pacing(days)
    variety_score = _score_variety(days)

    update_status("review", "done", f"score={result.score}")

    return {
        "review_status": {
            "passed":              result.passed,
            "score":               result.score,
            "issues":              result.issues,
            "warnings":            result.warnings,
            "positive_highlights": result.positive_highlights,
            "overall_verdict":     result.overall_verdict,
            "retry_agents":        result.retry_agents,
        },
        "eval_scores": {
            "pacing":   pacing_score,
            "variety":  variety_score,
            "overall":  result.score,
        },
        "last_completed_node": "review_agent",
    }
