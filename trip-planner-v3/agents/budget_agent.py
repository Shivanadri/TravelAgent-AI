import os
import httpx
from datetime import date, timedelta
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

_http = httpx.Client(verify=False)
_http_async = httpx.AsyncClient(verify=False)
_llm = None

DAY_WEIGHTS = {
    "arrival":           0.85,
    "departure":         0.80,
    "heavy_sightseeing": 1.10,
    "standard":          1.00,
    "relaxed":           0.90,
    "adventure":         1.15,
}


def _get_llm():
    global _llm
    if _llm is None:
        _llm = ChatOpenAI(
            model="openai/gpt-oss-120b:free",
            temperature=0,
            max_tokens=4096,
            openai_api_key=os.getenv("OPENROUTER_API_KEY"),
            openai_api_base="https://openrouter.ai/api/v1",
            http_client=_http,
            http_async_client=_http_async,
        )
    return _llm


class BudgetBreakdown(BaseModel):
    food_per_day:        int        = Field(description="Average food cost per day per person in INR")
    activities_per_day:  int        = Field(description="Average activities/entry cost per day per person in INR")
    misc_per_day:        int        = Field(description="Miscellaneous/local transport per day per person in INR")
    emergency_buffer_pct: int       = Field(default=10, description="Emergency buffer percentage")
    savings_tips:        list[str]  = Field(description="3-5 budget saving tips for this trip")


BUDGET_SYSTEM = """
You are a travel budget expert for India with deep knowledge of realistic trip costs.

────────────────────────────────────────────────
DAILY COST BENCHMARKS (per person, 2024–25)
────────────────────────────────────────────────
  FOOD:
    budget  → Rs.   300 –   600 /day  (street food, local dhabas)
    mid     → Rs.   600 – 1,200 /day  (casual restaurants, cafés)
    luxury  → Rs. 1,200 +       /day  (restaurants, hotel dining)

  ACTIVITIES (entry fees + local experiences):
    budget  → Rs.   100 –   300 /day
    mid     → Rs.   300 –   700 /day
    luxury  → Rs.   700 – 2,000 /day

  MISC (auto/cab, tips, incidentals):
    All tiers → Rs. 200 – 500 /day

────────────────────────────────────────────────
ESTIMATION RULES
────────────────────────────────────────────────
  • Estimate for this destination specifically — Goa costs differ from Leh or Spiti
  • Consider travel_type — beach relaxation days are cheaper than adventure sport days
  • The entry fees total is already provided — factor it in, do not double-count
  • emergency_buffer_pct : 10% for budget, 8% for mid, 5% for luxury
  • savings_tips : destination-specific only — not generic travel advice

────────────────────────────────────────────────
OUTPUT RULES
────────────────────────────────────────────────
  • All amounts in INR, whole numbers only
  • Be realistic — do not over-estimate to appear safe
"""


def _classify_day(day_idx: int, total_days: int) -> str:
    if day_idx == 0:
        return "arrival"
    if day_idx == total_days - 1:
        return "departure"
    if day_idx == 1:
        return "heavy_sightseeing"
    if day_idx == total_days - 2 and total_days > 3:
        return "relaxed"
    return "standard"


def run_budget_agent(state: dict) -> dict:
    from rich_ui import update_status

    update_status("budget", "running")
    print("\n" + "=" * 55)
    print("   TRIP PLANNER v3 — Agent 7: Budget Agent")
    print("=" * 55)

    prefs     = state.get("trip_preferences", {})
    transport = state.get("transport_data", {})
    hotel     = state.get("hotel_data", {})
    places    = state.get("places_confirmed", state.get("places_data", {}))

    total_budget  = prefs.get("budget", 30000)
    travelers     = max(prefs.get("travelers", 1), 1)

    try:
        s          = date.fromisoformat(prefs.get("start_date", ""))
        e          = date.fromisoformat(prefs.get("end_date", ""))
        nights     = max((e - s).days, 1)
        total_days = nights + 1
    except Exception:
        nights, total_days, s = 3, 4, date.today()

    transport_cost = transport.get("total_transport_cost", 0)
    hotel_cost     = hotel.get("total_stay_cost", 0)
    fixed_costs    = transport_cost + hotel_cost
    remaining      = total_budget - fixed_costs

    total_entry_fees = sum(
        p.get("entry_fee_inr", 0)
        for cat in ["must_visit", "activities"]
        for p in places.get(cat, [])
    ) if places else 0

    llm = _get_llm()
    budget_llm = llm.with_structured_output(BudgetBreakdown, method="function_calling")

    prompt = f"""
Destination    : {prefs.get('destination')}
Duration       : {nights} nights, {total_days} days
Travelers      : {travelers}
Hotel category : {prefs.get('hotel_pref')}
Food pref      : {prefs.get('food_pref')}
Travel type    : {prefs.get('travel_type')}

Total budget   : Rs.{total_budget:,}
Transport cost : Rs.{transport_cost:,}
Hotel cost     : Rs.{hotel_cost:,}
Entry fees est : Rs.{total_entry_fees:,} total
Remaining      : Rs.{remaining:,} for {total_days} days

Estimate daily costs per person for food, activities, misc.
"""

    bd: BudgetBreakdown = None
    for attempt in range(3):
        bd = budget_llm.invoke([
            SystemMessage(content=BUDGET_SYSTEM),
            HumanMessage(content=prompt),
        ])
        if bd is not None:
            break
        print(f"  [budget] structured output returned None, retrying ({attempt + 1}/3)...")
    if bd is None:
        hotel_pref = prefs.get("hotel_pref", "mid")
        _defaults = {
            "budget":  (450, 200, 300),
            "mid":     (900, 500, 350),
            "luxury":  (1500, 1200, 500),
        }
        food, acts, misc = _defaults.get(hotel_pref, _defaults["mid"])
        bd = BudgetBreakdown(
            food_per_day=food,
            activities_per_day=acts,
            misc_per_day=misc,
            emergency_buffer_pct=10,
            savings_tips=["Book transport and hotels in advance for better rates."],
        )
        print("  [budget] Using estimated defaults based on hotel preference.")

    daily_variable = (bd.food_per_day + bd.activities_per_day + bd.misc_per_day) * travelers

    daily_targets = []
    for i in range(total_days):
        day_type  = _classify_day(i, total_days)
        weight    = DAY_WEIGHTS.get(day_type, 1.0)
        target    = daily_variable * weight
        variation = target * 0.05
        daily_targets.append({
            "day":        i + 1,
            "date":       (s + timedelta(days=i)).isoformat(),
            "day_type":   day_type,
            "weight":     weight,
            "target_inr": round(target),
            "min_inr":    round(target - variation),
            "max_inr":    round(target + variation),
        })

    total_variable_est = sum(d["target_inr"] for d in daily_targets)
    total_estimate     = fixed_costs + total_variable_est
    within_budget      = total_estimate <= total_budget
    surplus            = total_budget - total_estimate

    print(f"\n  Total Budget    : Rs.{total_budget:,}")
    print(f"  Transport       : Rs.{transport_cost:,}")
    print(f"  Accommodation   : Rs.{hotel_cost:,}")
    print(f"  Variable/day    : Rs.{daily_variable:,} × {total_days} days")
    print(f"  Total estimate  : Rs.{total_estimate:,}")
    status_str = "✓ Within budget" if within_budget else f"⚠ Over by Rs.{abs(surplus):,}"
    print(f"  Status          : {status_str}")
    print(f"\n  Daily distribution ({total_days} days):")
    for d in daily_targets:
        print(f"    Day {d['day']} ({d['date']}) [{d['day_type']}]: Rs.{d['target_inr']:,}")
    print("=" * 55)

    update_status("budget", "done", "computed")

    return {
        "budget_summary": {
            "total_budget":     total_budget,
            "transport_cost":   transport_cost,
            "hotel_cost":       hotel_cost,
            "daily_variable":   daily_variable,
            "total_estimate":   total_estimate,
            "within_budget":    within_budget,
            "surplus_deficit":  surplus,
            "emergency_buffer": round(total_budget * bd.emergency_buffer_pct / 100),
            "breakdown": {
                "transport":  transport_cost,
                "hotel":      hotel_cost,
                "food_daily": bd.food_per_day,
                "activities": bd.activities_per_day,
                "misc":       bd.misc_per_day,
            },
            "savings_tips": bd.savings_tips,
        },
        "daily_budget_targets": daily_targets,
        "last_completed_node":  "budget_agent",
    }
