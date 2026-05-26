import os
import math
import time
import httpx
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_openai import ChatOpenAI
from openai import APIConnectionError as OpenAIConnectionError, APIStatusError as OpenAIStatusError, RateLimitError as OpenAIRateLimitError
from api_clients.geocoding_client import get_coordinates


# ── Structured output model ────────────────────────────────────────────────────

class TripPreferences(BaseModel):
    user_id:        str  = Field(description="Unique ID for this user (name-based slug)")
    user_name:      str  = Field(description="User's first name")
    source:         str  = Field(description="Departure city")
    destination:    str  = Field(description="Destination city or place")
    start_date:     str  = Field(description="Trip start date YYYY-MM-DD")
    end_date:       str  = Field(description="Trip end date YYYY-MM-DD")
    budget:         int  = Field(description="Total budget in INR for all travelers")
    travelers:      int  = Field(description="Number of travelers")
    travel_type:    str  = Field(description="beach | hill | city | adventure | religious | wildlife")
    hotel_pref:     str  = Field(description="budget | mid | luxury")
    food_pref:      str  = Field(default="no preference", description="veg | non-veg | no preference")
    transport_pref: str  = Field(default="any", description="flight | train | bus | any")
    special_requests: str = Field(default="", description="Any special requests or places user wants to visit")


# ── LLM (OpenRouter) — lazy init so load_dotenv() runs first ──────────────────

_http = httpx.Client(verify=False)
_http_async = httpx.AsyncClient(verify=False)

_llm = None
_extract_llm = None


def _get_llms():
    global _llm, _extract_llm
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
        _extract_llm = _llm.with_structured_output(TripPreferences, method="function_calling")
    return _llm, _extract_llm

SYSTEM_PROMPT = """
You are a warm, friendly Indian travel planning assistant named Priya.
Collect trip details through natural flowing conversation — like chatting with a friend.

────────────────────────────────────────────────
CRITICAL RULE — READ FIRST
────────────────────────────────────────────────
ONE QUESTION PER MESSAGE. ALWAYS. NO EXCEPTIONS.
Never ask two or more questions in a single reply.
Ask exactly ONE question, wait for the user's answer, then ask the next.
Violating this rule makes the conversation unusable.

────────────────────────────────────────────────
QUESTION ORDER & HOW TO ASK
────────────────────────────────────────────────
Ask fields in this order, skipping already-collected ones:

  1. travel_type       — "Are you travelling Solo, as a Couple, with Family, or with Friends/Group?"
  2. num_travelers     — Ask based on travel_type context:
                         Solo → confirm it's just them (num=1)
                         Couple → confirm 2, or ask if more
                         Family/Friends → "How many people in total, including yourself?"
  3. source_location   — "Which city or town are you starting from?"
  4. destination       — "Where are you heading? (city, region, or country)"
  5. travel_dates      — "When are you planning to travel?"
                         Accept ANY natural language: "next Monday to Friday",
                         "last week of June", "26th to 30th July", "next weekend" etc.
                         Resolve relative dates using today's date and store as "YYYY-MM-DD to YYYY-MM-DD".
  6. places_to_visit   — "Any specific places, landmarks, or attractions you'd like to cover?"
                         These can be anywhere within the COUNTRY of the destination — not limited to the city.
                         Example: destination = Mumbai → places can include Goa, Pune, Ajanta Caves (all within India).
                         Only flag if a place is in a completely different country.
  7. transportation_preferences — "How would you prefer to travel? (Flight / Train / Bus / Road trip / Mixed)"
  8. food_preferences  — "Any food preferences or dietary needs? (Veg / Non-veg / Vegan / Halal / No preference)"
  9. trip_budget_luxury — "What's your comfort level? (Budget / Standard / Luxury / Ultra-Luxury)"
  10. budget           — "What's your total budget for this trip? (please include currency, e.g. ₹50,000 or $2,000)"

────────────────────────────────────────────────
INLINE VALIDATION (reject and re-ask immediately)
────────────────────────────────────────────────
  ✗ Same-location check — ONLY use the [COORD_CHECK] system message (injected automatically
      after source and destination are collected). Do NOT guess geography yourself.
      Format: [COORD_CHECK: source='...' (lat,lon), dest='...' (lat,lon), distance=X km — <verdict>]
        • verdict = "same location"  → re-ask: "Looks like source and destination are the same — where are you heading?"
        • verdict = "different locations" → accept and continue
      If no [COORD_CHECK] has arrived yet, accept any destination the user gives and wait.
  ✗ num_travelers ≤ 0 → "That doesn't seem right — how many people are travelling?"
  ✗ travel_dates fully in the past → "Those dates have already passed — when are you planning to go?"
  ✗ budget < ~₹500 / $10 for any trip → "That budget seems too low for travel — could you share a realistic number?"
  Keep all rejection messages to one friendly sentence.

────────────────────────────────────────────────
HOLISTIC VALIDATION (run once ALL fields are collected)
────────────────────────────────────────────────
Check and add a short note to validation_issues for each problem found:

  • Budget reality    — Is budget reasonable for destination + duration + num_travelers + luxury tier?
                        e.g. "₹2,000 for 7 nights in Paris for 3 people is unrealistic."
  • Luxury mismatch  — Does trip_budget_luxury match the budget?
                        e.g. budget = ₹5,000 total but luxury = Ultra-Luxury → flag.
  • Geographic outlier — Are places_to_visit within the destination COUNTRY?
                        Flag only if a place is clearly outside the geographical bounds of the destination cluster.
                        e.g. list of all destinations = India, places include "Tokyo" → flag.
                        e.g. list of all destinations = Kerala, places include "Ayodhya" → flag.
  • Travel type mismatch — Does travel_type match num_travelers?
                        e.g. travel_type = Solo but num_travelers = 4 → flag.
  • Duration check   — Is the trip long enough to cover all listed places?

If issues found → status = "validating", list issues briefly, ask user to confirm or correct.
If all clear    → status = "confirmed", say: "All good — ready to plan your trip! ✈️"

────────────────────────────────────────────────
RE-COLLECTION (if user fixes something)
────────────────────────────────────────────────
- Ask only for the fields that need correction.
- Re-run holistic validation after each correction.
- Repeat until confirmed.

RULES:
- ONE QUESTION PER MESSAGE — this is the most important rule.
- Sound like a helpful friend, not a form.
- Accept casual / shorthand answers and extract the right value.
- Only put newly provided/updated values in field_updates.
- Never echo back all collected data unless the user asks.
- Don't ask the questions repeatedly if the correct answer is already provided.
- If you catch yourself about to ask more than one question, stop and send only the first one.

Once ALL fields are confirmed and holistic validation passes, append exactly this token
on a new line at the very end of your final message:
[READY_TO_PLAN]
"""

EXTRACT_PROMPT = """Extract all trip details from the conversation below into structured fields.
For user_id: create a lowercase slug from user's name + 4 random digits, e.g. 'rahul_4821'.
Use context clues if a detail is implied (e.g. "just me" means 1 traveler, "couple" means 2).
For dates: always store as YYYY-MM-DD regardless of how the user said them.

Conversation:
{conversation}
"""


# ── Helpers ───────────────────────────────────────────────────────────────────

def _trim_to_single_question(reply: str) -> str:
    """If the model asked multiple questions, keep only the first one."""
    if reply.count("?") <= 1:
        return reply
    # Split into sentences on ? boundaries, keep the first question sentence
    import re
    # Split after each '?' (keeping the delimiter)
    parts = re.split(r'(?<=\?)\s*', reply)
    # Walk parts and accumulate until we have one sentence ending in '?'
    for i, part in enumerate(parts):
        segment = " ".join(parts[: i + 1]).strip()
        if segment.endswith("?"):
            return segment
    return reply


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2
         + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2)
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _try_extract_src_dest(conversation_text: str, llm) -> tuple[str, str]:
    """
    Quick partial extraction: returns (source, destination) if both have been
    mentioned in the conversation so far, otherwise ('', '').
    """
    prompt = (
        "From the conversation below, extract the departure city (source) and destination city "
        "only if the user has already provided BOTH in their replies.\n"
        "Reply with EXACTLY: source=<value>,dest=<value>\n"
        "If either is missing, reply: source=,dest=\n\n"
        + conversation_text
    )
    try:
        reply = llm.invoke([HumanMessage(content=prompt)]).content.strip()
        src = dest = ""
        for part in reply.split(","):
            k, _, v = part.partition("=")
            k = k.strip()
            v = v.strip()
            if k == "source":
                src = v
            elif k == "dest":
                dest = v
        return src, dest
    except Exception:
        return "", ""


def _resolve_and_validate_locations(source: str, destination: str) -> tuple[dict, dict, float]:
    """
    Fetch coordinates for source and destination, print them, and return
    (src_data, dest_data, distance_km).  Raises ValueError if either location
    cannot be resolved.
    """
    print("\n  Resolving coordinates...")
    src_data  = get_coordinates(source)
    dest_data = get_coordinates(destination)

    if not src_data:
        raise ValueError(f"Could not resolve coordinates for source: '{source}'")
    if not dest_data:
        raise ValueError(f"Could not resolve coordinates for destination: '{destination}'")

    distance_km = _haversine_km(
        src_data["lat"], src_data["lon"],
        dest_data["lat"], dest_data["lon"],
    )

    print(f"\n  📍 Source      : {src_data['display_name']}")
    print(f"     Coordinates : {src_data['lat']:.4f}°N, {src_data['lon']:.4f}°E")
    print(f"\n  📍 Destination : {dest_data['display_name']}")
    print(f"     Coordinates : {dest_data['lat']:.4f}°N, {dest_data['lon']:.4f}°E")
    print(f"\n  📏 Distance    : {distance_km:.1f} km")

    return src_data, dest_data, distance_km


# ── Entry point ────────────────────────────────────────────────────────────────

def run_user_input_agent(state: dict) -> dict:
    """
    Multi-turn conversation to collect trip preferences.
    Writes trip_preferences and user_profile into state.
    """
    print("\n" + "=" * 55)
    print("   TRIP PLANNER v3 — Agent 1: User Input")
    print("=" * 55)
    print("Type your responses below. Press Ctrl+C to exit.\n")

    llm, extract_llm = _get_llms()
    history: list = [SystemMessage(content=SYSTEM_PROMPT)]
    conversation_text = ""
    coord_injected = False  # inject [COORD_CHECK] only once per conversation

    # ── Conversation loop ──────────────────────────────────────────────────────
    while True:
        for attempt in range(1, 4):
            try:
                ai_msg = llm.invoke(history)
                break
            except OpenAIRateLimitError as e:
                if attempt == 3:
                    print(f"\n  ✗ Rate limit hit after 3 attempts — try again in a minute.")
                    raise
                # use retry_after_seconds from metadata if provided, else backoff
                retry_after = 30
                try:
                    retry_after = int(e.body["error"]["metadata"].get("retry_after_seconds", 30))
                except Exception:
                    pass
                print(f"\n  ⚠ Rate limited (attempt {attempt}/3), retrying in {retry_after}s…")
                time.sleep(retry_after)
            except OpenAIStatusError as e:
                if e.status_code == 402:
                    print(f"\n  ✗ Insufficient OpenRouter credits. Top up at https://openrouter.ai/settings/credits")
                raise
            except (OpenAIConnectionError, httpx.ConnectError, httpx.TimeoutException) as e:
                if attempt == 3:
                    print(f"\n  ✗ Could not reach AI service after 3 attempts: {e}")
                    raise
                wait = 2 ** attempt
                print(f"\n  ⚠ Network error (attempt {attempt}/3), retrying in {wait}s…")
                time.sleep(wait)
        reply = ai_msg.content
        reply = _trim_to_single_question(reply)
        print(f"\n🤖 Planner: {reply}\n")

        if "[READY_TO_PLAN]" in reply:
            break

        history.append(AIMessage(content=reply))
        conversation_text += f"Assistant: {reply}\n"

        user_input = input("You: ").strip()
        if not user_input:
            continue

        conversation_text += f"User: {user_input}\n"
        history.append(HumanMessage(content=user_input))

        # ── Mid-conversation coordinate check ─────────────────────────────────
        if not coord_injected:
            src_guess, dest_guess = _try_extract_src_dest(conversation_text, llm)
            if src_guess and dest_guess:
                coord_injected = True
                try:
                    sd, dd, dist = _resolve_and_validate_locations(src_guess, dest_guess)
                    verdict = "same location, re-ask destination" if dist < 10 else "different locations, proceed"
                    coord_msg = (
                        f"[COORD_CHECK: source='{sd['display_name']}' "
                        f"({sd['lat']:.4f},{sd['lon']:.4f}), "
                        f"dest='{dd['display_name']}' "
                        f"({dd['lat']:.4f},{dd['lon']:.4f}), "
                        f"distance={dist:.1f}km — {verdict}]"
                    )
                    history.append(HumanMessage(content=coord_msg))
                    conversation_text += f"System: {coord_msg}\n"
                except Exception:
                    pass

    # ── Extract structured preferences from conversation ───────────────────────
    print("\n  Extracting trip details...")
    prefs: TripPreferences = None
    for attempt in range(3):
        prefs = extract_llm.invoke([
            HumanMessage(content=EXTRACT_PROMPT.format(conversation=conversation_text))
        ])
        if prefs is not None:
            break
        print(f"  [extract] structured output returned None, retrying ({attempt + 1}/3)...")

    if prefs is None:
        raise RuntimeError(
            "Failed to extract trip preferences after 3 attempts. "
            "The model did not return a structured response. Please try again."
        )

    # ── Coordinate-based same-location check (re-ask until valid) ─────────────
    while True:
        try:
            src_data, dest_data, distance_km = _resolve_and_validate_locations(
                prefs.source, prefs.destination
            )
        except ValueError as e:
            print(f"\n  ✗ {e}")
            prefs.destination = input("  Please re-enter destination: ").strip()
            continue

        if distance_km < 10:
            print(f"\n  ✗ Source and destination are only {distance_km:.1f} km apart — they appear to be the same location.")
            prefs.destination = input("  Please enter a different destination: ").strip()
        else:
            break

    print(f"\n  ✓ Trip: {prefs.source} → {prefs.destination}")
    print(f"  ✓ Dates: {prefs.start_date} to {prefs.end_date}")
    print(f"  ✓ Budget: Rs.{prefs.budget:,} | Travelers: {prefs.travelers}")
    print("=" * 55)

    return {
        "user_profile": {
            "user_id":   prefs.user_id,
            "user_name": prefs.user_name,
        },
        "trip_preferences": {
            "source":           prefs.source,
            "destination":      prefs.destination,
            "start_date":       prefs.start_date,
            "end_date":         prefs.end_date,
            "budget":           prefs.budget,
            "travelers":        prefs.travelers,
            "travel_type":      prefs.travel_type,
            "hotel_pref":       prefs.hotel_pref,
            "food_pref":        prefs.food_pref,
            "transport_pref":   prefs.transport_pref,
            "special_requests": prefs.special_requests,
        },
        "coordinates": {
            "source_lat":  src_data["lat"],
            "source_lon":  src_data["lon"],
            "dest_lat":    dest_data["lat"],
            "dest_lon":    dest_data["lon"],
            "source_name": src_data["display_name"],
            "dest_name":   dest_data["display_name"],
            "distance_km": round(distance_km, 1),
        },
        "last_completed_node": "user_input_agent",
    }
