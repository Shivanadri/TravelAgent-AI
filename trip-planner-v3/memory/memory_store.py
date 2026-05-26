import os
import json
from datetime import datetime

MEMORY_DIR = os.path.join("memory", "user_memories")


def _path(user_id: str) -> str:
    os.makedirs(MEMORY_DIR, exist_ok=True)
    return os.path.join(MEMORY_DIR, f"{user_id}.json")


def load(user_id: str) -> dict:
    """Load memory for a user. Returns empty context for first-time users."""
    fpath = _path(user_id)
    if not os.path.exists(fpath):
        return {
            "user_id":               user_id,
            "past_trips":            [],
            "preferred_hotel":       None,
            "preferred_transport":   None,
            "favourite_destinations": [],
            "avg_budget":            None,
            "notes":                 "",
            "total_trips_planned":   0,
        }
    with open(fpath, "r", encoding="utf-8") as f:
        return json.load(f)


def save(user_id: str, trip_preferences: dict, itinerary_summary: dict) -> None:
    """
    Append the completed trip to the user's memory file.
    Called after Gate 5 (PDF confirmed).
    """
    memory = load(user_id)

    trip_record = {
        "destination":    trip_preferences.get("destination"),
        "start_date":     trip_preferences.get("start_date"),
        "end_date":       trip_preferences.get("end_date"),
        "budget":         trip_preferences.get("budget"),
        "travelers":      trip_preferences.get("travelers"),
        "travel_type":    trip_preferences.get("travel_type"),
        "hotel_pref":     trip_preferences.get("hotel_pref"),
        "transport_pref": trip_preferences.get("transport_pref"),
        "itinerary_title": itinerary_summary.get("title", ""),
        "planned_on":     datetime.now().strftime("%Y-%m-%d"),
    }

    memory["past_trips"].append(trip_record)
    memory["total_trips_planned"] = len(memory["past_trips"])

    # Update rolling preferences from last 3 trips
    recent = memory["past_trips"][-3:]
    hotel_counts = {}
    transport_counts = {}
    budgets = []
    destinations = []

    for t in recent:
        h = t.get("hotel_pref")
        if h:
            hotel_counts[h] = hotel_counts.get(h, 0) + 1
        tr = t.get("transport_pref")
        if tr:
            transport_counts[tr] = transport_counts.get(tr, 0) + 1
        b = t.get("budget")
        if b:
            budgets.append(b)
        d = t.get("destination")
        if d and d not in memory["favourite_destinations"]:
            destinations.append(d)

    if hotel_counts:
        memory["preferred_hotel"] = max(hotel_counts, key=hotel_counts.get)
    if transport_counts:
        memory["preferred_transport"] = max(transport_counts, key=transport_counts.get)
    if budgets:
        memory["avg_budget"] = sum(budgets) // len(budgets)
    memory["favourite_destinations"] = (
        memory["favourite_destinations"] + destinations
    )[-5:]  # keep last 5

    with open(_path(user_id), "w", encoding="utf-8") as f:
        json.dump(memory, f, ensure_ascii=False, indent=2)
