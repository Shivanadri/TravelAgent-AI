def evaluate_itinerary(state: dict) -> dict:
    """Score itinerary pacing and variety. Called inline by review_agent."""
    days = state.get("itinerary", {}).get("days", [])
    return {
        "pacing":  _score_pacing(days),
        "variety": _score_variety(days),
    }


def _score_pacing(days: list) -> int:
    if not days:
        return 5
    counts = [len(d.get("activities", [])) for d in days]
    avg    = sum(counts) / len(counts)
    if 3 <= avg <= 5:
        return 9
    if 2 <= avg <= 6:
        return 7
    if avg < 2:
        return 5
    return 6


def _score_variety(days: list) -> int:
    if not days:
        return 5
    themes    = set(d.get("theme", "") for d in days)
    all_acts  = [a for d in days for a in d.get("activities", [])]
    locations = set(a.get("location", "") for a in all_acts)

    if not all_acts:
        return 5

    ratio = len(locations) / len(all_acts)
    if ratio >= 0.7 and len(themes) >= max(len(days) - 1, 1):
        return 9
    if ratio >= 0.5:
        return 7
    return 5
