def evaluate_plan(state: dict) -> dict:
    """Score preference alignment and distance-vs-budget fit."""
    prefs      = state.get("trip_preferences", {})
    hotel_data = state.get("hotel_data", {})
    transport  = state.get("transport_data", {})
    budget_sum = state.get("budget_summary", {})

    pref_score = _score_preference_alignment(prefs, hotel_data, transport)
    dist_score = _score_distance_vs_budget(transport, budget_sum)

    return {
        "preference_alignment": pref_score,
        "distance_vs_budget":   dist_score,
    }


def _score_preference_alignment(prefs, hotel_data, transport):
    score = 10

    hotel_pref  = prefs.get("hotel_pref", "mid")
    hotel_cat   = hotel_data.get("recommended", {}).get("category", "mid")
    if hotel_pref != hotel_cat:
        score -= 2

    transport_pref = prefs.get("transport_pref", "any")
    final_mode     = transport.get("final_mode", "")
    if transport_pref != "any" and transport_pref.lower() != final_mode.lower():
        score -= 2

    return max(score, 4)


def _score_distance_vs_budget(transport, budget_sum):
    duration  = transport.get("duration_hours", 0)
    within_b  = budget_sum.get("within_budget", True)

    if within_b and duration <= 8:
        return 9
    if within_b or duration <= 12:
        return 7
    return 5
