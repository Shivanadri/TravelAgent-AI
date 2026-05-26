def run_output_guardrails(state: dict) -> dict:
    """Validate itinerary output before it goes to the review agent."""
    itinerary     = state.get("itinerary", {})
    budget_sum    = state.get("budget_summary", {})
    prefs         = state.get("trip_preferences", {})
    daily_targets = state.get("daily_budget_targets", [])

    warnings = []

    days = itinerary.get("days", [])
    if not days:
        warnings.append("Itinerary has no days planned")

    expected = len(daily_targets)
    if expected and len(days) != expected:
        warnings.append(f"Day count mismatch: expected {expected}, got {len(days)}")

    for target, day in zip(daily_targets, days):
        actual = day.get("estimated_total_inr", 0)
        max_t  = target.get("max_inr", float("inf"))
        if actual > max_t * 1.2:
            warnings.append(
                f"Day {day.get('day')} over budget: Rs.{actual:,} vs max Rs.{int(max_t):,}"
            )

    total_est = budget_sum.get("total_estimate", 0)
    total_bud = prefs.get("budget", 0)
    if total_bud and total_est > total_bud * 1.1:
        warnings.append(
            f"Total estimate Rs.{total_est:,} exceeds budget Rs.{total_bud:,} by >10%"
        )

    if not itinerary.get("title"):
        warnings.append("Itinerary missing title")

    if not itinerary.get("packing_list"):
        warnings.append("Itinerary missing packing list")

    output_valid = not any(
        kw in w for w in warnings for kw in ("over budget", "mismatch", "no days")
    )

    if warnings:
        print("\n  ⚠ Output Guardrail Warnings:")
        for w in warnings:
            print(f"    • {w}")

    existing = state.get("guardrail_results", {})
    return {
        "guardrail_results": {
            **existing,
            "output_valid":    output_valid,
            "output_warnings": warnings,
        }
    }
