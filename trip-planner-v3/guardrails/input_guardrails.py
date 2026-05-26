from datetime import date


def run_input_guardrails(state: dict) -> dict:
    """Validate trip inputs. Returns updated guardrail_results in state."""
    prefs     = state.get("trip_preferences", {})
    warnings  = []
    errors    = []

    budget    = prefs.get("budget", 0)
    travelers = max(prefs.get("travelers", 1), 1)

    if not prefs.get("destination"):
        errors.append("No destination specified")

    if not prefs.get("source"):
        errors.append("No source city specified")

    src  = prefs.get("source", "").strip().lower()
    dest = prefs.get("destination", "").strip().lower()
    if src and dest and src == dest:
        errors.append(f"Source and destination are the same city: '{prefs.get('source')}'")


    try:
        s    = date.fromisoformat(prefs.get("start_date", ""))
        e    = date.fromisoformat(prefs.get("end_date", ""))
        days = max((e - s).days, 1)

        if s < date.today():
            errors.append("Start date is in the past")

        if days > 30:
            warnings.append(f"Long trip: {days} days — budget estimates may be less accurate")

        bppd = budget / travelers / days
        if bppd < 500:
            errors.append(f"Budget too low: Rs.{bppd:.0f}/person/day (minimum Rs.500)")
        elif bppd < 1000:
            warnings.append(f"Tight budget: Rs.{bppd:.0f}/person/day — options may be limited")

    except Exception:
        errors.append("Invalid date format — use YYYY-MM-DD")

    if not budget:
        errors.append("No budget specified")

    input_valid = len(errors) == 0

    if errors:
        print("\n  ✗ Input Guardrail Errors:")
        for e in errors:
            print(f"    • {e}")

    if warnings:
        print("\n  ⚠ Input Guardrail Warnings:")
        for w in warnings:
            print(f"    • {w}")

    existing = state.get("guardrail_results", {})
    return {
        "guardrail_results": {
            **existing,
            "input_valid":    input_valid,
            "input_errors":   errors,
            "input_warnings": warnings,
        }
    }


def check_budget_conflict(state: dict) -> bool:
    """Return True if inputs are invalid and planning should stop."""
    return not state.get("guardrail_results", {}).get("input_valid", True)
