"""
HITL Gates 1–5.
Gate 0 is embedded in agents/weather_agent.py.
Each gate receives the full state dict and returns a partial state update.
"""


def _safe_input(prompt: str, default: str = "1") -> str:
    """Return user input, or the default when stdin is not interactive (e.g. Render)."""
    try:
        return input(prompt).strip()
    except EOFError:
        return default


# ── Gate 1: Confirm trip details ───────────────────────────────────────────────

def run_gate1_confirm(state: dict) -> dict:
    prefs   = state.get("trip_preferences", {})
    weather = state.get("weather_data", {})

    print("\n" + "=" * 55)
    print("  ✋ HITL GATE 1 — Confirm Trip Details")
    print("=" * 55)
    print(f"\n  Source      : {prefs.get('source')}")
    print(f"  Destination : {prefs.get('destination')}")
    print(f"  Dates       : {prefs.get('start_date')} to {prefs.get('end_date')}")
    print(f"  Budget      : Rs.{prefs.get('budget', 0):,} | {prefs.get('travelers')} traveler(s)")
    print(f"  Type        : {prefs.get('travel_type')} | Hotel: {prefs.get('hotel_pref')}")
    print(f"  Weather     : {weather.get('condition', '?')} (score {weather.get('score', '?')}/10)")
    print(f"  Summary     : {weather.get('summary', '')}")

    print("\n  [1] Confirm — proceed with planning")
    print("  [2] Change destination")
    print("  [3] Change dates")
    print("  [4] Change budget")

    choice = _safe_input("\n  Your choice (1/2/3/4): ")

    if choice == "2":
        new_dest = _safe_input("  New destination: ", default="")
        if new_dest:
            print(f"\n  ✓ Destination changed to {new_dest}\n")
            return {
                "trip_preferences":  {**prefs, "destination": new_dest},
                "hitl_approved":     True,
                "hitl_gate":         "gate1",
                "human_feedback":    f"destination changed to {new_dest}",
                "hitl_change_count": state.get("hitl_change_count", 0) + 1,
            }

    if choice == "3":
        start = _safe_input("  New start date (YYYY-MM-DD): ", default="")
        end   = _safe_input("  New end date   (YYYY-MM-DD): ", default="")
        if start and end:
            print(f"\n  ✓ Dates changed to {start} → {end}\n")
            return {
                "trip_preferences":  {**prefs, "start_date": start, "end_date": end},
                "hitl_approved":     True,
                "hitl_gate":         "gate1",
                "human_feedback":    f"dates changed to {start} → {end}",
                "hitl_change_count": state.get("hitl_change_count", 0) + 1,
            }

    if choice == "4":
        raw = _safe_input("  New budget (INR): ", default="")
        try:
            new_budget = int(raw.replace(",", ""))
            print(f"\n  ✓ Budget changed to Rs.{new_budget:,}\n")
            return {
                "trip_preferences":  {**prefs, "budget": new_budget},
                "hitl_approved":     True,
                "hitl_gate":         "gate1",
                "human_feedback":    f"budget changed to Rs.{new_budget:,}",
                "hitl_change_count": state.get("hitl_change_count", 0) + 1,
            }
        except ValueError:
            pass

    print("\n  ✓ Trip confirmed. Proceeding...\n")
    return {
        "hitl_approved":     True,
        "hitl_gate":         "gate1",
        "human_feedback":    "confirmed",
        "hitl_change_count": state.get("hitl_change_count", 0),
    }


# ── Gate 2: Confirm / modify places list ──────────────────────────────────────

def run_gate2_places(state: dict) -> dict:
    places = state.get("places_data", {})

    print("\n" + "=" * 55)
    print("  ✋ HITL GATE 2 — Confirm Places to Visit")
    print("=" * 55)

    must_visit = list(places.get("must_visit", []))
    activities = list(places.get("activities", []))
    food_spots = list(places.get("food_spots", []))
    hidden     = list(places.get("hidden_gems", []))

    print(f"\n  Must-see places:")
    for i, p in enumerate(must_visit, 1):
        print(f"    [{i}] {p.get('name')} — {p.get('description')}")

    print(f"\n  Activities:")
    for i, p in enumerate(activities, 1):
        print(f"    [{i}] {p.get('name')} — {p.get('description')}")

    print(f"\n  Food spots:")
    for i, p in enumerate(food_spots, 1):
        print(f"    [{i}] {p.get('name')}")

    if hidden:
        print(f"\n  Hidden gems:")
        for i, p in enumerate(hidden, 1):
            print(f"    [{i}] {p.get('name')}")

    print("\n  [1] Looks good — confirm all")
    print("  [2] Add a place I want to visit")
    print("  [3] Remove a place")

    choice = _safe_input("\n  Your choice (1/2/3): ")

    user_added   = []
    user_removed = []

    if choice == "2":
        while True:
            extra = _safe_input("  Place to add (or Enter to finish): ", default="")
            if not extra:
                break
            user_added.append({"name": extra, "description": "User requested",
                                "entry_fee_inr": 0, "duration_hours": 1.0, "best_time": "Anytime"})

    elif choice == "3":
        remove = _safe_input("  Name of place to remove: ", default="")
        if remove:
            user_removed.append(remove)
            rl = remove.lower()
            must_visit = [p for p in must_visit if rl not in p.get("name", "").lower()]
            activities = [p for p in activities if rl not in p.get("name", "").lower()]

    print(f"\n  ✓ Places confirmed. {len(user_added)} added, {len(user_removed)} removed.\n")

    return {
        "places_confirmed": {
            "must_visit":   must_visit,
            "food_spots":   food_spots,
            "activities":   activities,
            "hidden_gems":  hidden,
            "packing_tips": places.get("packing_tips", []),
            "user_added":   user_added,
            "user_removed": user_removed,
        },
        "hitl_approved":     True,
        "hitl_gate":         "gate2",
        "human_feedback":    f"added={[p['name'] for p in user_added]}, removed={user_removed}",
        "hitl_change_count": state.get("hitl_change_count", 0) + (1 if user_added or user_removed else 0),
    }


# ── Gate 3: Budget approval (max 3 rounds) ────────────────────────────────────

def run_gate3_budget(state: dict) -> dict:
    budget     = state.get("budget_summary", {})
    transport  = state.get("transport_data", {})
    hotel      = state.get("hotel_data", {})
    prefs      = state.get("trip_preferences", {})
    gate_round = state.get("budget_gate_round", 0)
    travelers  = max(prefs.get("travelers", 1), 1)

    print("\n" + "=" * 55)
    print(f"  ✋ HITL GATE 3 — Budget Approval (Round {gate_round + 1}/3)")
    print("=" * 55)

    surplus   = budget.get("surplus_deficit", 0)
    status_str = f"✓ Rs.{surplus:,} to spare" if surplus >= 0 else f"⚠ Rs.{abs(surplus):,} over budget"

    print(f"\n  Your Budget    : Rs.{budget.get('total_budget', 0):,}")
    print(f"  Our Estimate   : Rs.{budget.get('total_estimate', 0):,}")
    print(f"  Status         : {status_str}")
    print(f"\n  Breakdown:")
    print(f"    Transport    : Rs.{budget.get('transport_cost', 0):,} ({transport.get('final_mode', '?')})")
    print(f"    Hotel        : Rs.{budget.get('hotel_cost', 0):,} ({hotel.get('recommended', {}).get('name', '?')})")
    print(f"    Daily var    : Rs.{budget.get('daily_variable', 0):,}/day")

    tips = budget.get("savings_tips", [])
    if tips:
        print(f"\n  Cost-saving tips:")
        for tip in tips[:3]:
            print(f"    • {tip}")

    can_revise = gate_round < 2
    print("\n  [1] Approve — proceed with this plan")
    if can_revise:
        print("  [2] Switch transport option")
        print("  [3] Switch hotel option")

    choice = _safe_input(f"\n  Your choice (1{'/2/3' if can_revise else ''}): ")

    if choice == "2" and can_revise:
        all_opts = transport.get("all_options", [])
        if all_opts:
            print("\n  Transport options:")
            for i, opt in enumerate(all_opts, 1):
                print(f"    [{i}] {opt.get('mode','?').upper()} — Rs.{opt.get('cost_per_person_inr', 0):,}/person | {opt.get('description', '')}")
            raw = _safe_input(f"  Choose (1-{len(all_opts)}): ")
            try:
                chosen = all_opts[int(raw) - 1]
                new_transport_total = chosen.get("cost_per_person_inr", 0) * travelers
                old_total           = budget.get("total_estimate", 0)
                new_total           = old_total - budget.get("transport_cost", 0) + new_transport_total
                print(f"\n  ✓ Transport switched. New estimate: Rs.{new_total:,}\n")
                return {
                    "transport_data": {
                        **state.get("transport_data", {}),
                        "final_mode":              chosen.get("mode"),
                        "description":             chosen.get("description"),
                        "estimated_cost_per_person": chosen.get("cost_per_person_inr", 0),
                        "total_transport_cost":    new_transport_total,
                    },
                    "hitl_approved":     False,
                    "hitl_gate":         "gate3",
                    "budget_gate_round": gate_round + 1,
                    "budget_revisions":  state.get("budget_revisions", []) + [{
                        "round": gate_round + 1, "action": "transport_swap",
                        "old_total": old_total,  "new_total": new_total,
                    }],
                    "human_feedback":    f"switched transport to {chosen.get('mode')}",
                    "hitl_change_count": state.get("hitl_change_count", 0) + 1,
                }
            except (ValueError, IndexError):
                pass

    elif choice == "3" and can_revise:
        all_hotels = hotel.get("top_3", [])
        nights     = hotel.get("nights", 3)
        rooms      = max(travelers // 2, 1)
        if all_hotels:
            print("\n  Hotel options:")
            for i, h in enumerate(all_hotels, 1):
                total = h.get("price_per_night_inr", 0) * nights
                print(f"    [{i}] {h.get('name','?')} — Rs.{h.get('price_per_night_inr', 0):,}/night (Rs.{total:,} total)")
            raw = _safe_input(f"  Choose (1-{len(all_hotels)}): ")
            try:
                chosen        = all_hotels[int(raw) - 1]
                new_hotel_cost = chosen.get("price_per_night_inr", 0) * nights * rooms
                old_total      = budget.get("total_estimate", 0)
                new_total      = old_total - budget.get("hotel_cost", 0) + new_hotel_cost
                print(f"\n  ✓ Hotel switched. New estimate: Rs.{new_total:,}\n")
                return {
                    "hotel_data": {
                        **state.get("hotel_data", {}),
                        "recommended":     chosen,
                        "price_per_night": chosen.get("price_per_night_inr", 0),
                        "total_stay_cost": new_hotel_cost,
                    },
                    "hitl_approved":     False,
                    "hitl_gate":         "gate3",
                    "budget_gate_round": gate_round + 1,
                    "budget_revisions":  state.get("budget_revisions", []) + [{
                        "round": gate_round + 1, "action": "hotel_swap",
                        "old_total": old_total,  "new_total": new_total,
                    }],
                    "human_feedback":    f"switched hotel to {chosen.get('name')}",
                    "hitl_change_count": state.get("hitl_change_count", 0) + 1,
                }
            except (ValueError, IndexError):
                pass

    if gate_round >= 2 and choice != "1":
        print("\n  Maximum revision rounds reached. Proceeding with current plan.\n")
    else:
        print("\n  ✓ Budget approved.\n")

    return {
        "hitl_approved":     True,
        "hitl_gate":         "gate3",
        "budget_gate_round": gate_round,
        "human_feedback":    "budget approved",
    }


# ── Gate 4: Full itinerary approval ───────────────────────────────────────────

def run_gate4_plan(state: dict) -> dict:
    itinerary = state.get("itinerary", {})
    budget    = state.get("budget_summary", {})
    review    = state.get("review_status", {})

    print("\n" + "=" * 55)
    print("  ✋ HITL GATE 4 — Final Plan Approval")
    print("=" * 55)

    print(f"\n  {itinerary.get('title', 'Trip Plan')}")
    print(f"  Review score : {review.get('score', '?')}/10 — {review.get('overall_verdict', '')}")
    print(f"\n  Day-by-day overview:")
    for day in itinerary.get("days", []):
        print(f"    Day {day.get('day')} ({day.get('date')}) — {day.get('theme')}")
        for act in day.get("activities", [])[:2]:
            print(f"      • {act.get('time', '')} {act.get('activity')} @ {act.get('location')}")

    print(f"\n  Estimated cost : Rs.{budget.get('total_estimate', 0):,}")
    print("\n  [1] Approve — generate output files")
    print("  [2] Request changes (itinerary will be regenerated)")

    choice = _safe_input("\n  Your choice (1/2): ")

    if choice == "2":
        feedback = _safe_input("  Describe the changes you want: ", default="")
        print(f"\n  ✓ Changes noted. Regenerating itinerary...\n")
        return {
            "hitl_approved":     False,
            "hitl_gate":         "gate4",
            "human_feedback":    feedback,
            "hitl_change_count": state.get("hitl_change_count", 0) + 1,
        }

    print("\n  ✓ Plan approved! Generating output files...\n")
    return {
        "hitl_approved":     True,
        "hitl_gate":         "gate4",
        "human_feedback":    "plan approved",
        "hitl_change_count": state.get("hitl_change_count", 0),
    }


# ── Gate 5: PDF generation confirmation ──────────────────────────────────────

def run_gate5_pdf(state: dict) -> dict:
    session = state.get("session_id", "")[:8]

    print("\n" + "=" * 55)
    print("  ✋ HITL GATE 5 — Generate Output Files")
    print("=" * 55)

    print(f"\n  Ready to generate:")
    print(f"    • PDF itinerary    → output/trip_{session}.pdf")
    print(f"    • WhatsApp summary → output/whatsapp_{session}.txt")

    print("\n  [1] Generate both PDF + WhatsApp summary")
    print("  [2] WhatsApp summary only (skip PDF)")
    print("  [3] Skip — don't generate anything")

    choice = _safe_input("\n  Your choice (1/2/3): ")

    if choice == "3":
        print("\n  Output generation skipped.\n")
        return {
            "hitl_approved": False,
            "hitl_gate":     "gate5",
            "human_feedback": "skipped",
        }

    generate_pdf = choice != "2"
    label = "PDF + WhatsApp" if generate_pdf else "WhatsApp only"
    print(f"\n  ✓ Will generate {label}.\n")

    return {
        "hitl_approved":  True,
        "hitl_gate":      "gate5",
        "human_feedback": f"generate_pdf={generate_pdf}",
    }
