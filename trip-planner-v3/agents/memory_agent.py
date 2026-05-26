from memory.memory_store import load


def run_memory_agent(state: dict) -> dict:
    """
    Load the user's past trip history and preferences from their JSON file.
    Writes memory_context into state.
    First-time users get an empty context — no error.
    """
    print("\n" + "=" * 55)
    print("   TRIP PLANNER v3 — Agent 2: Memory Agent")
    print("=" * 55)

    user_id = state.get("user_profile", {}).get("user_id", "guest")
    memory = load(user_id)

    trips_count = memory.get("total_trips_planned", 0)
    if trips_count == 0:
        print("  First-time user — no memory found. Starting fresh.")
    else:
        print(f"  Found {trips_count} past trip(s) for user '{user_id}'")
        fav = memory.get("favourite_destinations", [])
        if fav:
            print(f"  Favourite destinations: {', '.join(fav)}")
        if memory.get("preferred_hotel"):
            print(f"  Preferred hotel: {memory['preferred_hotel']}")
        if memory.get("avg_budget"):
            print(f"  Average budget: Rs.{memory['avg_budget']:,}")

    print("=" * 55)

    return {
        "memory_context":      memory,
        "last_completed_node": "memory_agent",
    }
