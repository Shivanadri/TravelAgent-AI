import os
import uuid
import math
import logging
from dotenv import load_dotenv

# load_dotenv MUST run before any agent imports so os.getenv() works at module level
load_dotenv()

from langgraph.graph import StateGraph, START, END

from state import TripState
from checkpointer import get_checkpointer, list_incomplete_sessions
from rich_ui import start_display, update_status, stop_display, pause_display, resume_display, console

from agents.user_input_agent  import run_user_input_agent
from agents.memory_agent      import run_memory_agent
from agents.weather_agent     import run_weather_agent
from agents.transport_agent   import run_transport_agent
from agents.hotel_agent       import run_hotel_agent
from agents.places_agent      import run_places_agent
from agents.budget_agent      import run_budget_agent
from agents.itinerary_agent   import run_itinerary_agent
from agents.review_agent      import run_review_agent
from agents.orchestrator_agent import run_orchestrator_agent
from agents.pdf_agent         import run_pdf_agent
from api_clients.geocoding_client import get_coordinates
from hitl.checkpoints import (
    run_gate1_confirm, run_gate2_places,
    run_gate3_budget, run_gate4_plan, run_gate5_pdf,
)
from guardrails.input_guardrails  import run_input_guardrails
from guardrails.output_guardrails import run_output_guardrails
from memory.memory_store import save as save_memory


# ── Haversine distance (km) ────────────────────────────────────────────────────

def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


# ── Logger ─────────────────────────────────────────────────────────────────────

def _setup_logger(session_id: str) -> logging.Logger:
    os.makedirs("logs", exist_ok=True)
    logger = logging.getLogger(session_id)
    logger.setLevel(getattr(logging, os.getenv("LOG_LEVEL", "INFO")))
    fh = logging.FileHandler(f"logs/session_{session_id}.log", encoding="utf-8")
    fh.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)-5s %(message)s", "%H:%M:%S"))
    logger.addHandler(fh)
    return logger


# ── Node wrappers ──────────────────────────────────────────────────────────────

def user_input_node(state: TripState) -> dict:
    update_status("user_input", "running")
    logger = logging.getLogger(state.get("session_id", ""))
    logger.info("user_input_agent START")
    result = run_user_input_agent(state)
    update_status("user_input", "done", "interactive")
    logger.info(f"user_input_agent DONE  dest={result['trip_preferences']['destination']}")
    return result


def input_guardrail_node(state: TripState) -> dict:
    """Validate trip inputs; warns but never blocks the flow."""
    return run_input_guardrails(state)


def memory_node(state: TripState) -> dict:
    # Start the live panel here — after user_input completes — so the
    # conversation is never mixed with the status table.
    prefs      = state.get("trip_preferences", {})
    trip_title = f"{prefs.get('source', '?')} → {prefs.get('destination', '?')}"
    start_display(trip_title)

    update_status("memory", "running")
    logger = logging.getLogger(state.get("session_id", ""))
    logger.info("memory_agent START")
    result = run_memory_agent(state)
    trips  = result["memory_context"].get("total_trips_planned", 0)
    update_status("memory", "done", f"{trips} past trip(s)")
    logger.info(f"memory_agent DONE  past_trips={trips}")
    return result


def geocoding_node(state: TripState) -> dict:
    update_status("geocoding", "running")
    logger = logging.getLogger(state.get("session_id", ""))
    logger.info("geocoding START")

    prefs      = state.get("trip_preferences", {})
    cache_hits = state.get("cache_hits", {})

    src_data  = get_coordinates(prefs.get("source", ""))
    dest_data = get_coordinates(prefs.get("destination", ""))

    if not src_data or not dest_data:
        update_status("geocoding", "failed", "error")
        logger.error("geocoding FAILED — could not resolve coordinates")
        return {"error": "Could not resolve coordinates for source or destination."}

    cache_hits["geocode"] = (
        src_data.get("source") == "cache" or dest_data.get("source") == "cache"
    )

    # ── Coordinate-based same-location check ──────────────────────────────────
    distance_km = _haversine_km(
        src_data["lat"], src_data["lon"],
        dest_data["lat"], dest_data["lon"],
    )
    existing_guardrail = state.get("guardrail_results", {})
    geo_errors   = list(existing_guardrail.get("input_errors",   []))
    geo_warnings = list(existing_guardrail.get("input_warnings", []))

    if distance_km < 10:
        msg = (
            f"Source '{prefs.get('source')}' and destination '{prefs.get('destination')}' "
            f"are only {distance_km:.1f} km apart — they appear to be the same location."
        )
        geo_errors.append(msg)
        print(f"\n  ✗ Geocoding check: {msg}")
        logger.warning(msg)
    elif distance_km < 30:
        msg = (
            f"Source and destination are only {distance_km:.1f} km apart "
            f"({prefs.get('source')} ↔ {prefs.get('destination')}) — are you sure?"
        )
        geo_warnings.append(msg)
        print(f"\n  ⚠ Geocoding check: {msg}")
        logger.warning(msg)

    update_status("geocoding", "done", f"OWM ({distance_km:.0f} km)")
    logger.info(f"geocoding DONE  dest=({dest_data['lat']},{dest_data['lon']})  distance={distance_km:.1f} km")

    return {
        "coordinates": {
            "source_lat":    src_data["lat"],
            "source_lon":    src_data["lon"],
            "dest_lat":      dest_data["lat"],
            "dest_lon":      dest_data["lon"],
            "source_name":   src_data["display_name"],
            "dest_name":     dest_data["display_name"],
            "distance_km":   round(distance_km, 1),
        },
        "guardrail_results": {
            **existing_guardrail,
            "input_errors":   geo_errors,
            "input_warnings": geo_warnings,
            "input_valid":    len(geo_errors) == 0,
        },
        "cache_hits":          cache_hits,
        "last_completed_node": "geocoding_node",
    }


def weather_node(state: TripState) -> dict:
    logger = logging.getLogger(state.get("session_id", ""))
    logger.info("weather_agent START")
    result = run_weather_agent(state)
    logger.info(f"weather_agent DONE  score={result.get('weather_data', {}).get('score')}")
    return result


def gate1_node(state: TripState) -> dict:
    pause_display()
    result = run_gate1_confirm(state)
    resume_display()
    return result


def transport_node(state: TripState) -> dict:
    logger = logging.getLogger(state.get("session_id", ""))
    logger.info("transport_agent START")
    pause_display()
    result = run_transport_agent(state)
    resume_display()
    logger.info("transport_agent DONE")
    return result


def hotel_node(state: TripState) -> dict:
    logger = logging.getLogger(state.get("session_id", ""))
    logger.info("hotel_agent START")
    pause_display()
    result = run_hotel_agent(state)
    resume_display()
    logger.info("hotel_agent DONE")
    return result


def places_node(state: TripState) -> dict:
    logger = logging.getLogger(state.get("session_id", ""))
    logger.info("places_agent START")
    result = run_places_agent(state)
    logger.info("places_agent DONE")
    return result


def gate2_node(state: TripState) -> dict:
    pause_display()
    result = run_gate2_places(state)
    resume_display()
    return result


def budget_node(state: TripState) -> dict:
    logger = logging.getLogger(state.get("session_id", ""))
    logger.info("budget_agent START")
    result = run_budget_agent(state)
    logger.info("budget_agent DONE")
    return result


def gate3_node(state: TripState) -> dict:
    pause_display()
    result = run_gate3_budget(state)
    resume_display()
    return result


def itinerary_node(state: TripState) -> dict:
    logger = logging.getLogger(state.get("session_id", ""))
    logger.info("itinerary_agent START")
    result = run_itinerary_agent(state)
    logger.info("itinerary_agent DONE")
    return result


def output_guardrail_node(state: TripState) -> dict:
    return run_output_guardrails(state)


def review_node(state: TripState) -> dict:
    logger = logging.getLogger(state.get("session_id", ""))
    logger.info("review_agent START")
    result = run_review_agent(state)
    logger.info(f"review_agent DONE  score={result.get('review_status', {}).get('score')}")
    return result


def orchestrator_node(state: TripState) -> dict:
    update_status("orchestrator", "running")
    logger = logging.getLogger(state.get("session_id", ""))
    logger.info("orchestrator START")
    result = run_orchestrator_agent(state)
    approved = result.get("orchestrator_decision", {}).get("approved")
    update_status("orchestrator", "done", "approved" if approved else "retry")
    logger.info(f"orchestrator DONE  approved={approved}")
    return result


def gate4_node(state: TripState) -> dict:
    pause_display()
    result = run_gate4_plan(state)
    resume_display()
    return result


def gate5_node(state: TripState) -> dict:
    pause_display()
    result = run_gate5_pdf(state)
    resume_display()
    return result


def pdf_node(state: TripState) -> dict:
    logger = logging.getLogger(state.get("session_id", ""))
    logger.info("pdf_agent START")
    result = run_pdf_agent(state)
    logger.info("pdf_agent DONE")
    return result


# ── Routing functions ──────────────────────────────────────────────────────────

def _route_after_gate3(state: TripState) -> str:
    """After gate3: if approved or max rounds → itinerary, else re-run budget."""
    if state.get("hitl_approved") and state.get("hitl_gate") == "gate3":
        return "itinerary"
    if state.get("budget_gate_round", 0) >= 3:
        return "itinerary"
    return "budget"


def _route_after_orchestrator(state: TripState) -> str:
    """After orchestrator: approve → gate4, retry → appropriate agent."""
    decision = state.get("orchestrator_decision", {})
    if decision.get("approved") or decision.get("force_stop"):
        return "gate4"
    retry_agents = decision.get("retry_agents", [])
    if "places" in retry_agents:
        return "places"
    if "budget" in retry_agents:
        return "budget"
    return "itinerary"


def _route_after_gate4(state: TripState) -> str:
    """After gate4: approved → gate5, changes requested → re-run itinerary."""
    if state.get("hitl_approved") and state.get("hitl_gate") == "gate4":
        return "gate5"
    return "itinerary"


def _route_after_gate5(state: TripState) -> str:
    """After gate5: approved → pdf, skipped → end."""
    if state.get("hitl_approved") and state.get("hitl_gate") == "gate5":
        return "pdf"
    return "end"


# ── Build full graph ───────────────────────────────────────────────────────────

def build_graph():
    checkpointer = get_checkpointer()
    workflow     = StateGraph(TripState)

    # Register all nodes
    workflow.add_node("user_input",       user_input_node)
    workflow.add_node("input_guardrail",  input_guardrail_node)
    workflow.add_node("memory",           memory_node)
    workflow.add_node("geocoding",        geocoding_node)
    workflow.add_node("weather",          weather_node)
    workflow.add_node("gate1_confirm",    gate1_node)
    workflow.add_node("transport",        transport_node)
    workflow.add_node("hotel",            hotel_node)
    workflow.add_node("places",           places_node)
    workflow.add_node("gate2_places",     gate2_node)
    workflow.add_node("budget",           budget_node)
    workflow.add_node("gate3_budget",     gate3_node)
    workflow.add_node("itinerary",        itinerary_node)
    workflow.add_node("output_guardrail", output_guardrail_node)
    workflow.add_node("review",           review_node)
    workflow.add_node("orchestrator",     orchestrator_node)
    workflow.add_node("gate4_plan",       gate4_node)
    workflow.add_node("gate5_pdf",        gate5_node)
    workflow.add_node("pdf_agent",        pdf_node)

    # Sequential edges (no branching)
    workflow.add_edge(START,             "user_input")
    workflow.add_edge("user_input",      "input_guardrail")
    workflow.add_edge("input_guardrail", "memory")
    workflow.add_edge("memory",          "geocoding")
    workflow.add_edge("geocoding",       "weather")
    workflow.add_edge("weather",         "gate1_confirm")
    workflow.add_edge("gate1_confirm",   "transport")
    workflow.add_edge("transport",       "hotel")
    workflow.add_edge("hotel",           "places")
    workflow.add_edge("places",          "gate2_places")
    workflow.add_edge("gate2_places",    "budget")
    workflow.add_edge("budget",          "gate3_budget")
    workflow.add_edge("itinerary",       "output_guardrail")
    workflow.add_edge("output_guardrail","review")
    workflow.add_edge("review",          "orchestrator")
    workflow.add_edge("pdf_agent",       END)

    # Conditional edges (with looping)
    workflow.add_conditional_edges(
        "gate3_budget",
        _route_after_gate3,
        {"itinerary": "itinerary", "budget": "budget"},
    )
    workflow.add_conditional_edges(
        "orchestrator",
        _route_after_orchestrator,
        {"gate4": "gate4_plan", "itinerary": "itinerary",
         "budget": "budget",   "places":    "places"},
    )
    workflow.add_conditional_edges(
        "gate4_plan",
        _route_after_gate4,
        {"gate5": "gate5_pdf", "itinerary": "itinerary"},
    )
    workflow.add_conditional_edges(
        "gate5_pdf",
        _route_after_gate5,
        {"pdf": "pdf_agent", "end": END},
    )

    return workflow.compile(checkpointer=checkpointer)


# ── Session management ─────────────────────────────────────────────────────────

def _pick_or_create_session() -> str:
    incomplete = list_incomplete_sessions()
    if incomplete:
        console.print("\n[bold yellow]Incomplete sessions found:[/]")
        for i, s in enumerate(incomplete, 1):
            console.print(f"  [{i}] Resume session {s['session_id']}")
        console.print("  [N] Start new session")
        choice = input("\nYour choice: ").strip().upper()
        if choice.isdigit() and 1 <= int(choice) <= len(incomplete):
            return incomplete[int(choice) - 1]["session_id"]
    return str(uuid.uuid4())


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    console.print("\n[bold cyan]🧳 Trip Planner v3 — Multi-Agent System[/]\n")

    session_id = _pick_or_create_session()
    log_path   = f"logs/session_{session_id}.log"
    logger     = _setup_logger(session_id)
    logger.info(f"SESSION START  id={session_id}")

    graph = build_graph()

    initial_state: TripState = {
        "session_id":        session_id,
        "checkpoint_path":   f"checkpoints/trip_{session_id}.db",
        "log_path":          log_path,
        "cache_hits":        {},
        "retry_count":       0,
        "hitl_change_count": 0,
        "budget_gate_round": 0,
        "budget_revisions":  [],
        "failed_agents":     [],
        "api_status":        {},
    }

    config = {"configurable": {"thread_id": session_id}, "recursion_limit": 50}

    try:
        final_state = graph.invoke(initial_state, config=config)
        stop_display()
        logger.info("SESSION COMPLETE")

        prefs     = final_state.get("trip_preferences", {})
        coords    = final_state.get("coordinates", {})
        memory    = final_state.get("memory_context", {})
        weather   = final_state.get("weather_data", {})
        transport = final_state.get("transport_data", {})
        hotel     = final_state.get("hotel_data", {})
        budget    = final_state.get("budget_summary", {})
        review    = final_state.get("review_status", {})

        # Persist this trip to the user's memory for future sessions
        user_id = final_state.get("user_profile", {}).get("user_id", "guest")
        prefs_for_save = final_state.get("trip_preferences", {})
        itin_for_save  = final_state.get("itinerary", {})
        if user_id and prefs_for_save.get("destination"):
            try:
                save_memory(user_id, prefs_for_save, itin_for_save)
                logger.info(f"Memory saved for user {user_id}")
            except Exception as mem_err:
                logger.warning(f"Memory save failed: {mem_err}")

        console.print("\n[bold green]✓ Trip planning complete![/]")
        console.print(f"  Destination  : {prefs.get('destination')} ({coords.get('dest_lat')}, {coords.get('dest_lon')})")
        console.print(f"  Dates        : {prefs.get('start_date')} → {prefs.get('end_date')}")
        console.print(f"  Budget       : Rs.{prefs.get('budget', 0):,} | {prefs.get('travelers')} traveler(s)")
        console.print(f"  Transport    : {transport.get('final_mode', '?').upper()}")
        console.print(f"  Hotel        : {hotel.get('recommended', {}).get('name', 'TBD')}")
        console.print(f"  Weather      : {weather.get('condition', '?')} (score {weather.get('score', '?')}/10)")
        console.print(f"  Memory       : {memory.get('total_trips_planned', 0)} past trip(s) loaded")
        console.print(f"  Plan quality : {review.get('score', '?')}/10 — {review.get('overall_verdict', '')}")
        console.print(f"  Total est.   : Rs.{budget.get('total_estimate', 0):,}")

        if final_state.get("pdf_path"):
            console.print(f"\n  [bold]PDF[/]       : {final_state['pdf_path']}")
        if final_state.get("whatsapp_summary_path"):
            console.print(f"  [bold]WhatsApp[/]  : {final_state['whatsapp_summary_path']}")

        console.print(f"\n[dim]Session ID : {session_id}[/]")
        console.print(f"[dim]Log        : {log_path}[/]\n")

    except KeyboardInterrupt:
        stop_display()
        logger.warning("SESSION INTERRUPTED by user")
        console.print("\n[yellow]Session paused. Run again with the same session ID to resume.[/]")
        console.print(f"[dim]Session ID: {session_id}[/]\n")


if __name__ == "__main__":
    main()
