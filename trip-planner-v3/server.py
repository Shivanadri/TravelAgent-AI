import os
import sys
import uuid
import logging

from dotenv import load_dotenv
load_dotenv()

os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Optional, Dict

app = FastAPI(title="Trip Planner v3 API")

_chat_sessions: Dict[str, "UserInputChat"] = {}
_trip_results:  Dict[str, dict] = {}


# ── Request / Response models ──────────────────────────────────────────────────

class MessageRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    session_id: str
    reply: str
    ready: bool


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "service": "Trip Planner v3 API",
        "status": "running",
        "docs": "/docs",
        "endpoints": {
            "POST /session":              "Start a new conversation, returns first question",
            "POST /chat/{session_id}":    "Send a message, get next question",
            "POST /plan/{session_id}":    "Trigger planning once conversation is complete",
            "GET  /result/{session_id}":  "Poll for the completed itinerary",
            "GET  /health":               "Health check",
        },
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/session", response_model=ChatResponse)
def create_session():
    """Start a new trip-planning conversation. Returns the first question."""
    from agents.user_input_agent import UserInputChat
    session_id = str(uuid.uuid4())
    chat = UserInputChat()
    first_question = chat.start()
    _chat_sessions[session_id] = chat
    return ChatResponse(session_id=session_id, reply=first_question, ready=chat.ready)


@app.post("/chat/{session_id}", response_model=ChatResponse)
def chat(session_id: str, req: MessageRequest):
    """Send one user reply and get the next question."""
    chat = _chat_sessions.get(session_id)
    if not chat:
        raise HTTPException(404, "Session not found. Create one with POST /session")
    if chat.ready:
        raise HTTPException(400, "Conversation complete. Call POST /plan/{session_id}.")

    reply = chat.reply(req.message)
    return ChatResponse(session_id=session_id, reply=reply, ready=chat.ready)


@app.post("/plan/{session_id}")
def start_plan(session_id: str, background_tasks: BackgroundTasks):
    """Trigger the planning pipeline once the conversation is complete."""
    chat = _chat_sessions.get(session_id)
    if not chat:
        raise HTTPException(404, "Session not found.")
    if not chat.ready:
        raise HTTPException(400, "Conversation not complete yet. Keep chatting.")

    if _trip_results.get(session_id, {}).get("status") in ("running", "complete"):
        return {"session_id": session_id, "status": _trip_results[session_id]["status"]}

    _trip_results[session_id] = {"status": "running"}
    background_tasks.add_task(_run_planning, session_id, chat)
    return {"session_id": session_id, "status": "started"}


@app.get("/result/{session_id}")
def get_result(session_id: str):
    """Poll for the planning result."""
    result = _trip_results.get(session_id)
    if not result:
        raise HTTPException(404, "No planning job found for this session.")
    return result


# ── Background planning task ───────────────────────────────────────────────────

def _run_planning(session_id: str, chat) -> None:
    """Runs in a background thread. Extracts preferences then invokes the graph."""
    logger = logging.getLogger(session_id)
    try:
        _trip_results[session_id]["status"] = "extracting"
        prefs_state = chat.extract()

        _trip_results[session_id]["status"] = "planning"

        # Import here so load_dotenv() runs before any agent-level module init
        from main import build_graph
        from memory.memory_store import save as save_memory

        os.makedirs("logs", exist_ok=True)
        os.makedirs("checkpoints", exist_ok=True)
        fh = logging.FileHandler(f"logs/session_{session_id}.log", encoding="utf-8")
        fh.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)-5s %(message)s", "%H:%M:%S"))
        logger.addHandler(fh)
        logger.setLevel(logging.INFO)

        graph = build_graph()

        initial_state = {
            "session_id":        session_id,
            "checkpoint_path":   f"checkpoints/trip_{session_id}.db",
            "log_path":          f"logs/session_{session_id}.log",
            "cache_hits":        {},
            "retry_count":       0,
            "hitl_change_count": 0,
            "budget_gate_round": 0,
            "budget_revisions":  [],
            "failed_agents":     [],
            "api_status":        {},
            **prefs_state,
        }
        config = {"configurable": {"thread_id": session_id}, "recursion_limit": 50}
        final_state = graph.invoke(initial_state, config=config)

        # Persist to user memory
        user_id = final_state.get("user_profile", {}).get("user_id", "guest")
        prefs   = final_state.get("trip_preferences", {})
        itin    = final_state.get("itinerary", {})
        if user_id and prefs.get("destination"):
            try:
                save_memory(user_id, prefs, itin)
            except Exception as mem_err:
                logger.warning(f"Memory save failed: {mem_err}")

        _trip_results[session_id] = {
            "status":                "complete",
            "session_id":            session_id,
            "trip_preferences":      final_state.get("trip_preferences", {}),
            "itinerary":             final_state.get("itinerary", {}),
            "budget_summary":        final_state.get("budget_summary", {}),
            "review_status":         final_state.get("review_status", {}),
            "weather_data":          final_state.get("weather_data", {}),
            "transport_data":        final_state.get("transport_data", {}),
            "hotel_data":            final_state.get("hotel_data", {}),
            "coordinates":           final_state.get("coordinates", {}),
            "pdf_path":              final_state.get("pdf_path"),
            "whatsapp_summary_path": final_state.get("whatsapp_summary_path"),
        }
        logger.info("SESSION COMPLETE")

    except Exception as exc:
        logger.error(f"Planning failed: {exc}", exc_info=True)
        _trip_results[session_id] = {"status": "failed", "error": str(exc)}


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
