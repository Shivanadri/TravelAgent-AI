import os
import sys
import uuid
import logging

from dotenv import load_dotenv
load_dotenv()

os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse
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

@app.get("/", response_class=HTMLResponse)
def root():
    return HTMLResponse(content=_UI_HTML)


_UI_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>🧳 Trip Planner v3</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: 'Segoe UI', sans-serif; background: #0f172a; color: #e2e8f0; height: 100vh; display: flex; flex-direction: column; }
  header { background: #1e293b; padding: 16px 24px; border-bottom: 1px solid #334155; }
  header h1 { font-size: 1.3rem; color: #38bdf8; }
  header p  { font-size: 0.8rem; color: #94a3b8; margin-top: 2px; }
  #chat { flex: 1; overflow-y: auto; padding: 24px; display: flex; flex-direction: column; gap: 12px; }
  .bubble { max-width: 72%; padding: 12px 16px; border-radius: 16px; line-height: 1.5; font-size: 0.95rem; }
  .bot  { background: #1e293b; border: 1px solid #334155; align-self: flex-start; border-bottom-left-radius: 4px; }
  .user { background: #0ea5e9; align-self: flex-end; border-bottom-right-radius: 4px; color: #fff; }
  .sys  { background: #1e3a2e; border: 1px solid #166534; align-self: center; font-size: 0.82rem; color: #86efac; text-align: center; padding: 8px 16px; border-radius: 8px; }
  #footer { background: #1e293b; border-top: 1px solid #334155; padding: 12px 16px; display: flex; gap: 10px; }
  #msg { flex: 1; background: #0f172a; border: 1px solid #334155; color: #e2e8f0; border-radius: 10px; padding: 10px 14px; font-size: 0.95rem; outline: none; }
  #msg:focus { border-color: #38bdf8; }
  #send { background: #0ea5e9; color: #fff; border: none; border-radius: 10px; padding: 10px 20px; cursor: pointer; font-size: 0.95rem; }
  #send:disabled { background: #334155; cursor: not-allowed; }
  #plan-card { background: #1e293b; border: 1px solid #334155; border-radius: 12px; padding: 20px; margin: 12px 0; }
  #plan-card h2 { color: #38bdf8; margin-bottom: 12px; font-size: 1.1rem; }
  #plan-card table { width: 100%; border-collapse: collapse; font-size: 0.85rem; }
  #plan-card td { padding: 6px 10px; border-bottom: 1px solid #334155; vertical-align: top; }
  #plan-card td:first-child { color: #94a3b8; width: 35%; }
  .day-block { background: #0f172a; border-radius: 8px; padding: 12px; margin-top: 8px; }
  .day-block h3 { color: #f0abfc; font-size: 0.9rem; margin-bottom: 6px; }
  .day-block li { font-size: 0.82rem; color: #cbd5e1; margin-left: 16px; margin-top: 3px; }
  #spinner { display: none; align-self: flex-start; }
  #spinner.active { display: flex; align-items: center; gap: 8px; }
  .dot { width: 8px; height: 8px; background: #38bdf8; border-radius: 50%; animation: bounce 1s infinite; }
  .dot:nth-child(2) { animation-delay: .15s; }
  .dot:nth-child(3) { animation-delay: .3s; }
  @keyframes bounce { 0%,80%,100% { transform: translateY(0); } 40% { transform: translateY(-6px); } }
</style>
</head>
<body>
<header>
  <h1>🧳 Trip Planner v3</h1>
  <p>AI-powered multi-agent travel planner</p>
</header>
<div id="chat">
  <div id="spinner"><div class="dot"></div><div class="dot"></div><div class="dot"></div></div>
</div>
<div id="footer">
  <input id="msg" type="text" placeholder="Type your reply..." disabled/>
  <button id="send" disabled>Send</button>
</div>

<script>
let sessionId = null;
let polling   = null;

const chat   = document.getElementById('chat');
const msg    = document.getElementById('msg');
const send   = document.getElementById('send');
const spinner = document.getElementById('spinner');

function addBubble(text, type) {
  spinner.classList.remove('active');
  const d = document.createElement('div');
  d.className = 'bubble ' + type;
  d.textContent = text;
  chat.insertBefore(d, spinner);
  chat.scrollTop = chat.scrollHeight;
}

function addSys(text) {
  spinner.classList.remove('active');
  const d = document.createElement('div');
  d.className = 'sys';
  d.textContent = text;
  chat.insertBefore(d, spinner);
  chat.scrollTop = chat.scrollHeight;
}

function showSpinner() {
  spinner.classList.add('active');
  chat.scrollTop = chat.scrollHeight;
}

function setInput(enabled) {
  msg.disabled  = !enabled;
  send.disabled = !enabled;
  if (enabled) msg.focus();
}

async function startSession() {
  showSpinner();
  const res  = await fetch('/session', { method: 'POST' });
  const data = await res.json();
  sessionId  = data.session_id;
  addBubble(data.reply, 'bot');
  setInput(true);
}

async function sendMessage() {
  const text = msg.value.trim();
  if (!text) return;
  msg.value = '';
  setInput(false);
  addBubble(text, 'user');
  showSpinner();

  const res  = await fetch('/chat/' + sessionId, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ message: text }),
  });
  const data = await res.json();
  addBubble(data.reply.replace('[READY_TO_PLAN]', '').trim(), 'bot');

  if (data.ready) {
    addSys('✅ All details collected — generating your trip plan...');
    startPlanning();
  } else {
    setInput(true);
  }
}

async function startPlanning() {
  await fetch('/plan/' + sessionId, { method: 'POST' });
  polling = setInterval(pollResult, 4000);
}

async function pollResult() {
  const res  = await fetch('/result/' + sessionId);
  const data = await res.json();

  if (data.status === 'complete') {
    clearInterval(polling);
    renderPlan(data);
  } else if (data.status === 'failed') {
    clearInterval(polling);
    addSys('❌ Planning failed: ' + (data.error || 'unknown error'));
  } else {
    const labels = { extracting: 'Extracting preferences...', planning: 'Planning your trip...' };
    const el = document.querySelector('.sys:last-of-type');
    if (el) el.textContent = '⏳ ' + (labels[data.status] || 'Working...');
  }
}

function renderPlan(data) {
  const prefs    = data.trip_preferences   || {};
  const budget   = data.budget_summary     || {};
  const review   = data.review_status      || {};
  const weather  = data.weather_data       || {};
  const transport = data.transport_data    || {};
  const hotel    = data.hotel_data         || {};
  const itin     = data.itinerary          || {};

  const card = document.createElement('div');
  card.id = 'plan-card';
  card.innerHTML = `
    <h2>✈️ ${itin.title || prefs.source + ' → ' + prefs.destination}</h2>
    <table>
      <tr><td>Destination</td><td>${prefs.destination}</td></tr>
      <tr><td>Dates</td><td>${prefs.start_date} → ${prefs.end_date}</td></tr>
      <tr><td>Travelers</td><td>${prefs.travelers} (${prefs.travel_type})</td></tr>
      <tr><td>Budget</td><td>₹${(prefs.budget||0).toLocaleString()} | Est. ₹${(budget.total_estimate||0).toLocaleString()}</td></tr>
      <tr><td>Transport</td><td>${transport.final_mode || '—'}</td></tr>
      <tr><td>Hotel</td><td>${(hotel.recommended||{}).name || '—'}</td></tr>
      <tr><td>Weather</td><td>${weather.condition || '—'} (score ${weather.score||'?'}/10)</td></tr>
      <tr><td>Plan quality</td><td>${review.score||'?'}/10 — ${review.overall_verdict||''}</td></tr>
    </table>
    ${(itin.days||[]).map(day => `
      <div class="day-block">
        <h3>Day ${day.day} · ${day.date} — ${day.theme}</h3>
        <ul>${(day.activities||[]).map(a => `<li>${a.time||''} ${a.activity} @ ${a.location}</li>`).join('')}</ul>
      </div>`).join('')}
  `;
  chat.insertBefore(card, spinner);
  chat.scrollTop = chat.scrollHeight;
  addSys('🎉 Trip plan ready! Scroll up to review.');
}

send.addEventListener('click', sendMessage);
msg.addEventListener('keydown', e => { if (e.key === 'Enter') sendMessage(); });

startSession();
</script>
</body>
</html>"""


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
    import time, json, tempfile
    import mlflow

    logger = logging.getLogger(session_id)
    mlflow.set_experiment("trip-planner-v3")

    with mlflow.start_run(run_name=session_id):
        try:
            _trip_results[session_id]["status"] = "extracting"
            prefs_state = chat.extract()

            _trip_results[session_id]["status"] = "planning"

            from main import build_graph
            from memory.memory_store import save as save_memory

            os.makedirs("logs", exist_ok=True)
            os.makedirs("checkpoints", exist_ok=True)
            fh = logging.FileHandler(f"logs/session_{session_id}.log", encoding="utf-8")
            fh.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)-5s %(message)s", "%H:%M:%S"))
            logger.addHandler(fh)
            logger.setLevel(logging.INFO)

            # ── Log input params ──────────────────────────────────────────────
            prefs = prefs_state.get("trip_preferences", {})
            mlflow.log_params({
                "session_id":    session_id,
                "source":        prefs.get("source", ""),
                "destination":   prefs.get("destination", ""),
                "start_date":    prefs.get("start_date", ""),
                "end_date":      prefs.get("end_date", ""),
                "travelers":     prefs.get("travelers", 0),
                "travel_type":   prefs.get("travel_type", ""),
                "hotel_pref":    prefs.get("hotel_pref", ""),
                "transport_pref": prefs.get("transport_pref", ""),
                "budget_inr":    prefs.get("budget", 0),
            })

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

            t0 = time.time()
            final_state = graph.invoke(initial_state, config=config)
            duration_s  = round(time.time() - t0, 1)

            # ── Log output metrics ────────────────────────────────────────────
            review  = final_state.get("review_status", {})
            budget  = final_state.get("budget_summary", {})
            evals   = final_state.get("eval_scores", {})
            weather = final_state.get("weather_data", {})

            mlflow.log_metrics({
                "planning_duration_s":    duration_s,
                "review_score":           float(review.get("score") or 0),
                "budget_inr":             float(prefs.get("budget") or 0),
                "budget_estimate_inr":    float(budget.get("total_estimate") or 0),
                "budget_surplus_inr":     float(budget.get("surplus_deficit") or 0),
                "weather_score":          float(weather.get("score") or 0),
                "hitl_change_count":      float(final_state.get("hitl_change_count") or 0),
                "retry_count":            float(final_state.get("retry_count") or 0),
                "pacing_score":           float(evals.get("pacing") or 0),
                "variety_score":          float(evals.get("variety") or 0),
                "preference_alignment":   float(evals.get("preference_alignment") or 0),
                "distance_vs_budget":     float(evals.get("distance_vs_budget") or 0),
                "days_planned":           float(len(final_state.get("itinerary", {}).get("days", []))),
                "cache_hits_geocode":     float(final_state.get("cache_hits", {}).get("geocode", False)),
                "cache_hits_weather":     float(final_state.get("cache_hits", {}).get("weather", False)),
            })

            mlflow.set_tags({
                "verdict":        review.get("overall_verdict", ""),
                "transport_mode": final_state.get("transport_data", {}).get("final_mode", ""),
                "hotel_name":     (final_state.get("hotel_data", {}).get("recommended") or {}).get("name", ""),
                "within_budget":  str(budget.get("within_budget", "")),
                "failed_agents":  ",".join(final_state.get("failed_agents", [])),
            })

            # ── Log itinerary as artifact ─────────────────────────────────────
            itin = final_state.get("itinerary", {})
            with tempfile.NamedTemporaryFile(mode="w", suffix=".json",
                                             delete=False, encoding="utf-8") as f:
                json.dump(itin, f, indent=2)
                tmp_path = f.name
            mlflow.log_artifact(tmp_path, artifact_path="itinerary")
            os.unlink(tmp_path)

            # ── Persist to user memory ────────────────────────────────────────
            user_id = final_state.get("user_profile", {}).get("user_id", "guest")
            if user_id and prefs.get("destination"):
                try:
                    save_memory(user_id, prefs, itin)
                except Exception as mem_err:
                    logger.warning(f"Memory save failed: {mem_err}")

            _trip_results[session_id] = {
                "status":                "complete",
                "session_id":            session_id,
                "trip_preferences":      final_state.get("trip_preferences", {}),
                "itinerary":             itin,
                "budget_summary":        budget,
                "review_status":         review,
                "weather_data":          weather,
                "transport_data":        final_state.get("transport_data", {}),
                "hotel_data":            final_state.get("hotel_data", {}),
                "coordinates":           final_state.get("coordinates", {}),
                "pdf_path":              final_state.get("pdf_path"),
                "whatsapp_summary_path": final_state.get("whatsapp_summary_path"),
                "mlflow_run_id":         mlflow.active_run().info.run_id,
            }
            logger.info(f"SESSION COMPLETE  duration={duration_s}s  review={review.get('score')}/10")

        except Exception as exc:
            mlflow.set_tag("error", str(exc))
            logger.error(f"Planning failed: {exc}", exc_info=True)
            _trip_results[session_id] = {"status": "failed", "error": str(exc)}


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
