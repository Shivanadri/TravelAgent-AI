from rich.console import Console
from rich.table import Table
from rich.live import Live
from rich.panel import Panel
from rich.text import Text
from datetime import datetime
import time

console = Console()

# Agent rows in display order
AGENTS = [
    ("user_input",   "🗣  User Input Agent"),
    ("memory",       "🧠  Memory Agent"),
    ("geocoding",    "🌐  Geocoding"),
    ("weather",      "🌤  Weather Agent"),
    ("transport",    "✈   Transport Agent"),
    ("hotel",        "🏨  Hotel Agent"),
    ("places",       "📍  Places Agent"),
    ("budget",       "💰  Budget Agent"),
    ("itinerary",    "📅  Itinerary Agent"),
    ("review",       "🔍  Review Agent"),
    ("orchestrator", "🎛  Orchestrator"),
    ("pdf",          "📄  PDF Generator"),
]

_status: dict[str, dict] = {
    key: {"status": "waiting", "source": "—", "elapsed": "—"}
    for key, _ in AGENTS
}
_start_times: dict[str, float] = {}
_live: Live | None = None
_trip_title: str = ""


def _build_table() -> Table:
    table = Table(box=None, show_header=True, header_style="bold cyan", padding=(0, 1))
    table.add_column("Agent",   style="white",   min_width=26)
    table.add_column("Status",  style="white",   min_width=12)
    table.add_column("Source",  style="dim",     min_width=18)
    table.add_column("Time",    style="dim",     min_width=8)

    status_style = {
        "waiting":    ("⏳ waiting",  "dim"),
        "running":    ("⟳  running",  "bold yellow"),
        "done":       ("✓  done",     "bold green"),
        "failed":     ("✗  failed",   "bold red"),
        "fallback":   ("↩  fallback", "bold orange3"),
        "skipped":    ("—  skipped",  "dim"),
    }

    for key, label in AGENTS:
        s = _status[key]
        raw_status = s["status"]
        display, style = status_style.get(raw_status, (raw_status, "white"))
        table.add_row(
            label,
            Text(display, style=style),
            s["source"],
            s["elapsed"],
        )
    return table


def start_display(trip_title: str = "Trip Planner v3") -> None:
    """Start the live status panel. Call once at the beginning of the run."""
    global _live, _trip_title
    _trip_title = trip_title
    panel = Panel(_build_table(), title=f"[bold cyan]{trip_title}[/]", border_style="blue")
    _live = Live(panel, console=console, refresh_per_second=4)
    _live.start()


def update_status(agent_key: str, status: str, source: str = "—") -> None:
    """
    Update one agent row.
    status: "running" | "done" | "failed" | "fallback" | "skipped"
    source: "live" | "cache" | "llm_fallback" | "—"
    """
    now = time.time()

    if status == "running":
        _start_times[agent_key] = now
        elapsed = "—"
    else:
        start = _start_times.get(agent_key, now)
        elapsed = f"{now - start:.1f}s"

    _status[agent_key] = {"status": status, "source": source, "elapsed": elapsed}

    if _live:
        panel = Panel(_build_table(), title=f"[bold cyan]{_trip_title}[/]", border_style="blue")
        _live.update(panel)


def stop_display() -> None:
    """Stop the live panel after all agents finish."""
    if _live:
        _live.stop()


def pause_display() -> None:
    """Temporarily stop the live panel before interactive input (HITL gates)."""
    if _live and _live.is_started:
        _live.stop()


def resume_display() -> None:
    """Restart the live panel after an interactive HITL gate completes."""
    global _live, _trip_title
    if _live and not _live.is_started:
        panel = Panel(_build_table(), title=f"[bold cyan]{_trip_title}[/]", border_style="blue")
        _live.update(panel)
        _live.start()
