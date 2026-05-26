from langgraph.checkpoint.memory import MemorySaver

# Single shared instance — lives in memory for the duration of the process.
# If the process restarts, checkpoints are lost (no external DB needed).
_saver = MemorySaver()


def get_checkpointer() -> MemorySaver:
    """Return the in-memory checkpointer (shared across all sessions)."""
    return _saver


def list_incomplete_sessions() -> list[dict]:
    """
    Return sessions stored in memory that have not yet completed.
    Only meaningful within the same process run.
    """
    try:
        threads = list(_saver.storage.keys())
        return [{"session_id": t} for t in threads]
    except Exception:
        return []


def delete_checkpoint(session_id: str) -> None:
    """Remove a session's checkpoint from memory after completion."""
    try:
        _saver.storage.pop(session_id, None)
    except Exception:
        pass
