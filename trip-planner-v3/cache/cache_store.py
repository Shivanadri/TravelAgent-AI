import os
import json
import time

CACHE_DIR = "cache"


def _path(key: str) -> str:
    os.makedirs(CACHE_DIR, exist_ok=True)
    safe_key = key.replace(" ", "_").replace("/", "-")
    return os.path.join(CACHE_DIR, f"{safe_key}.json")


def get(key: str, ttl_seconds: int) -> dict | None:
    """Return cached data if it exists and is not expired, else None."""
    fpath = _path(key)
    if not os.path.exists(fpath):
        return None
    with open(fpath, "r", encoding="utf-8") as f:
        record = json.load(f)
    age = time.time() - record.get("saved_at", 0)
    if age > ttl_seconds:
        return None
    return record["data"]


def set(key: str, data: dict | list) -> None:
    """Save data to cache with current timestamp."""
    fpath = _path(key)
    record = {"saved_at": time.time(), "data": data}
    with open(fpath, "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)


# ── TTL constants (seconds) ────────────────────────────────────────────────────
TTL_GEOCODE = 7 * 24 * 3600   # 7 days  — cities don't move
TTL_WEATHER = 1 * 3600        # 1 hour  — forecast changes
TTL_PLACES  = 24 * 3600       # 24 hours — attractions are stable
