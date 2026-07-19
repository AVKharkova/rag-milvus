"""In-memory ingest task registry for background Admin API jobs."""

from __future__ import annotations

import threading
from datetime import datetime, timezone
from typing import Any, Optional

_lock = threading.Lock()
_TASKS: dict[str, dict[str, Any]] = {}
_MAX_TASKS = 200


def _trim_locked() -> None:
    if len(_TASKS) <= _MAX_TASKS:
        return
    # Drop oldest by updated_at
    ordered = sorted(
        _TASKS.items(),
        key=lambda kv: kv[1].get("updated_at") or "",
    )
    for task_id, _ in ordered[: len(_TASKS) - _MAX_TASKS]:
        _TASKS.pop(task_id, None)


def create_task(task_id: str, *, title: str, org_id: str) -> dict[str, Any]:
    now = datetime.now(timezone.utc).isoformat()
    record = {
        "task_id": task_id,
        "status": "queued",
        "title": title,
        "org_id": org_id,
        "created_at": now,
        "updated_at": now,
        "result": None,
        "error": None,
    }
    with _lock:
        _TASKS[task_id] = record
        _trim_locked()
    return dict(record)


def set_task_running(task_id: str) -> None:
    with _lock:
        task = _TASKS.get(task_id)
        if not task:
            return
        task["status"] = "running"
        task["updated_at"] = datetime.now(timezone.utc).isoformat()


def set_task_done(task_id: str, result: dict[str, Any]) -> None:
    status = result.get("status") or "ok"
    with _lock:
        task = _TASKS.get(task_id)
        if not task:
            return
        task["status"] = status
        task["result"] = result
        task["error"] = (result.get("postgres") or {}).get("error")
        task["updated_at"] = datetime.now(timezone.utc).isoformat()


def set_task_error(task_id: str, error: str) -> None:
    with _lock:
        task = _TASKS.get(task_id)
        if not task:
            return
        task["status"] = "error"
        task["error"] = error
        task["updated_at"] = datetime.now(timezone.utc).isoformat()


def get_task(task_id: str) -> Optional[dict[str, Any]]:
    with _lock:
        task = _TASKS.get(task_id)
        return dict(task) if task else None
