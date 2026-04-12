"""
backend/app/modules/training/api.py
Live training log monitor via Server-Sent Events (SSE).

Endpoints:
  POST /api/v1/training/start      — register new run, clears buffer
  POST /api/v1/training/log        — push a log line (called by train_model.py)
  GET  /api/v1/training/log/stream — SSE stream for the frontend
  GET  /api/v1/training/status     — current job summary
  POST /api/v1/training/clear      — wipe buffer
"""

import asyncio
import json
import time
from collections import deque
from typing import AsyncGenerator

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

router = APIRouter()

# ── In-memory store ────────────────────────────────────────────────────────────
MAX_LINES   = 2000
_log_buffer: deque[dict] = deque(maxlen=MAX_LINES)
_subscribers: list[asyncio.Queue] = []
_job_meta: dict = {
    "status": "idle",
    "job_id": None,
    "started_at": None,
    "run_date": None,
}


# ── Pydantic models ───────────────────────────────────────────────────────────
class LogLine(BaseModel):
    level: str   = "INFO"    # INFO | WARN | ERROR | SUCCESS
    message: str
    job_id: str  = ""
    run_date: str = ""
    step: str    = ""        # extract | feature_eng | hpo | train | sync | done
    progress: float = -1     # 0-100, or -1 = unknown


class JobStart(BaseModel):
    job_id: str
    run_date: str


# ── Helpers ───────────────────────────────────────────────────────────────────
def _now_utc() -> str:
    return time.strftime("%H:%M:%S", time.gmtime())


def _broadcast(entry: dict):
    for q in _subscribers:
        try:
            q.put_nowait(entry)
        except asyncio.QueueFull:
            pass


def _make_entry(line: LogLine) -> dict:
    return {
        "ts":       _now_utc(),
        "level":    line.level,
        "message":  line.message,
        "step":     line.step,
        "progress": line.progress,
        "job_id":   line.job_id  or _job_meta.get("job_id", ""),
        "run_date": line.run_date or _job_meta.get("run_date", ""),
    }


def _fmt_sse(entry: dict) -> str:
    return f"data: {json.dumps(entry)}\n\n"


# ── Routes ────────────────────────────────────────────────────────────────────
@router.post("/start")
def start_job(meta: JobStart):
    """Called at beginning of train_model.py to register the run."""
    global _job_meta
    _log_buffer.clear()
    _job_meta = {
        "status":     "running",
        "job_id":     meta.job_id,
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "run_date":   meta.run_date,
    }
    entry = _make_entry(LogLine(
        level="SUCCESS",
        message=f"Job started — run_date={meta.run_date}  job_id={meta.job_id}",
        step="init",
    ))
    _log_buffer.append(entry)
    _broadcast(entry)
    return {"ok": True}


@router.post("/log")
def push_log(line: LogLine):
    """Receive a single log line from train_model.py and broadcast to all SSE clients."""
    entry = _make_entry(line)
    _log_buffer.append(entry)
    _broadcast(entry)

    if line.level == "ERROR":
        _job_meta["status"] = "failed"
    elif line.step == "done":
        _job_meta["status"] = "completed"

    return {"ok": True}


@router.post("/clear")
def clear_log():
    _log_buffer.clear()
    _job_meta.update({"status": "idle", "job_id": None, "started_at": None, "run_date": None})
    return {"ok": True}


@router.get("/status")
def get_status():
    return {**_job_meta, "buffered_lines": len(_log_buffer)}


@router.get("/log/stream")
async def stream_logs():
    """
    SSE endpoint. Replays full buffer to reconnecting clients, then streams live.
    Connect with: const es = new EventSource('/api/v1/training/log/stream')
    """
    queue: asyncio.Queue = asyncio.Queue(maxsize=500)
    _subscribers.append(queue)

    async def generator() -> AsyncGenerator[str, None]:
        # Send existing backlog so late-joiners see full history
        for entry in list(_log_buffer):
            yield _fmt_sse(entry)
        # Stream live updates; heartbeat every 15s keeps connection alive
        try:
            while True:
                try:
                    entry = await asyncio.wait_for(queue.get(), timeout=15.0)
                    yield _fmt_sse(entry)
                except asyncio.TimeoutError:
                    yield ": heartbeat\n\n"
        finally:
            if queue in _subscribers:
                _subscribers.remove(queue)

    return StreamingResponse(
        generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control":     "no-cache",
            "X-Accel-Buffering": "no",  # tells nginx not to buffer SSE
        },
    )
