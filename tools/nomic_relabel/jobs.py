"""In-process background jobs with polled progress.

Why this exists: rote.isd.ngo sits behind Cloudflare, which cuts off any origin
response that takes longer than ~100 seconds (error 524). Relabelling a large
dataset takes minutes, so the work cannot happen inside the request that starts
it. The browser kicks off a job, gets an id back immediately, and polls.

State is a module-level dict, which is correct here precisely because ROTE runs
as a single process — requirements.txt ships no WSGI server, so the start
command is `python app.py` and Flask's built-in server handles every request in
one process. If a gunicorn with `-w 2` is ever introduced, a status poll could
land on a worker that has never heard of the job and would appear to hang at 0%
forever. Guard against that by keeping ROTE single-process, or move this state
to Redis/disk.
"""

from __future__ import annotations

import threading
import time
import traceback
import uuid
from typing import Any, Callable, Dict, Optional

# Jobs are held only long enough for the browser to collect the result.
JOB_TTL_SECONDS = 30 * 60
MAX_JOBS = 20

_JOBS: Dict[str, Dict[str, Any]] = {}
_LOCK = threading.Lock()


def _prune() -> None:
    """Drop finished jobs past their TTL, and the oldest if we're over budget.

    Results can hold a workbook worth of bytes, so this is memory management,
    not just tidiness.
    """
    now = time.time()
    with _LOCK:
        stale = [k for k, j in _JOBS.items()
                 if j["state"] in ("done", "error") and now - j["updated"] > JOB_TTL_SECONDS]
        for k in stale:
            _JOBS.pop(k, None)
        if len(_JOBS) > MAX_JOBS:
            for k, _ in sorted(_JOBS.items(), key=lambda kv: kv[1]["updated"])[:len(_JOBS) - MAX_JOBS]:
                if _JOBS[k]["state"] in ("done", "error"):
                    _JOBS.pop(k, None)


def start(fn: Callable[[Callable[[float, str], None]], Any], label: str = "") -> str:
    """Run fn(progress) on a daemon thread. Returns a job id."""
    _prune()
    job_id = uuid.uuid4().hex[:12]
    with _LOCK:
        _JOBS[job_id] = {
            "state": "running", "pct": 0.0, "msg": "starting…",
            "label": label, "result": None, "error": None,
            "created": time.time(), "updated": time.time(),
        }

    def progress(frac: float, msg: str) -> None:
        with _LOCK:
            job = _JOBS.get(job_id)
            if job is not None:
                job["pct"] = round(max(0.0, min(1.0, float(frac))) * 100, 1)
                job["msg"] = str(msg)[:300]
                job["updated"] = time.time()

    def run() -> None:
        try:
            result = fn(progress)
            with _LOCK:
                job = _JOBS.get(job_id)
                if job is not None:
                    job.update(state="done", pct=100.0, msg="done",
                               result=result, updated=time.time())
        except Exception as e:  # surfaced to the browser, not swallowed
            traceback.print_exc()
            with _LOCK:
                job = _JOBS.get(job_id)
                if job is not None:
                    job.update(state="error", error=f"{type(e).__name__}: {e}",
                               updated=time.time())

    threading.Thread(target=run, name=f"nomic_relabel:{job_id}", daemon=True).start()
    return job_id


def get(job_id: str) -> Optional[Dict[str, Any]]:
    with _LOCK:
        job = _JOBS.get(job_id)
        return dict(job) if job else None


def public_status(job_id: str) -> Optional[Dict[str, Any]]:
    """Status safe to serialise to the browser — omits result payloads, which
    may be megabytes of xlsx."""
    job = get(job_id)
    if not job:
        return None
    out = {k: job[k] for k in ("state", "pct", "msg", "label", "error")}
    result = job.get("result")
    if job["state"] == "done" and isinstance(result, dict):
        out["summary"] = {k: v for k, v in result.items() if k != "bytes"}
        out["has_download"] = "bytes" in result
    return out
