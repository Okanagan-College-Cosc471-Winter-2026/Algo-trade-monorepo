"""
backend/app/modules/ops/api.py

Ops & observability endpoints — single source of truth for the ops dashboard.

Endpoints:
  GET  /api/v1/ops/status          — full system snapshot (services + machine + NIBI + model + data)
  GET  /api/v1/ops/nibi/ssh        — check if ControlMaster socket is alive
  POST /api/v1/ops/nibi/exec       — execute a whitelisted read-only command on NIBI
  GET  /api/v1/ops/pipeline/logs   — last N rows from operation_logs.pipeline_logs
  GET  /api/v1/ops/data/freshness  — latest window_ts + row count from ml.market_data_15m

SSH note:
  Commands are sent over the existing ControlMaster socket opened by morning_login.sh.
  No MFA is needed as long as the socket is alive (~10h after morning login).
  If the socket is dead, /nibi/ssh returns {"alive": false} and exec returns 503.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import platform
import shlex
import subprocess
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import psutil
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.api.deps import get_db

router = APIRouter(prefix="/ops", tags=["ops"])

# ── Paths (match monorepo layout) ──────────────────────────────────────────
REPO_ROOT      = Path(os.getenv("REPO_ROOT", "/data/projects/Algo-trade-monorepo"))
# In Docker the host ./model_artifacts is mounted at /model_artifacts.
# Override with ARTIFACTS_DIR env var to work both inside and outside Docker.
ARTIFACTS_DIR  = Path(os.getenv("ARTIFACTS_DIR", "/model_artifacts"))
LOGS_DIR       = Path(os.getenv("LOG_DIR", "/app/logs"))
NIBI_ALIAS     = "nibi"
NIBI_USER      = os.getenv("NIBI_USER", "harshsaw")
NIBI_SIM_DIR   = os.getenv("NIBI_SIM_DIR", "/home/harshsaw/projects/def-youry/test_simulation")

# Read-only commands allowed through the NIBI exec endpoint.
# Prefix-matched — e.g. "squeue" allows "squeue -u harshsaw -h -o '%T'".
_NIBI_ALLOWED_PREFIXES = (
    "squeue", "sacct", "sinfo", "scontrol show job",
    "cat ", "tail ", "ls ", "echo ", "hostname",
    "du -sh", "quota", "nvidia-smi",
)


# ── Helpers ────────────────────────────────────────────────────────────────

def _ssh(cmd: str, timeout: int = 15) -> tuple[int, str, str]:
    """Run a command on NIBI over the existing ControlMaster socket."""
    import shutil
    if not shutil.which("ssh"):
        return 1, "", "ssh not available in this environment"
    try:
        r = subprocess.run(
            ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", NIBI_ALIAS, cmd],
            capture_output=True, text=True, timeout=timeout + 2,
        )
        return r.returncode, r.stdout.strip(), r.stderr.strip()
    except FileNotFoundError:
        return 1, "", "ssh not available in this environment"


def _socket_alive() -> bool:
    """
    Infer SSH ControlMaster liveness from two signals (ssh binary absent in container):
      1. A recent successful nibi_intraday_warmrefresh or nibi_daily_warm_refresh DAG run
         in the Airflow DB — if one succeeded within the last 35 min, SSH was alive then.
      2. The ssh_alive.json heartbeat file written by the Airflow DAG on each health-check pass.
    Falls back to False if neither signal is available.
    """
    # Signal 1: heartbeat file written by Airflow DAG
    heartbeat = LOGS_DIR / "ssh_alive.json"
    if heartbeat.exists():
        try:
            data = json.loads(heartbeat.read_text())
            ts = dt.datetime.fromisoformat(data.get("last_alive_utc", ""))
            if (dt.datetime.now(dt.timezone.utc) - ts).total_seconds() < 2400:  # 40 min
                return True
        except Exception:
            pass

    # Signal 2: recent successful Airflow run of either NIBI DAG
    try:
        import psycopg2
        conn = psycopg2.connect(
            host=os.getenv("POSTGRES_SERVER", "db"),
            port=int(os.getenv("POSTGRES_PORT", "5432")),
            dbname="airflow",
            user=os.getenv("POSTGRES_USER", "appuser"),
            password=os.getenv("POSTGRES_PASSWORD", ""),
            connect_timeout=3,
        )
        with conn.cursor() as cur:
            cur.execute("""
                SELECT end_date FROM dag_run
                WHERE dag_id IN ('nibi_intraday_warmrefresh','nibi_daily_warm_refresh')
                  AND state = 'success'
                  AND end_date >= NOW() - INTERVAL '35 minutes'
                ORDER BY end_date DESC LIMIT 1
            """)
            row = cur.fetchone()
        conn.close()
        return row is not None
    except Exception:
        return False


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text()) if path.exists() else {}
    except Exception:
        return {}


def _active_model_info() -> dict:
    """Read metadata from the currently active model symlink."""
    symlink = ARTIFACTS_DIR / "current_base"
    if not symlink.exists():
        return {"active": False, "path": None}

    target = symlink.resolve()
    info: dict[str, Any] = {
        "active": True,
        "path":   str(target),
        "name":   target.name,
    }

    # Read metadata.json from the bundle
    for meta_name in ("metadata.json", "meta.json"):
        meta_path = target / meta_name
        if meta_path.exists():
            try:
                m = json.loads(meta_path.read_text())
                # Support both old flat format and new nested hyperparameters format
                hp = m.get("hyperparameters", {})
                info["n_estimators"]   = (m.get("n_estimators") or m.get("base_trees")
                                          or hp.get("n_estimators"))
                info["train_end_date"] = (m.get("train_end_date") or m.get("cutoff")
                                          or m.get("effective_as_of_date")
                                          or m.get("load_end_date"))
                info["model_type"]     = m.get("model_type")
                info["warm_trees_per_step"] = m.get("warm_trees_per_step")
                info["promoted_at"]    = m.get("promoted_at") or m.get("last_promoted_at")
                info["model_id"]       = m.get("model_id")
            except Exception:
                pass
            break

    # SIMULATION_DONE sentinel
    done = target / "SIMULATION_DONE"
    if done.exists():
        info["simulation_done"] = True
        info["simulation_done_content"] = done.read_text().strip()
    else:
        info["simulation_done"] = False

    # simulation_progress.json  (window-level stats)
    prog = target / "simulation_progress.json"
    if not prog.exists():
        prog = ARTIFACTS_DIR / f"sim_{target.name.replace('sim_', '')}" / "simulation_progress.json"
    if prog.exists():
        p = _read_json(prog)
        steps = p.get("steps", [])
        info["windows_total"]     = 26
        info["windows_ok"]        = sum(1 for s in steps if s.get("status") == "ok")
        info["windows_error"]     = sum(1 for s in steps if str(s.get("status", "")).startswith("error"))
        info["windows_steps"]     = steps   # full detail for frontend

    # Symlink mtime = when it was last promoted
    try:
        info["promoted_at"] = dt.datetime.fromtimestamp(
            symlink.lstat().st_mtime, tz=dt.timezone.utc
        ).isoformat()
    except Exception:
        pass

    return info


def _nibi_job_info() -> dict:
    """Read the most recent nibi_job_*.json file."""
    files = sorted(LOGS_DIR.glob("nibi_job_*.json"), reverse=True)
    if not files:
        return {"found": False}
    rec = _read_json(files[0])
    rec["found"] = bool(rec)
    rec["log_file"] = files[0].name
    return rec


def _base_only_flow_info() -> dict:
    """Read the latest machine-readable base-only pipeline status, if present."""
    path = LOGS_DIR / "nibi_base_only_status.json"
    info = _read_json(path)
    if info:
        info["found"] = True
        info["status_file"] = path.name
        return info
    return {"found": False}


def _parse_iso_utc(value: Any) -> dt.datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        parsed = dt.datetime.fromisoformat(text.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=dt.timezone.utc)
        return parsed.astimezone(dt.timezone.utc)
    except Exception:
        return None


def _parse_date(value: Any) -> dt.date | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return dt.date.fromisoformat(text[:10])
    except Exception:
        return None


def _next_day(value: str | None) -> str | None:
    parsed = _parse_date(value)
    if not parsed:
        return None
    return (parsed + dt.timedelta(days=1)).isoformat()


def _resolve_nibi_job_card(
    file_job: dict[str, Any],
    flow: dict[str, Any],
    live_primary: dict[str, Any] | None,
) -> dict[str, Any]:
    """
    Pick the best job-card source:
    - Prefer live queue job when available.
    - Else prefer the freshest of flow-status vs legacy nibi_job_*.json.
    """
    if live_primary:
        return {
            "found": True,
            "source": "live_queue",
            "job_id": live_primary.get("job_id"),
            "name": live_primary.get("name"),
            "status": str(live_primary.get("state", "")).lower() or None,
            "live_state": str(live_primary.get("state", "")).upper() or None,
            "submitted_at": None,
            "sim_date": flow.get("sim_date") or file_job.get("sim_date"),
            "log_file": file_job.get("log_file"),
        }

    flow_ts = _parse_iso_utc(flow.get("updated_at_utc"))
    file_ts = _parse_iso_utc(file_job.get("submitted_at"))
    use_flow = bool(flow.get("found")) and (
        flow_ts is not None and (file_ts is None or flow_ts >= file_ts)
    )

    if use_flow:
        return {
            "found": True,
            "source": "base_only_flow",
            "job_id": flow.get("job_id"),
            "name": "sim_base_train",
            "status": flow.get("status"),
            "live_state": None,
            "submitted_at": flow.get("updated_at_utc"),
            "sim_date": flow.get("sim_date"),
            "log_file": flow.get("status_file"),
        }

    if file_job.get("found"):
        return {
            **file_job,
            "source": "nibi_job_file",
        }

    return {"found": False}


def _read_snapshot_meta(cutoff_date: str | None) -> dict:
    if not cutoff_date:
        return {}
    return _read_json(REPO_ROOT / "datasets" / f"snapshot_{cutoff_date}.meta.json")


def _job_kind(job: dict[str, Any] | None) -> str | None:
    if not job:
        return None
    name = str(job.get("name", "")).lower()
    if "sim_base_train" in name:
        return "base_train"
    if "simulate_full_day" in name or "warm" in name:
        return "warm_refresh"
    return None


def _normalize_stage_status(status: str | None) -> str:
    status = str(status or "").lower()
    if status == "ok":
        return "completed"
    if status in {"running", "error", "pending", "completed", "not_started"}:
        return status
    return "not_started"


def _build_training_flow(
    ssh_alive: bool,
    job: dict[str, Any],
    live_primary: dict[str, Any] | None,
    model: dict[str, Any],
    latest_data_date: str | None,
) -> dict[str, Any]:
    flow = _base_only_flow_info()
    active_cutoff = str(model.get("train_end_date") or "")
    cutoff_date = latest_data_date or flow.get("cutoff_date") or active_cutoff or None
    flow_matches_target = bool(flow.get("found")) and flow.get("cutoff_date") == cutoff_date
    if not flow_matches_target:
        flow = {"found": False}
    sim_date = flow.get("sim_date") or _next_day(cutoff_date)
    snapshot = flow.get("snapshot") or _read_snapshot_meta(cutoff_date)
    promoted_for_cutoff = bool(cutoff_date and active_cutoff == cutoff_date)

    live_state = str((live_primary or {}).get("state", "")).upper()
    live_kind = _job_kind(live_primary)
    flow_stage = str(flow.get("stage", ""))
    flow_status = _normalize_stage_status(flow.get("status"))
    flow_message = flow.get("message")
    def from_flow(stages: set[str]) -> str | None:
        return flow_status if flow.get("found") and flow_stage in stages else None

    snapshot_status = "completed" if snapshot.get("validation_ok") else "not_started"
    if flow_stage in {"export_snapshot", "validate_snapshot", "upload_snapshot"} and flow_status in {"running", "error"}:
        snapshot_status = flow_status

    base_train_status = "not_started"
    if live_kind == "base_train":
        base_train_status = "running" if live_state == "RUNNING" else "pending"
    elif promoted_for_cutoff or from_flow({"base_train", "verify_remote_artifacts", "download_artifacts", "promote_base", "reload_backend", "completed"}) == "completed":
        base_train_status = "completed"
    elif from_flow({"submit_base_train", "base_train"}) in {"running", "pending", "error"}:
        base_train_status = from_flow({"submit_base_train", "base_train"}) or "not_started"
    elif cutoff_date and active_cutoff and _parse_date(cutoff_date) and _parse_date(active_cutoff) and _parse_date(cutoff_date) > _parse_date(active_cutoff):
        base_train_status = "pending"
    elif cutoff_date and not active_cutoff:
        base_train_status = "pending"

    promote_status = "completed" if promoted_for_cutoff else "not_started"
    if flow_stage in {"download_artifacts", "promote_base", "reload_backend"} and flow_status in {"running", "error"}:
        promote_status = flow_status

    stages = [
        {
            "id": "ssh_socket",
            "label": "SSH Socket",
            "status": "completed" if ssh_alive else "error",
            "detail": "NIBI control socket available" if ssh_alive else "NIBI control socket unavailable",
        },
        {
            "id": "snapshot_validated",
            "label": "Snapshot Validated",
            "status": snapshot_status,
            "detail": (
                f"cutoff={cutoff_date} symbols={snapshot.get('cutoff_symbols')} "
                f"open={snapshot.get('open_bar_symbols')} close={snapshot.get('close_bar_symbols')}"
            ).strip(),
        },
        {
            "id": "base_train",
            "label": "Base Train",
            "status": base_train_status,
            "detail": (
                f"job={((live_primary or {}).get('job_id') or flow.get('job_id') or '—')} "
                f"state={live_state or flow.get('status') or 'scheduled'} "
                f"sim_date={sim_date or '—'}"
            ),
        },
        {
            "id": "promote_base",
            "label": "Promote Base",
            "status": promote_status,
            "detail": f"active_cutoff={active_cutoff or '—'}",
        },
    ]

    current_stage = flow_stage
    if not current_stage:
        if live_kind == "base_train":
            current_stage = "base_train"
        elif base_train_status == "pending":
            current_stage = "base_train"
        elif promoted_for_cutoff:
            current_stage = "promote_base"
        elif snapshot.get("validation_ok"):
            current_stage = "snapshot_validated"
        else:
            current_stage = "ssh_socket"

    return {
        "pipeline": flow.get("pipeline") or "derived",
        "cutoff_date": cutoff_date,
        "sim_date": sim_date,
        "current_stage": current_stage,
        "status": flow_status if flow.get("found") else "derived",
        "message": flow_message,
        "job_id": flow.get("job_id") or (live_primary or {}).get("job_id") or job.get("job_id"),
        "snapshot": snapshot,
        "status_file": flow.get("status_file"),
        "stages": stages,
    }


def _machine_info() -> dict:
    """Collect VPS hardware metrics via psutil."""
    cpu_pct      = psutil.cpu_percent(interval=0.5)
    cpu_per_core = psutil.cpu_percent(interval=0.1, percpu=True)
    mem          = psutil.virtual_memory()
    swap         = psutil.swap_memory()
    disk_root    = psutil.disk_usage("/")
    disk_data    = psutil.disk_usage("/data") if Path("/data").exists() else None
    net          = psutil.net_io_counters()
    boot_ts      = psutil.boot_time()
    uptime_hrs   = round((dt.datetime.now().timestamp() - boot_ts) / 3600, 1)

    try:
        load1, load5, load15 = os.getloadavg()
    except AttributeError:
        load1 = load5 = load15 = 0.0

    info: dict[str, Any] = {
        "hostname":      platform.node(),
        "os":            f"{platform.system()} {platform.release()}",
        "cpu_cores":     psutil.cpu_count(logical=True),
        "cpu_model":     _cpu_model(),
        "cpu_pct":       round(cpu_pct, 1),
        "cpu_per_core":  [round(p, 1) for p in (cpu_per_core or [])],
        "load_avg":      {"1m": round(load1, 2), "5m": round(load5, 2), "15m": round(load15, 2)},
        "uptime_hrs":    uptime_hrs,
        "process_count": len(psutil.pids()),
        "ram_total_gb":  round(mem.total / 1e9, 1),
        "ram_used_gb":   round(mem.used  / 1e9, 1),
        "ram_pct":       mem.percent,
        "swap_total_gb": round(swap.total / 1e9, 1),
        "swap_used_gb":  round(swap.used  / 1e9, 1),
        "swap_pct":      swap.percent,
        "net_sent_gb":   round(net.bytes_sent / 1e9, 2),
        "net_recv_gb":   round(net.bytes_recv / 1e9, 2),
        "net_pkts_sent": net.packets_sent,
        "net_pkts_recv": net.packets_recv,
        "disk": {
            "/": {
                "total_gb": round(disk_root.total / 1e9, 1),
                "used_gb":  round(disk_root.used  / 1e9, 1),
                "pct":      disk_root.percent,
            }
        },
        "gpu": None,    # No GPU on this VPS — training happens on NIBI H100
    }
    if disk_data:
        info["disk"]["/data"] = {
            "total_gb": round(disk_data.total / 1e9, 1),
            "used_gb":  round(disk_data.used  / 1e9, 1),
            "pct":      disk_data.percent,
        }
    return info


def _cpu_model() -> str:
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return platform.processor() or "unknown"


def _nibi_jobs_info(alive: bool) -> dict:
    """Fetch live squeue + recent sacct from NIBI when socket is alive."""
    if not alive:
        return {"available": False}

    result: dict[str, Any] = {"available": True, "user": NIBI_USER}

    # Active / pending jobs
    rc, out, _ = _ssh(
        f"squeue -u {NIBI_USER} --format='%.10i %.20j %.8T %.10M %.9l %S' --noheader 2>/dev/null",
        timeout=12,
    )
    queued = []
    if rc == 0 and out.strip():
        for line in out.strip().splitlines():
            parts = line.split()
            if len(parts) >= 5:
                queued.append({
                    "job_id":    parts[0],
                    "name":      parts[1],
                    "state":     parts[2],
                    "elapsed":   parts[3],
                    "time_lim":  parts[4],
                    "start":     parts[5] if len(parts) > 5 else "—",
                })
    result["queued"] = queued

    # Recent job history (last 10 jobs)
    rc2, out2, _ = _ssh(
        f"sacct -u {NIBI_USER} --starttime=now-7days "
        "--format='JobID,JobName,State,ExitCode,Elapsed,Start' "
        "--noheader --parsable2 2>/dev/null | grep -v '\\.' | head -15",
        timeout=15,
    )
    history = []
    if rc2 == 0 and out2.strip():
        for line in out2.strip().splitlines():
            parts = line.split("|")
            if len(parts) >= 6:
                history.append({
                    "job_id":   parts[0],
                    "name":     parts[1],
                    "state":    parts[2],
                    "exit":     parts[3],
                    "elapsed":  parts[4],
                    "start":    parts[5][:16].replace("T", " ") if parts[5] else "—",
                })
    result["history"] = history

    # Scratch quota
    rc3, out3, _ = _ssh(f"quota -s 2>/dev/null || df -h /scratch/{NIBI_USER} 2>/dev/null | tail -1", timeout=8)
    result["quota_raw"] = out3[:300] if rc3 == 0 else None

    return result


def _primary_live_job(nibi_jobs: dict) -> dict[str, Any] | None:
    """Pick a primary live job for top-card display: RUNNING first, then PENDING."""
    queued = nibi_jobs.get("queued", [])
    if not queued:
        return None
    for state in ("RUNNING", "PENDING"):
        for job in queued:
            if str(job.get("state", "")).upper() == state:
                return job
    return queued[0]


def _market_state(now_utc: dt.datetime) -> dict[str, Any]:
    """
    Lightweight market-hours state (NYSE regular session).
    Note: holiday-aware precision can be added later with calendar integration.
    """
    now_et = now_utc.astimezone(ZoneInfo("America/New_York"))
    is_weekday = now_et.weekday() < 5
    open_time = now_et.replace(hour=9, minute=30, second=0, microsecond=0)
    close_time = now_et.replace(hour=16, minute=0, second=0, microsecond=0)
    is_open = is_weekday and (open_time <= now_et <= close_time)
    return {
        "tz": "America/New_York",
        "now_et": now_et.isoformat(),
        "is_weekday": is_weekday,
        "is_open": is_open,
        "session_open_et": "09:30",
        "session_close_et": "16:00",
    }


def _compute_freshness(last_ts: dt.datetime | None, total_rows: int, now_utc: dt.datetime) -> dict[str, Any]:
    """Normalize freshness into explicit states to avoid off-hours false alarms."""
    market = _market_state(now_utc)
    if not last_ts:
        return {
            "last_window_ts": None,
            "total_rows": int(total_rows),
            "staleness_min": None,
            "freshness_state": "error",
            "freshness_reason": "No market_data_15m rows found",
            "market_state": market,
        }

    staleness_min = round((now_utc - last_ts.replace(tzinfo=dt.timezone.utc)).total_seconds() / 60, 1)
    if market["is_open"]:
        state = "fresh" if staleness_min < 20 else "stale"
        reason = "Within live-session freshness threshold" if state == "fresh" else "Live session data lag exceeds threshold"
    else:
        state = "expected_idle"
        reason = "Outside regular market session"

    return {
        "last_window_ts": last_ts.isoformat(),
        "total_rows": int(total_rows),
        "staleness_min": staleness_min,
        "freshness_state": state,
        "freshness_reason": reason,
        "market_state": market,
    }


def _nibi_gpu_info(alive: bool) -> dict | None:
    """Query nvidia-smi on the NIBI node (only if socket is alive and a job is running)."""
    if not alive:
        return None
    rc, out, _ = _ssh("nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu,temperature.gpu "
                      "--format=csv,noheader,nounits 2>/dev/null | head -1", timeout=10)
    if rc != 0 or not out.strip():
        return None
    try:
        parts = [p.strip() for p in out.split(",")]
        return {
            "name":       parts[0],
            "mem_total":  int(parts[1]),
            "mem_used":   int(parts[2]),
            "util_pct":   int(parts[3]),
            "temp_c":     int(parts[4]),
        }
    except Exception:
        return None


# ── Routes ─────────────────────────────────────────────────────────────────

@router.get("/status")
def get_ops_status(db: Session = Depends(get_db)) -> dict:
    """
    Full ops snapshot — called by the Streamlit dashboard on page load.
    Returns in one shot: services, VPS machine, NIBI job, active model, data freshness.
    """
    now = dt.datetime.now(dt.timezone.utc)

    # 1. SSH socket
    ssh_alive = _socket_alive()

    # 3. Active model
    model = _active_model_info()

    # 4. Live queue/history (authoritative for current job state)
    nibi_jobs = _nibi_jobs_info(ssh_alive)
    live_primary = _primary_live_job(nibi_jobs) if nibi_jobs.get("available") else None
    nibi_live_state = str((live_primary or {}).get("state", "")).upper() if live_primary else None

    # 5. NIBI GPU (only when at least one live job is RUNNING)
    any_running = any(str(j.get("state", "")).upper() == "RUNNING" for j in nibi_jobs.get("queued", []))
    nibi_gpu = _nibi_gpu_info(ssh_alive) if any_running else None

    # 6. Data freshness (computed first so latest_data_date is available for training flow)
    freshness: dict[str, Any] = {}
    latest_data_date: str | None = None
    try:
        row = db.execute(text(
            "SELECT MAX(window_ts) AS last_ts, COUNT(*) AS total_rows "
            "FROM ml.market_data_15m"
        )).mappings().one()
        freshness = _compute_freshness(row["last_ts"], int(row["total_rows"]), now)
        if row["last_ts"] is not None:
            latest_data_date = row["last_ts"].strftime("%Y-%m-%d")
    except Exception as exc:
        freshness = {"error": str(exc)}

    # 5b. End-to-end training flow snapshot + normalized job card
    job_file = _nibi_job_info()
    flow_info = _base_only_flow_info()
    training_flow = _build_training_flow(ssh_alive, job_file, live_primary, model, latest_data_date)
    job = _resolve_nibi_job_card(job_file, flow_info, live_primary)

    # 7. Collector last run (from operation_logs.pipeline_logs)
    collector: dict[str, Any] = {}
    try:
        rows = db.execute(text(
            "SELECT pipeline_stage, status, created_at, message "
            "FROM operation_logs.pipeline_logs "
            "ORDER BY created_at DESC LIMIT 1"
        )).mappings().all()
        if rows:
            r = rows[0]
            last_run = r["created_at"].replace(tzinfo=dt.timezone.utc)
            market = _market_state(now)
            age_min = round((now - last_run).total_seconds() / 60, 1)
            if market["is_open"]:
                collector_state = "healthy" if r["status"] == "success" and age_min < 30 else "stale_or_error"
                collector_reason = "Collector updates within live-session threshold" if collector_state == "healthy" else "Collector lag/error during market hours"
            else:
                collector_state = "expected_idle"
                collector_reason = "Outside regular market session"
            collector = {
                "last_stage":  r["pipeline_stage"],
                "last_status": r["status"],
                "last_run_at": last_run.isoformat(),
                "age_min":     age_min,
                "message":     r["message"],
                "collector_state": collector_state,
                "collector_reason": collector_reason,
            }
    except Exception as exc:
        collector = {"error": str(exc)}

    return {
        "generated_at": now.isoformat(),
        "nibi_user":    NIBI_USER,
        "ssh_socket":   {"alive": ssh_alive},
        "nibi_job":     {**job, "live_state": nibi_live_state or job.get("live_state")},
        "live_job_primary": live_primary,
        "nibi_jobs":    nibi_jobs,
        "model":        model,
        "training_flow": training_flow,
        "machine":      _machine_info(),
        "nibi_gpu":     nibi_gpu,
        "data":         freshness,
        "collector":    collector,
    }


@router.get("/nibi/ssh")
def nibi_ssh_status() -> dict:
    """Quick check: is the ControlMaster socket alive?"""
    alive = _socket_alive()
    msg = "Socket active — automated commands will work without MFA." if alive \
          else "Socket dead — run  bash ml/ml/nibi/morning_login.sh  to re-authenticate."
    return {"alive": alive, "message": msg}


class NibiExecRequest(BaseModel):
    command: str


@router.post("/nibi/exec")
def nibi_exec(req: NibiExecRequest) -> dict:
    """
    Execute a read-only command on NIBI via the ControlMaster socket.
    Only whitelisted command prefixes are allowed (squeue, sacct, tail logs, etc.).
    Returns stdout + stderr + exit code.
    """
    if not _socket_alive():
        raise HTTPException(
            status_code=503,
            detail="SSH ControlMaster socket is not alive. "
                   "Run morning_login.sh to authenticate.",
        )

    cmd = req.command.strip()
    allowed = any(cmd.startswith(p) for p in _NIBI_ALLOWED_PREFIXES)
    if not allowed:
        raise HTTPException(
            status_code=403,
            detail=f"Command not in whitelist. Allowed prefixes: {list(_NIBI_ALLOWED_PREFIXES)}",
        )

    rc, stdout, stderr = _ssh(cmd, timeout=20)
    return {
        "command":  cmd,
        "rc":       rc,
        "stdout":   stdout[:4000],   # cap to avoid huge payloads
        "stderr":   stderr[:1000],
    }


@router.get("/pipeline/logs")
def get_pipeline_logs(limit: int = 50, db: Session = Depends(get_db)) -> list[dict]:
    """Last N rows from operation_logs.pipeline_logs — collector pipeline history."""
    try:
        rows = db.execute(text(
            "SELECT pipeline_stage, status, created_at, message "
            "FROM operation_logs.pipeline_logs "
            "ORDER BY created_at DESC "
            "LIMIT :lim"
        ), {"lim": min(limit, 200)}).mappings().all()
        return [
            {
                "stage":      r["pipeline_stage"],
                "status":     r["status"],
                "ts":         r["created_at"].isoformat(),
                "message":    r["message"],
            }
            for r in rows
        ]
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/nibi/relogin")
def nibi_relogin() -> dict:
    """
    Trigger auto_login.py to re-establish the SSH ControlMaster socket.

    Behaviour depends on env vars (set in .env / docker-compose):
      - NIBI_TOTP_SECRET set → fully headless (no phone, TOTP code sent automatically)
      - NIBI_TOTP_SECRET absent → sends Duo push; user must approve on their phone

    The script runs in the background (non-blocking). Poll /ops/nibi/ssh to
    check when the socket comes alive.

    Returns immediately with {"started": true, "mode": "totp"|"duo_push"}.
    """
    if _socket_alive():
        return {"started": False, "already_alive": True, "message": "Socket already active."}

    password = os.getenv("NIBI_PASSWORD", "")
    if not password:
        raise HTTPException(
            status_code=503,
            detail="NIBI_PASSWORD not set in environment. "
                   "Add it to .env and restart the backend container.",
        )

    totp_secret = os.getenv("NIBI_TOTP_SECRET", "")
    mode = "totp" if totp_secret else "duo_push"

    script = REPO_ROOT / "ml" / "ml" / "nibi" / "auto_login.py"
    if not script.exists():
        raise HTTPException(status_code=500, detail=f"auto_login.py not found at {script}")

    env = {
        **os.environ,
        "NIBI_PASSWORD":    password,
        "NIBI_TOTP_SECRET": totp_secret,
        "NIBI_SSH_ALIAS":   NIBI_ALIAS,
        "NIBI_TIMEOUT":     "90",
    }

    # Run detached — caller polls /ops/nibi/ssh to know when it succeeds
    subprocess.Popen(
        ["python3", str(script)],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    return {
        "started": True,
        "mode":    mode,
        "message": "Duo push sent — approve on your phone, then poll /ops/nibi/ssh."
                   if mode == "duo_push" else
                   "TOTP login started — socket should be active within ~10s.",
    }


def _airflow_status() -> dict:
    """Query Airflow metadata DB for DAG statuses and recent run history."""
    try:
        import psycopg
    except ImportError:
        return {"error": "psycopg not available", "dags": [], "recent_runs": []}

    host   = os.getenv("POSTGRES_SERVER", "db")
    port   = int(os.getenv("POSTGRES_PORT", "5432"))
    user   = os.getenv("POSTGRES_USER", "appuser")
    passwd = os.getenv("POSTGRES_PASSWORD", "changeme")

    try:
        conn = psycopg.connect(
            host=host, port=port, dbname="airflow",
            user=user, password=passwd,
            connect_timeout=5,
        )
    except Exception as exc:
        return {"error": str(exc), "dags": [], "recent_runs": []}

    try:
        with conn:
            with conn.cursor() as cur:
                # DAG list with latest run state
                cur.execute("""
                    SELECT
                        d.dag_id,
                        d.is_paused,
                        d.is_active,
                        d.schedule_interval,
                        d.next_dagrun,
                        dr.state        AS last_state,
                        dr.run_type     AS last_run_type,
                        dr.start_date   AS last_start,
                        dr.end_date     AS last_end,
                        dr.run_id       AS last_run_id
                    FROM dag d
                    LEFT JOIN LATERAL (
                        SELECT state, run_type, start_date, end_date, run_id
                        FROM dag_run
                        WHERE dag_id = d.dag_id
                        ORDER BY execution_date DESC
                        LIMIT 1
                    ) dr ON true
                    ORDER BY d.dag_id
                """)
                cols = [desc[0] for desc in cur.description]
                dags = []
                for row in cur.fetchall():
                    r = dict(zip(cols, row))
                    dags.append({
                        "dag_id":        r["dag_id"],
                        "is_paused":     r["is_paused"],
                        "is_active":     r["is_active"],
                        "schedule":      r["schedule_interval"],
                        "next_run":      r["next_dagrun"].isoformat() if r["next_dagrun"] else None,
                        "last_state":    r["last_state"],
                        "last_run_type": r["last_run_type"],
                        "last_start":    r["last_start"].isoformat() if r["last_start"] else None,
                        "last_end":      r["last_end"].isoformat() if r["last_end"] else None,
                        "last_run_id":   r["last_run_id"],
                    })

                # Recent runs (last 20 across all DAGs)
                cur.execute("""
                    SELECT dag_id, run_id, state, run_type,
                           execution_date, start_date, end_date
                    FROM dag_run
                    ORDER BY execution_date DESC
                    LIMIT 20
                """)
                cols2 = [desc[0] for desc in cur.description]
                recent_runs = []
                for row in cur.fetchall():
                    r = dict(zip(cols2, row))
                    duration = None
                    if r["start_date"] and r["end_date"]:
                        duration = round((r["end_date"] - r["start_date"]).total_seconds())
                    recent_runs.append({
                        "dag_id":     r["dag_id"],
                        "run_id":     r["run_id"],
                        "state":      r["state"],
                        "run_type":   r["run_type"],
                        "started":    r["start_date"].isoformat() if r["start_date"] else None,
                        "ended":      r["end_date"].isoformat() if r["end_date"] else None,
                        "duration_s": duration,
                    })

        return {"error": None, "dags": dags, "recent_runs": recent_runs}
    except Exception as exc:
        return {"error": str(exc), "dags": [], "recent_runs": []}
    finally:
        conn.close()


@router.get("/airflow")
def get_airflow_status() -> dict:
    """DAG statuses and recent run history from the Airflow metadata database."""
    return _airflow_status()


@router.get("/logs/{log_name}")
def get_log_tail(log_name: str, lines: int = 80) -> dict:
    """
    Return the last N lines of a known log file.
    log_name: pipeline_15m | warm_refresh | nibi_usage
    """
    allowed = {
        "pipeline_15m":  LOGS_DIR / "pipeline_15m.log",
        "warm_refresh":  LOGS_DIR / "nibi_warm_refresh.log",
        "nibi_usage":    LOGS_DIR / "nibi_usage_meter.jsonl",
    }
    path = allowed.get(log_name)
    if path is None:
        raise HTTPException(status_code=404, detail=f"Unknown log: {log_name}")
    if not path.exists():
        return {"log_name": log_name, "lines": [], "exists": False}
    try:
        # Efficient tail without reading whole file
        with open(path, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            chunk = min(size, lines * 200)
            f.seek(max(0, size - chunk))
            raw = f.read().decode("utf-8", errors="replace")
        all_lines = raw.splitlines()
        tail = all_lines[-lines:] if len(all_lines) > lines else all_lines
        return {"log_name": log_name, "lines": tail, "exists": True, "path": str(path)}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/data/freshness")
def get_data_freshness(db: Session = Depends(get_db)) -> dict:
    """Latest window_ts and row count from ml.market_data_15m."""
    now = dt.datetime.now(dt.timezone.utc)
    try:
        row = db.execute(text(
            "SELECT MAX(window_ts) AS last_ts, COUNT(*) AS total_rows, "
            "COUNT(DISTINCT symbol) AS symbols "
            "FROM ml.market_data_15m"
        )).mappings().one()
        payload = _compute_freshness(row["last_ts"], int(row["total_rows"]), now)
        payload["symbols"] = int(row["symbols"])
        return payload
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
