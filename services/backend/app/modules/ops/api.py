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
LOGS_DIR       = REPO_ROOT / "logs"
NIBI_ALIAS     = "nibi"
NIBI_USER      = os.getenv("NIBI_USER", "harshsaw")
NIBI_SIM_DIR   = os.getenv("NIBI_SIM_DIR", "/home/harshsaw/projects/def-youry/test_simulation")

# Read-only commands allowed through the NIBI exec endpoint.
# Prefix-matched — e.g. "squeue" allows "squeue -u harshsaw -h -o '%T'".
_NIBI_ALLOWED_PREFIXES = (
    "squeue", "sacct", "sinfo", "scontrol show job",
    "cat ", "tail ", "ls ", "find ", "echo ", "hostname",
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
    """Check if the SSH ControlMaster socket is active."""
    import shutil
    if not shutil.which("ssh"):
        return False
    try:
        r = subprocess.run(
            ["ssh", "-O", "check", NIBI_ALIAS],
            capture_output=True, text=True, timeout=5,
        )
        return r.returncode == 0
    except FileNotFoundError:
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
                info["n_estimators"]   = m.get("n_estimators") or m.get("base_trees")
                info["train_end_date"] = m.get("train_end_date") or m.get("cutoff")
                info["model_type"]     = m.get("model_type")
                info["warm_trees_per_step"] = m.get("warm_trees_per_step")
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

    result: dict[str, Any] = {"available": True}

    # Active / pending jobs
    rc, out, _ = _ssh(
        "squeue -u harshsaw --format='%.10i %.20j %.8T %.10M %.9l %S' --noheader 2>/dev/null",
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
        "sacct -u harshsaw --starttime=now-7days "
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
    rc3, out3, _ = _ssh("quota -s 2>/dev/null || df -h /scratch/harshsaw 2>/dev/null | tail -1", timeout=8)
    result["quota_raw"] = out3[:300] if rc3 == 0 else None

    return result


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

    # 2. NIBI job record
    job = _nibi_job_info()

    # 3. Live squeue check (only if socket alive and a job is known)
    nibi_live_state: str | None = None
    if ssh_alive and job.get("job_id"):
        rc, sq, _ = _ssh(
            f"squeue -j {job['job_id']} -h -o '%T' 2>/dev/null || "
            f"sacct -j {job['job_id']} --format=State --noheader 2>/dev/null | head -1",
            timeout=12,
        )
        if rc == 0 and sq.strip():
            nibi_live_state = sq.strip().split()[0]

    # 4. Active model
    model = _active_model_info()

    # 5. NIBI GPU (only if a job is RUNNING) + live queue/history
    nibi_gpu  = _nibi_gpu_info(ssh_alive) if nibi_live_state == "RUNNING" else None
    nibi_jobs = _nibi_jobs_info(ssh_alive)

    # 6. Data freshness
    freshness: dict[str, Any] = {}
    try:
        row = db.execute(text(
            "SELECT MAX(window_ts) AS last_ts, COUNT(*) AS total_rows "
            "FROM ml.market_data_15m"
        )).mappings().one()
        last_ts = row["last_ts"]
        freshness = {
            "last_window_ts": last_ts.isoformat() if last_ts else None,
            "total_rows":     int(row["total_rows"]),
            "staleness_min":  round((now - last_ts.replace(tzinfo=dt.timezone.utc)).total_seconds() / 60, 1)
                              if last_ts else None,
        }
    except Exception as exc:
        freshness = {"error": str(exc)}

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
            collector = {
                "last_stage":  r["pipeline_stage"],
                "last_status": r["status"],
                "last_run_at": last_run.isoformat(),
                "age_min":     round((now - last_run).total_seconds() / 60, 1),
                "message":     r["message"],
            }
    except Exception as exc:
        collector = {"error": str(exc)}

    return {
        "generated_at": now.isoformat(),
        "ssh_socket":   {"alive": ssh_alive},
        "nibi_job":     {**job, "live_state": nibi_live_state},
        "nibi_jobs":    nibi_jobs,
        "model":        model,
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
        last_ts = row["last_ts"]
        return {
            "last_window_ts": last_ts.isoformat() if last_ts else None,
            "total_rows":     int(row["total_rows"]),
            "symbols":        int(row["symbols"]),
            "staleness_min":  round((now - last_ts.replace(tzinfo=dt.timezone.utc)).total_seconds() / 60, 1)
                              if last_ts else None,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
