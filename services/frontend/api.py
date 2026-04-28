from __future__ import annotations

import io
import os
from typing import Any

import requests

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000/api/v1").rstrip("/")
REQUEST_TIMEOUT = float(os.getenv("API_TIMEOUT_SECONDS", "30"))


class ApiError(RuntimeError):
    pass


def _request(method: str, path: str, **kwargs: Any) -> Any:
    url = f"{API_BASE_URL}{path}"
    response = requests.request(method, url, timeout=REQUEST_TIMEOUT, **kwargs)
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        detail = ""
        try:
            payload = response.json()
            detail = payload.get("detail", payload)
        except ValueError:
            detail = response.text
        raise ApiError(f"{method} {path} failed: {detail}") from exc

    if not response.content:
        return None

    content_type = response.headers.get("content-type", "")
    if "application/json" in content_type:
        return response.json()

    return response.content


def health_check() -> bool:
    result = _request("GET", "/utils/health-check/")
    return bool(result)


def list_stocks() -> list[dict[str, Any]]:
    return _request("GET", "/market/stocks")


def get_stock(symbol: str) -> dict[str, Any]:
    return _request("GET", f"/market/stocks/{symbol.upper()}")


def get_ohlc(symbol: str, days: int = 365) -> list[dict[str, Any]]:
    return _request("GET", f"/market/stocks/{symbol.upper()}/ohlc", params={"days": days})


def predict(symbol: str) -> dict[str, Any]:
    return _request("GET", f"/inference/predict/{symbol.upper()}")


def predict_base(symbol: str) -> dict[str, Any]:
    return _request("GET", f"/inference/predict-base/{symbol.upper()}")


def build_snapshot(payload: dict[str, Any]) -> dict[str, Any]:
    return _request("POST", "/data/build-snapshot", json=payload)


def list_snapshots() -> dict[str, Any]:
    return _request("GET", "/data/snapshots")


def download_snapshot(filename: str) -> io.BytesIO:
    payload = _request("GET", f"/data/snapshots/download/{filename}")
    return io.BytesIO(payload)


# ---------------------------------------------------------------------------
# Simulation / replay endpoints
# ---------------------------------------------------------------------------

def sim_symbols() -> list[str]:
    """Return the list of symbols available in the simulation loader."""
    return _request("GET", "/simulation/symbols")


def sim_session() -> dict[str, Any]:
    """Return replay session metadata (step count, labels, tree info)."""
    return _request("GET", "/simulation/session")


def sim_base(symbol: str) -> dict[str, Any]:
    """Full-day prediction from the base model (step_00, no warm refresh)."""
    return _request("GET", f"/simulation/base/{symbol.upper()}")


def sim_step(symbol: str, step: int) -> dict[str, Any]:
    """Prediction from the warm-refreshed model at a specific 15-min step (0–25)."""
    return _request("GET", f"/simulation/step/{symbol.upper()}/{step}")


def sim_history(symbol: str) -> list[dict[str, Any]]:
    """Fetch 15-min close prices for Mar 17–23 (5 trading days) for the context line chart."""
    return _request("GET", f"/simulation/history/{symbol.upper()}")


def sim_ohlc(symbol: str) -> list[dict[str, Any]]:
    """Fetch real 15-min OHLC bars specifically for the 2026-04-07 simulation day."""
    return _request("GET", f"/simulation/ohlc/{symbol.upper()}")


# ---------------------------------------------------------------------------
# Ops / observability endpoints
# ---------------------------------------------------------------------------

def ops_status() -> dict[str, Any]:
    """Full ops snapshot: services, machine, NIBI job, active model, data freshness."""
    return _request("GET", "/ops/status")


def ops_nibi_ssh() -> dict[str, Any]:
    """Check if the SSH ControlMaster socket to NIBI is alive."""
    return _request("GET", "/ops/nibi/ssh")


def ops_nibi_exec(command: str) -> dict[str, Any]:
    """Execute a whitelisted read-only command on NIBI."""
    return _request("POST", "/ops/nibi/exec", json={"command": command})


def ops_pipeline_logs(limit: int = 50) -> list[dict[str, Any]]:
    """Last N rows from operation_logs.pipeline_logs."""
    return _request("GET", "/ops/pipeline/logs", params={"limit": limit})


def ops_data_freshness() -> dict[str, Any]:
    """Latest window_ts and row count from ml.market_data_15m."""
    return _request("GET", "/ops/data/freshness")


def ops_airflow() -> dict[str, Any]:
    """DAG statuses and recent runs from the Airflow metadata database."""
    return _request("GET", "/ops/airflow")


def ops_nibi_relogin() -> dict[str, Any]:
    """
    Trigger auto_login.py on the backend to re-establish the SSH ControlMaster.
    In Duo push mode: returns immediately, user must approve push on phone.
    In TOTP mode: fully headless, socket active within ~10s.
    Poll ops_nibi_ssh() to know when socket comes alive.
    """
    return _request("POST", "/ops/nibi/relogin")


