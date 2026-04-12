from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from dotenv import load_dotenv
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sqlalchemy import create_engine, text

router = APIRouter()

ROOT = Path(__file__).resolve().parents[4]
load_dotenv(ROOT / ".env")

DB_USER = os.getenv("POSTGRES_USER", "postgres")
DB_PASS = os.getenv("POSTGRES_PASSWORD", "changethis")
DB_NAME = os.getenv("POSTGRES_DB", "app")
DB_HOST = os.getenv("POSTGRES_SERVER") or os.getenv("POSTGRES_HOST", "db")
DB_PORT = os.getenv("POSTGRES_PORT", "5432")
SNAPSHOT_DIR = Path(os.getenv("SNAPSHOT_DIR", str(ROOT / "datasets"))).expanduser()


class SnapshotRequest(BaseModel):
    ticker: str = "ALL"
    start_date: str | None = None
    end_date: str | None = None
    format: str = "parquet"


def ensure_snapshot_dir() -> Path:
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    return SNAPSHOT_DIR


def get_db_engine():
    conn_str = f"postgresql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    return create_engine(conn_str)


def get_all_tables() -> list[str]:
    engine = get_db_engine()
    query = "SELECT table_name FROM information_schema.tables WHERE table_schema = 'market'"
    with engine.connect() as conn:
        result = conn.execute(text(query))
        return [row[0] for row in result if row[0] not in ("daily_prices", "stocks")]


@router.post("/build-snapshot")
def build_snapshot(req: SnapshotRequest) -> dict[str, Any]:
    try:
        t0 = time.time()
        engine = get_db_engine()
        snapshot_dir = ensure_snapshot_dir()

        tickers_to_process = get_all_tables() if req.ticker.upper() == "ALL" else [req.ticker.upper()]
        if not tickers_to_process:
            raise HTTPException(status_code=404, detail="No valid ticker tables found in database.")

        all_dfs: list[pd.DataFrame] = []
        for tick in tickers_to_process:
            query = f'SELECT * FROM market."{tick}"'
            conditions: list[str] = []
            if req.start_date:
                conditions.append(f"date >= '{req.start_date}'")
            if req.end_date:
                conditions.append(f"date <= '{req.end_date}'")
            if conditions:
                query += " WHERE " + " AND ".join(conditions)
            query += " ORDER BY date ASC"

            with engine.connect() as conn:
                df = pd.read_sql(text(query), conn)

            if not df.empty:
                if "symbol" not in df.columns:
                    df["symbol"] = tick
                all_dfs.append(df)

        if not all_dfs:
            raise HTTPException(status_code=404, detail="No data found for requested parameters.")

        final_df = pd.concat(all_dfs, ignore_index=True)
        timestamp_label = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        results: dict[str, Any] = {}

        formats_to_build = ["parquet", "csv"] if req.format.lower() == "both" else [req.format.lower()]
        for fmt in formats_to_build:
            filename = f"snapshot_{req.ticker}_{timestamp_label}.{fmt}"
            filepath = snapshot_dir / filename

            if fmt == "parquet":
                table = pa.Table.from_pandas(final_df)
                pq.write_table(table, filepath)
                results["parquet_file"] = filename
            elif fmt == "csv":
                final_df.to_csv(filepath, index=False)
                results["csv_file"] = filename

        t1 = time.time()
        results.update(
            {
                "status": "success",
                "tickers_processed": len(tickers_to_process),
                "total_rows_extracted": len(final_df),
                "extraction_dir": str(snapshot_dir),
                "time_taken_sec": round(t1 - t0, 3),
            }
        )
        return results
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/snapshots")
def list_snapshots() -> dict[str, Any]:
    if not SNAPSHOT_DIR.exists():
        return {"snapshots": [], "directory": str(SNAPSHOT_DIR)}

    snapshot_info: list[dict[str, Any]] = []
    for fpath in SNAPSHOT_DIR.iterdir():
        if fpath.is_file():
            size_mb = fpath.stat().st_size / (1024 * 1024)
            snapshot_info.append({"filename": fpath.name, "size_mb": round(size_mb, 2)})

    return {"directory": str(SNAPSHOT_DIR), "snapshots": snapshot_info}


@router.get("/snapshots/download/{filename}")
def download_snapshot(filename: str) -> FileResponse:
    filepath = SNAPSHOT_DIR / filename
    if not filepath.exists():
        raise HTTPException(status_code=404, detail="Snapshot file not found.")

    return FileResponse(path=filepath, filename=filename, media_type="application/octet-stream")
