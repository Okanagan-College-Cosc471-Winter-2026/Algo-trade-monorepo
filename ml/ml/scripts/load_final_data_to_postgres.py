#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import psycopg

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = REPO_ROOT / "ml" / "data" / "final_data"
DEFAULT_SCHEMA = "market_intraday"
DEFAULT_TABLE = "prices_5min"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load intraday CSVs from ml/data/final_data into Postgres.",
    )
    parser.add_argument(
        "--data-dir",
        default=str(DEFAULT_DATA_DIR),
        help=f"Directory containing per-symbol CSVs. Default: {DEFAULT_DATA_DIR}",
    )
    parser.add_argument(
        "--symbols",
        default="ALL",
        help="Comma-separated symbols to load, or ALL.",
    )
    parser.add_argument(
        "--schema",
        default=DEFAULT_SCHEMA,
        help=f"Target schema. Default: {DEFAULT_SCHEMA}",
    )
    parser.add_argument(
        "--table",
        default=DEFAULT_TABLE,
        help=f"Target table. Default: {DEFAULT_TABLE}",
    )
    parser.add_argument(
        "--truncate",
        action="store_true",
        help="Truncate the target table before loading.",
    )
    parser.add_argument(
        "--host",
        default=os.getenv("POSTGRES_SERVER") or os.getenv("POSTGRES_HOST") or "localhost",
        help="Postgres host. Defaults to env or localhost.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("POSTGRES_PORT", "5432")),
        help="Postgres port. Default: 5432",
    )
    parser.add_argument(
        "--db",
        default=os.getenv("POSTGRES_DB", "app"),
        help="Postgres database name.",
    )
    parser.add_argument(
        "--user",
        default=os.getenv("POSTGRES_USER", "postgres"),
        help="Postgres user.",
    )
    parser.add_argument(
        "--password",
        default=os.getenv("POSTGRES_PASSWORD", "changethis"),
        help="Postgres password.",
    )
    return parser.parse_args()


def resolve_files(data_dir: Path, symbols_arg: str) -> list[Path]:
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory does not exist: {data_dir}")

    if symbols_arg.upper() == "ALL":
        files = sorted(data_dir.glob("*.csv"))
    else:
        requested = [symbol.strip().upper() for symbol in symbols_arg.split(",") if symbol.strip()]
        files = [data_dir / f"{symbol}.csv" for symbol in requested]

    missing = [str(path.name) for path in files if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing CSV files: {', '.join(missing)}")

    return files


def ensure_table(cur: psycopg.Cursor, schema: str, table: str) -> None:
    cur.execute(f"CREATE SCHEMA IF NOT EXISTS {schema}")
    cur.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {schema}.{table} (
            id BIGSERIAL PRIMARY KEY,
            symbol TEXT NOT NULL,
            ts TIMESTAMP NOT NULL,
            open DOUBLE PRECISION NOT NULL,
            high DOUBLE PRECISION NOT NULL,
            low DOUBLE PRECISION NOT NULL,
            close DOUBLE PRECISION NOT NULL,
            volume BIGINT NOT NULL,
            source_file TEXT NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            UNIQUE (symbol, ts)
        )
        """
    )
    cur.execute(
        f"CREATE INDEX IF NOT EXISTS ix_{table}_symbol ON {schema}.{table} (symbol)"
    )
    cur.execute(
        f"CREATE INDEX IF NOT EXISTS ix_{table}_ts ON {schema}.{table} (ts)"
    )


def load_files(conn: psycopg.Connection, files: list[Path], schema: str, table: str) -> tuple[int, int]:
    staged_rows = 0
    with conn.cursor() as cur:
        cur.execute("DROP TABLE IF EXISTS temp_intraday_stage")
        cur.execute(
            """
            CREATE TEMP TABLE temp_intraday_stage (
                symbol TEXT NOT NULL,
                ts TIMESTAMP NOT NULL,
                open DOUBLE PRECISION NOT NULL,
                high DOUBLE PRECISION NOT NULL,
                low DOUBLE PRECISION NOT NULL,
                close DOUBLE PRECISION NOT NULL,
                volume BIGINT NOT NULL,
                source_file TEXT NOT NULL
            ) ON COMMIT DROP
            """
        )

        for path in files:
            symbol = path.stem.upper()
            print(f"[load] {symbol} <- {path}")
            with path.open(newline="") as handle:
                reader = csv.DictReader(handle)
                with cur.copy(
                    """
                    COPY temp_intraday_stage
                    (symbol, ts, open, high, low, close, volume, source_file)
                    FROM STDIN
                    """
                ) as copy:
                    for row in reader:
                        copy.write_row(
                            (
                                symbol,
                                row["date"],
                                float(row["open"]),
                                float(row["high"]),
                                float(row["low"]),
                                float(row["close"]),
                                int(float(row["volume"])),
                                path.name,
                            )
                        )
                        staged_rows += 1

        cur.execute(
            f"""
            INSERT INTO {schema}.{table}
                (symbol, ts, open, high, low, close, volume, source_file)
            SELECT
                symbol,
                ts,
                open,
                high,
                low,
                close,
                volume,
                source_file
            FROM temp_intraday_stage
            ON CONFLICT (symbol, ts) DO UPDATE
            SET
                open = EXCLUDED.open,
                high = EXCLUDED.high,
                low = EXCLUDED.low,
                close = EXCLUDED.close,
                volume = EXCLUDED.volume,
                source_file = EXCLUDED.source_file
            """
        )
        upserted_rows = cur.rowcount

    return staged_rows, upserted_rows


def main() -> None:
    if load_dotenv is not None:
        load_dotenv(REPO_ROOT / ".env")

    args = parse_args()
    data_dir = Path(args.data_dir).resolve()
    files = resolve_files(data_dir, args.symbols)

    conninfo = (
        f"host={args.host} port={args.port} dbname={args.db} "
        f"user={args.user} password={args.password}"
    )
    print(
        f"[connect] host={args.host} port={args.port} db={args.db} "
        f"schema={args.schema} table={args.table}"
    )
    with psycopg.connect(conninfo) as conn:
        with conn.cursor() as cur:
            ensure_table(cur, args.schema, args.table)
            if args.truncate:
                print(f"[reset] Truncating {args.schema}.{args.table}")
                cur.execute(f"TRUNCATE TABLE {args.schema}.{args.table} RESTART IDENTITY")

        staged_rows, upserted_rows = load_files(conn, files, args.schema, args.table)
        conn.commit()

    print(
        f"[done] processed_files={len(files)} staged_rows={staged_rows} upserted_rows={upserted_rows}"
    )


if __name__ == "__main__":
    main()
