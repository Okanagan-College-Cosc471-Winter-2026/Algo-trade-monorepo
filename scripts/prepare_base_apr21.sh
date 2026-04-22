#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/data/projects/Algo-trade-monorepo"
cd "${REPO_ROOT}"

echo "=== Prepare base-only model (cutoff 2026-04-21) ==="

echo "[1/7] Build full symbol list from DB..."
SYMBOLS="$(docker compose exec -T db psql -U appuser -d algotrade -t -A -c "SELECT string_agg(symbol, ',' ORDER BY symbol) FROM market.stocks;" | tr -d '\r\n')"
echo "    symbols: $(echo "${SYMBOLS}" | tr ',' '\n' | wc -l)"

echo "[2/7] Backfill 2026-04-20..2026-04-21 into stg_raw..."
docker compose exec -e SYMBOLS="${SYMBOLS}" scheduler \
  python /app/src/gather_past_data.py \
    --from-date 2026-04-20 \
    --to-date 2026-04-21 \
    --symbols "${SYMBOLS}"

echo "[3/7] Export stg_raw -> core_dbms..."
docker compose exec -T scheduler python /app/src/run_scheduled_operations.py

echo "[4/7] Aggregate 5m -> 15m (bulk UPSERT + feature update for 2026-04-20..2026-04-21)..."
docker compose exec -T db psql -U appuser -d algotrade <<'SQL'
INSERT INTO dw.market_data_15m (
    symbol, window_ts, open, high, low, close, volume, slot_count, status, created_at
)
SELECT
    symbol,
    date_trunc('hour', ts) + INTERVAL '15 min' * (EXTRACT(MINUTE FROM ts)::int / 15) AS window_ts,
    (array_agg(open  ORDER BY ts ASC))[1]  AS open,
    MAX(high)   AS high,
    MIN(low)    AS low,
    (array_agg(close ORDER BY ts DESC))[1] AS close,
    SUM(volume) AS volume,
    COUNT(*)    AS slot_count,
    CASE WHEN COUNT(*) >= 3 THEN 'complete' ELSE 'provisional' END AS status,
    CURRENT_TIMESTAMP AS created_at
FROM core_dbms.market_data_5m
WHERE ts >= '2026-04-20' AND ts < '2026-04-22'
  AND symbol IN (SELECT symbol FROM staging.ingestion_progress)
GROUP BY symbol,
    date_trunc('hour', ts) + INTERVAL '15 min' * (EXTRACT(MINUTE FROM ts)::int / 15)
ON CONFLICT (symbol, window_ts) DO UPDATE SET
    open       = EXCLUDED.open,
    high       = EXCLUDED.high,
    low        = EXCLUDED.low,
    close      = EXCLUDED.close,
    volume     = EXCLUDED.volume,
    slot_count = EXCLUDED.slot_count,
    status     = EXCLUDED.status,
    created_at = CURRENT_TIMESTAMP;

UPDATE dw.market_data_15m t
SET
    lag_close_1    = e.lag_close_1,
    lag_close_5    = e.lag_close_5,
    lag_close_10   = e.lag_close_10,
    close_diff_1   = CASE WHEN e.lag_close_1  IS NULL THEN NULL ELSE t.close - e.lag_close_1  END,
    close_diff_5   = CASE WHEN e.lag_close_5  IS NULL THEN NULL ELSE t.close - e.lag_close_5  END,
    pct_change_1   = CASE WHEN e.lag_close_1  IS NULL OR e.lag_close_1  = 0 THEN NULL ELSE (t.close - e.lag_close_1)  / e.lag_close_1  END,
    pct_change_5   = CASE WHEN e.lag_close_5  IS NULL OR e.lag_close_5  = 0 THEN NULL ELSE (t.close - e.lag_close_5)  / e.lag_close_5  END,
    log_return_1   = CASE WHEN e.lag_close_1  IS NULL OR e.lag_close_1  = 0 THEN NULL ELSE LN(t.close / e.lag_close_1) END,
    sma_close_5    = e.sma_close_5,
    sma_close_10   = e.sma_close_10,
    sma_close_20   = e.sma_close_20,
    sma_volume_5   = e.sma_volume_5,
    day_of_week    = e.day_of_week,
    hour_of_day    = e.hour_of_day,
    month_of_year  = e.month_of_year,
    trade_date     = (t.window_ts AT TIME ZONE 'America/New_York')::date,
    day_monday     = CASE WHEN e.day_of_week = 1 THEN 1 ELSE 0 END,
    day_tuesday    = CASE WHEN e.day_of_week = 2 THEN 1 ELSE 0 END,
    day_wednesday  = CASE WHEN e.day_of_week = 3 THEN 1 ELSE 0 END,
    day_thursday   = CASE WHEN e.day_of_week = 4 THEN 1 ELSE 0 END,
    day_friday     = CASE WHEN e.day_of_week = 5 THEN 1 ELSE 0 END,
    quarter_1      = CASE WHEN EXTRACT(QUARTER FROM t.window_ts) = 1 THEN 1 ELSE 0 END,
    quarter_2      = CASE WHEN EXTRACT(QUARTER FROM t.window_ts) = 2 THEN 1 ELSE 0 END,
    quarter_3      = CASE WHEN EXTRACT(QUARTER FROM t.window_ts) = 3 THEN 1 ELSE 0 END,
    quarter_4      = CASE WHEN EXTRACT(QUARTER FROM t.window_ts) = 4 THEN 1 ELSE 0 END,
    hour_early_morning  = CASE WHEN EXTRACT(HOUR FROM t.window_ts AT TIME ZONE 'America/New_York') BETWEEN 9  AND 10 THEN 1 ELSE 0 END,
    hour_mid_morning    = CASE WHEN EXTRACT(HOUR FROM t.window_ts AT TIME ZONE 'America/New_York') BETWEEN 10 AND 11 THEN 1 ELSE 0 END,
    hour_afternoon      = CASE WHEN EXTRACT(HOUR FROM t.window_ts AT TIME ZONE 'America/New_York') BETWEEN 13 AND 14 THEN 1 ELSE 0 END,
    hour_late_afternoon = CASE WHEN EXTRACT(HOUR FROM t.window_ts AT TIME ZONE 'America/New_York') BETWEEN 15 AND 16 THEN 1 ELSE 0 END
FROM (
    SELECT
        symbol, window_ts,
        close,
        LAG(close, 1)  OVER (PARTITION BY symbol ORDER BY window_ts) AS lag_close_1,
        LAG(close, 5)  OVER (PARTITION BY symbol ORDER BY window_ts) AS lag_close_5,
        LAG(close, 10) OVER (PARTITION BY symbol ORDER BY window_ts) AS lag_close_10,
        AVG(close)  OVER (PARTITION BY symbol ORDER BY window_ts ROWS BETWEEN 4  PRECEDING AND CURRENT ROW) AS sma_close_5,
        AVG(close)  OVER (PARTITION BY symbol ORDER BY window_ts ROWS BETWEEN 9  PRECEDING AND CURRENT ROW) AS sma_close_10,
        AVG(close)  OVER (PARTITION BY symbol ORDER BY window_ts ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) AS sma_close_20,
        AVG(volume) OVER (PARTITION BY symbol ORDER BY window_ts ROWS BETWEEN 4  PRECEDING AND CURRENT ROW) AS sma_volume_5,
        EXTRACT(DOW   FROM window_ts)::smallint AS day_of_week,
        EXTRACT(HOUR  FROM window_ts)::smallint AS hour_of_day,
        EXTRACT(MONTH FROM window_ts)::smallint AS month_of_year
    FROM dw.market_data_15m
    WHERE window_ts >= '2026-04-17'
) e
WHERE t.symbol = e.symbol
  AND t.window_ts = e.window_ts
  AND t.window_ts >= '2026-04-20';
SQL

echo "[5/7] Export parquet snapshot through 2026-04-21..."
python3 - <<'PYEOF'
import psycopg2
import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd
import time, os

DB = dict(host="localhost", port=5433, dbname="algotrade", user="appuser", password="changeme")
OUT = "/data/projects/Algo-trade-monorepo/datasets/training_snapshot_2026-04-21.parquet"
CHUNK = 150_000

QUERY = """
    SELECT agg_id, symbol, window_ts, trade_date::text,
        open::float, high::float, low::float, close::float, volume::bigint,
        slot_count, status, created_at,
        lag_close_1::float, lag_close_5::float, lag_close_10::float,
        close_diff_1::float, close_diff_5::float,
        pct_change_1::float, pct_change_5::float, log_return_1::float,
        sma_close_5::float, sma_close_10::float, sma_close_20::float, sma_volume_5::float,
        day_of_week, hour_of_day, month_of_year,
        day_monday, day_tuesday, day_wednesday, day_thursday, day_friday,
        quarter_1, quarter_2, quarter_3, quarter_4,
        hour_early_morning, hour_mid_morning, hour_afternoon, hour_late_afternoon,
        previous_close::float, overnight_gap_pct::float, overnight_log_return::float,
        is_gap_up, is_gap_down
    FROM dw.market_data_15m
    WHERE trade_date <= '2026-04-21'
      AND trade_date NOT IN ('2026-04-10','2026-04-11','2026-04-12','2026-04-18','2026-04-19')
    ORDER BY symbol, window_ts
"""

print("  Exporting parquet with server-side cursor...")
t0 = time.time()
conn = psycopg2.connect(**DB)
cur = conn.cursor(name="export_stream_apr21")
cur.itersize = CHUNK
cur.execute(QUERY)
first = cur.fetchmany(CHUNK)
cols = [d[0] for d in cur.description]

writer = None
total = 0
if first:
    df = pd.DataFrame(first, columns=cols)
    tbl = pa.Table.from_pandas(df, preserve_index=False)
    writer = pq.ParquetWriter(OUT, tbl.schema, compression="snappy")
    writer.write_table(tbl)
    total += len(first)
    print(f"    written {total:,} rows")
while True:
    rows = cur.fetchmany(CHUNK)
    if not rows:
        break
    df = pd.DataFrame(rows, columns=cols)
    tbl = pa.Table.from_pandas(df, preserve_index=False)
    if writer is None:
        writer = pq.ParquetWriter(OUT, tbl.schema, compression="snappy")
    writer.write_table(tbl)
    total += len(rows)
    print(f"    written {total:,} rows")

if writer is not None:
    writer.close()
conn.close()
print(f"  done: {OUT} ({total:,} rows, {os.path.getsize(OUT)/1e6:.1f} MB, {time.time()-t0:.0f}s)")
PYEOF

echo "[6/7] Upload parquet to NIBI..."
scp /data/projects/Algo-trade-monorepo/datasets/training_snapshot_2026-04-21.parquet \
  nibi:~/projects/def-youry/test_simulation/data/snapshot_2026-04-21.parquet

echo "[7/7] Submit base-only NIBI job (cutoff = 2026-04-21)..."
ssh nibi "cat > ~/projects/def-youry/test_simulation/run_base_only_apr21.sbatch << 'SBATCH_EOF'
#!/usr/bin/env bash
#SBATCH --job-name=algo_base_apr21
#SBATCH --output=/home/%u/projects/def-youry/test_simulation/logs/base_apr21_%j.out
#SBATCH --error=/home/%u/projects/def-youry/test_simulation/logs/base_apr21_%j.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --account=def-youry

set -euo pipefail
SIM_DIR=\"/home/\${USER}/projects/def-youry/test_simulation\"
RUN_ROOT=\"\${SIM_DIR}/run_root_base_apr21\"
PARQUET=\"\${SIM_DIR}/data/snapshot_2026-04-21.parquet\"
SIM_DATE=\"2026-04-22\"
SCRIPT=\"\${SIM_DIR}/ml/ml/nibi/run_simulation_day.py\"
VENV_PATH=\"\${HOME}/myenv\"

mkdir -p \"\${RUN_ROOT}\" \"\${SIM_DIR}/logs\"
rm -rf \"\${RUN_ROOT}\"/*
module load gcc arrow/17.0.0
source \"\${VENV_PATH}/bin/activate\"
python \"\${SCRIPT}\" --parquet \"\${PARQUET}\" --run-root \"\${RUN_ROOT}\" --sim-date \"\${SIM_DATE}\" --base-only
SBATCH_EOF
chmod +x ~/projects/def-youry/test_simulation/run_base_only_apr21.sbatch
cd ~/projects/def-youry/test_simulation && sbatch run_base_only_apr21.sbatch
squeue -u harshsaw -o '%.10i %.20j %.8T %.18S %R' 2>/dev/null"

echo "=== COMPLETE: base-only Apr21 prep queued ==="
