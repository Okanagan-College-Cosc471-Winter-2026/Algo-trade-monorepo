#!/usr/bin/env bash
# post_backfill_etl.sh — Run after gather_past_data.py completes.
#
# Pipeline:
#   1. stg_raw → core_dbms (export_stg_to_core)
#   2. core_dbms → dw.market_data_15m (process_15min_window for all new windows)
#   3. Export training parquet (datasets/training_snapshot_2026-04-17.parquet)
#   4. SCP parquet to NIBI
#   5. Submit new NIBI job with --sim-date 2026-04-17

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

echo "============================================="
echo " Post-Backfill ETL Pipeline"
echo " $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "============================================="

CONTAINER="algo-trade-monorepo-scheduler-1"
PARQUET_OUT="datasets/training_snapshot_2026-04-17.parquet"
NIBI_DATA_DIR="~/projects/def-youry/test_simulation/data"
NIBI_RUN_ROOT="~/projects/def-youry/test_simulation/run_root_apr21"

# ── 1. Export stg_raw → core_dbms ────────────────────────────────
echo ""
echo "[1/5] Exporting stg_raw → core_dbms ..."
docker compose exec -T scheduler python /app/src/run_scheduled_operations.py 2>&1 | grep -E "INFO|processed|exported|error" | tail -10

# ── 2. Aggregate core_dbms → dw.market_data_15m ──────────────────
echo ""
echo "[2/5] Aggregating all new 15-min windows → dw.market_data_15m ..."
docker compose exec -T db psql -U appuser -d algotrade << 'SQL'
DO $$
DECLARE
    w TIMESTAMPTZ;
    cnt INT := 0;
BEGIN
    FOR w IN
        SELECT DISTINCT
            date_trunc('hour', ts) + INTERVAL '15 min' * (EXTRACT(MINUTE FROM ts)::int / 15)
        FROM core_dbms.market_data_5m
        WHERE ts > (
            SELECT COALESCE(MAX(window_ts), '2000-01-01'::timestamptz)
            FROM dw.market_data_15m
        )
        ORDER BY 1
    LOOP
        CALL dw.process_15min_window(w);
        cnt := cnt + 1;
    END LOOP;
    RAISE NOTICE 'Processed % new 15-min windows', cnt;
END;
$$;

SELECT 
    trade_date, 
    COUNT(DISTINCT symbol) AS symbols,
    COUNT(*) AS bars
FROM dw.market_data_15m
GROUP BY trade_date
ORDER BY trade_date DESC
LIMIT 8;
SQL

# ── 3. Export training parquet ────────────────────────────────────
echo ""
echo "[3/5] Exporting training parquet to ${PARQUET_OUT} ..."
python3 - <<PYEOF
import pandas as pd, pyarrow as pa, pyarrow.parquet as pq, psycopg2, io, time

DB = dict(host="localhost", port=5433, dbname="algotrade", user="appuser", password="changeme")
OUT = "${PARQUET_OUT}"

print("  Connecting to DB...")
conn = psycopg2.connect(**DB)
cur = conn.cursor()

# Export all data from dw.market_data_15m (excluding today's partial AAPL/MSFT)
print("  Querying dw.market_data_15m up to 2026-04-17...")
cur.execute("""
    SELECT 
        agg_id, symbol, window_ts, trade_date::text,
        open::float, high::float, low::float, close::float, volume,
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
    WHERE trade_date <= '2026-04-17'
    ORDER BY symbol, window_ts
""")
rows = cur.fetchall()
cols = [d[0] for d in cur.description]
conn.close()

print(f"  {len(rows):,} rows fetched")
df = pd.DataFrame(rows, columns=cols)
pq.write_table(pa.Table.from_pandas(df, preserve_index=False), OUT)
print(f"  Saved to {OUT} ({df['symbol'].nunique()} symbols, {len(df):,} rows)")
print(f"  Date range: {df['trade_date'].min()} → {df['trade_date'].max()}")
PYEOF

# ── 4. SCP parquet to NIBI ────────────────────────────────────────
echo ""
echo "[4/5] Uploading ${PARQUET_OUT} to NIBI ..."
REMOTE_PARQUET="${NIBI_DATA_DIR}/snapshot_2026-04-17.parquet"
scp "${PARQUET_OUT}" "nibi:${REMOTE_PARQUET}"
echo "  Uploaded to NIBI:${REMOTE_PARQUET}"

# ── 5. Submit new NIBI job ────────────────────────────────────────
echo ""
echo "[5/5] Updating NIBI sbatch script and submitting with --sim-date 2026-04-17 ..."
ssh nibi "
sed -i 's|snapshot_2026-04-15.parquet|snapshot_2026-04-17.parquet|g' \
    ~/projects/def-youry/test_simulation/run_apr21_sim.sbatch && \
sed -i 's|SIM_DATE=\"2026-04-08\"|SIM_DATE=\"2026-04-17\"|g' \
    ~/projects/def-youry/test_simulation/run_apr21_sim.sbatch && \
rm -rf ~/projects/def-youry/test_simulation/run_root_apr21/* && \
cd ~/projects/def-youry/test_simulation && \
sbatch run_apr21_sim.sbatch
"

echo ""
echo "============================================="
echo " ETL + NIBI submission complete!"
echo " Monitor: ssh nibi 'squeue -u harshsaw -o \"%.10i %.20j %.8T %.12M %R\"'"
echo " Promote:  bash scripts/promote_new_base.sh --dry-run"
echo "============================================="
