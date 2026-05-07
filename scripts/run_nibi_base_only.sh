#!/usr/bin/env bash
# Base-model-only NIBI pipeline: export → sync → base train → rsync back → promote → reload
# No warm windows, no simulation. Run this nightly before market open.
set -euo pipefail

REPO=/data/projects/Algo-trade-monorepo
DATE=${1:-$(date +%Y-%m-%d)}   # pass date as arg or default to today
SIM_DATE=$(date -d "$DATE + 1 day" +%Y-%m-%d)
DATASETS="$REPO/datasets"
ARTIFACTS="$REPO/model_artifacts"
LOG="$REPO/logs/nibi_base_only_$DATE.log"
STATUS_JSON="$REPO/logs/nibi_base_only_status.json"
SNAPSHOT_META="$DATASETS/snapshot_$DATE.meta.json"

NIBI_USER=harshsaw
NIBI_HOST=nibi.sharcnet.ca
NIBI_KEY=/home/ubuntu/.ssh/nibi_key
NIBI_SOCKET=/home/ubuntu/.ssh/cm/nibi-harshsaw@nibi.sharcnet.ca:22
NIBI_SIM_DIR=/home/harshsaw/projects/def-youry/test_simulation
NIBI_RUN_ROOT=$NIBI_SIM_DIR/run_root
NIBI_DATA_DIR=$NIBI_SIM_DIR/data
NIBI_PARQUET=$NIBI_DATA_DIR/snapshot_$DATE.parquet

LOCAL_PARQUET=$DATASETS/snapshot_$DATE.parquet
WARM_DEST=$ARTIFACTS/warm_$DATE

SSH_E="ssh -i $NIBI_KEY -o ControlPath=$NIBI_SOCKET -o ControlMaster=no -o BatchMode=yes -o StrictHostKeyChecking=accept-new"
SSH="ssh -i $NIBI_KEY -o ControlPath=$NIBI_SOCKET -o ControlMaster=no -o BatchMode=yes -o ConnectTimeout=15 $NIBI_USER@$NIBI_HOST"

mkdir -p "$DATASETS" "$REPO/logs"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

write_status() {
    local stage="$1"
    local status="$2"
    local message="$3"
    local job_id="${4:-}"
    STAGE="$stage" STATUS="$status" MESSAGE="$message" JOB_ID="$job_id" \
    DATE="$DATE" SIM_DATE="$SIM_DATE" LOCAL_PARQUET="$LOCAL_PARQUET" \
    SNAPSHOT_META="$SNAPSHOT_META" STATUS_JSON="$STATUS_JSON" \
    python3 - <<'PY'
import json
import os
from datetime import datetime, timezone
from pathlib import Path

status_path = Path(os.environ["STATUS_JSON"])
payload = {}
if status_path.exists():
    try:
        payload = json.loads(status_path.read_text())
    except Exception:
        payload = {}

snapshot_meta_path = Path(os.environ["SNAPSHOT_META"])
snapshot_meta = {}
if snapshot_meta_path.exists():
    try:
        snapshot_meta = json.loads(snapshot_meta_path.read_text())
    except Exception:
        snapshot_meta = {}

payload.update(
    {
        "pipeline": "nibi_base_only",
        "cutoff_date": os.environ["DATE"],
        "sim_date": os.environ["SIM_DATE"],
        "stage": os.environ["STAGE"],
        "status": os.environ["STATUS"],
        "message": os.environ["MESSAGE"],
        "job_id": os.environ.get("JOB_ID") or payload.get("job_id"),
        "snapshot_path": os.environ["LOCAL_PARQUET"],
        "snapshot_meta_path": os.environ["SNAPSHOT_META"],
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "snapshot": snapshot_meta,
    }
)
status_path.write_text(json.dumps(payload, indent=2))
PY
}

log "=== Base-only pipeline: cutoff-date=$DATE | sim-date=$SIM_DATE ==="
write_status "init" "running" "Base-only pipeline started"

# ── STEP 1: Export parquet ────────────────────────────────────────────────────
log "=== STEP 1: Export parquet from local DB ==="
write_status "export_snapshot" "running" "Building or reusing parquet snapshot"
if [[ -f "$LOCAL_PARQUET" ]]; then
    MB=$(du -m "$LOCAL_PARQUET" | cut -f1)
    log "Parquet already exists (${MB} MB) — skipping export"
else
    /data/env/bin/python3 - <<PYEOF 2>&1 | tee -a "$LOG"
import pandas as pd, pyarrow as pa, pyarrow.parquet as pq
from sqlalchemy import create_engine, text
from pathlib import Path

out = Path("/data/projects/Algo-trade-monorepo/datasets/snapshot_${DATE}.parquet")
tmp = out.with_suffix(".parquet.tmp")
engine = create_engine("postgresql+psycopg2://appuser:changeme@localhost:5433/algotrade")
chunks = []
with engine.connect() as conn:
    total = conn.execute(text("SELECT COUNT(*) FROM ml.market_data_15m")).scalar()
    print(f"Exporting {total:,} rows from ml.market_data_15m ...")
    for chunk in pd.read_sql(
        text("SELECT * FROM ml.market_data_15m ORDER BY symbol, window_ts"),
        conn, chunksize=200_000,
    ):
        chunks.append(chunk)
        print(f"  loaded {sum(len(c) for c in chunks):,} / {total:,}")
df = pd.concat(chunks, ignore_index=True)
pq.write_table(pa.Table.from_pandas(df), tmp)
tmp.rename(out)
print(f"Saved: {out}  ({out.stat().st_size/1e6:.1f} MB, {df['symbol'].nunique()} symbols)")
PYEOF
fi
write_status "export_snapshot" "ok" "Parquet snapshot is available"

# ── STEP 1b: Validate parquet cutoff coverage ────────────────────────────────
log "=== STEP 1b: Validate parquet cutoff coverage ==="
write_status "validate_snapshot" "running" "Validating cutoff-day coverage in parquet snapshot"
/data/env/bin/python3 - <<PYEOF 2>&1 | tee -a "$LOG"
import json
from pathlib import Path

import pandas as pd

cutoff_date = "${DATE}"
snapshot_path = Path("${LOCAL_PARQUET}")
meta_path = Path("${SNAPSHOT_META}")

df = pd.read_parquet(snapshot_path, columns=["symbol", "window_ts", "trade_date"])
df["window_ts"] = pd.to_datetime(df["window_ts"], utc=True)
cutoff_rows = df[df["trade_date"].astype(str) == cutoff_date].copy()
if cutoff_rows.empty:
    raise SystemExit(f"ERROR: snapshot contains no rows for cutoff date {cutoff_date}")

day_symbol_count = int(cutoff_rows["symbol"].nunique())
open_ts = pd.Timestamp(f"{cutoff_date} 13:30:00+00:00")
close_ts = pd.Timestamp(f"{cutoff_date} 19:30:00+00:00")
open_symbol_count = int(cutoff_rows.loc[cutoff_rows["window_ts"] == open_ts, "symbol"].nunique())
close_symbol_count = int(cutoff_rows.loc[cutoff_rows["window_ts"] == close_ts, "symbol"].nunique())

ok = day_symbol_count >= 400 and open_symbol_count == day_symbol_count and close_symbol_count == day_symbol_count
meta = {
    "cutoff_date": cutoff_date,
    "snapshot_path": str(snapshot_path),
    "rows_total": int(len(df)),
    "symbols_total": int(df["symbol"].nunique()),
    "max_window_ts_utc": str(df["window_ts"].max()),
    "cutoff_rows": int(len(cutoff_rows)),
    "cutoff_symbols": day_symbol_count,
    "cutoff_min_ts_utc": str(cutoff_rows["window_ts"].min()),
    "cutoff_max_ts_utc": str(cutoff_rows["window_ts"].max()),
    "open_bar_ts_utc": str(open_ts),
    "open_bar_symbols": open_symbol_count,
    "close_bar_ts_utc": str(close_ts),
    "close_bar_symbols": close_symbol_count,
    "validation_ok": bool(ok),
}
meta_path.write_text(json.dumps(meta, indent=2))

print(json.dumps(meta, indent=2))
if not ok:
    raise SystemExit(
        "ERROR: snapshot validation failed — expected cutoff-day open/close bars to carry full symbol coverage"
    )
PYEOF
write_status "validate_snapshot" "ok" "Snapshot validation passed"

# ── STEP 2: Sync ML code to NIBI ─────────────────────────────────────────────
log "=== STEP 2: Rsync ml/ code to NIBI ==="
write_status "sync_code" "running" "Syncing ML code to NIBI"
$SSH "mkdir -p $NIBI_SIM_DIR/ml/ml"
rsync -az --delete \
  -e "$SSH_E" \
  "$REPO/ml/ml/" \
  "$NIBI_USER@$NIBI_HOST:$NIBI_SIM_DIR/ml/ml/"
log "Code synced"
write_status "sync_code" "ok" "ML code synced to NIBI"

# ── STEP 3: SCP parquet to NIBI ──────────────────────────────────────────────
log "=== STEP 3: SCP parquet to NIBI ==="
write_status "upload_snapshot" "running" "Uploading parquet snapshot to NIBI"
LOCAL_SIZE=$(stat -c%s "$LOCAL_PARQUET")
REMOTE_SIZE=$($SSH "stat -c%s $NIBI_PARQUET 2>/dev/null || echo 0")
if [[ "$LOCAL_SIZE" == "$REMOTE_SIZE" ]]; then
    log "Remote parquet matches local (${LOCAL_SIZE} bytes) — skipping SCP"
else
    log "Uploading $(du -h "$LOCAL_PARQUET" | cut -f1) ..."
    scp -i "$NIBI_KEY" \
      -o "ControlPath=$NIBI_SOCKET" -o ControlMaster=no -o BatchMode=yes \
      "$LOCAL_PARQUET" "$NIBI_USER@$NIBI_HOST:$NIBI_PARQUET"
    log "Parquet uploaded"
fi
write_status "upload_snapshot" "ok" "Remote parquet is current"

# ── STEP 4: Clean run_root on NIBI ───────────────────────────────────────────
log "=== STEP 4: Clean NIBI run_root ==="
write_status "prepare_remote" "running" "Cleaning remote run_root"
$SSH "rm -rf $NIBI_RUN_ROOT && mkdir -p $NIBI_RUN_ROOT"
log "run_root cleared"
write_status "prepare_remote" "ok" "Remote run_root prepared"

# ── STEP 5: Submit base train job (base-only, no warm windows) ───────────────
log "=== STEP 5: Submit sim_base_train.sbatch (sim-date=$SIM_DATE => cutoff=$DATE) ==="
write_status "submit_base_train" "running" "Submitting base-train job"
JOB1_OUT=$($SSH "cd $NIBI_SIM_DIR && sbatch ml/ml/nibi/sim_base_train.sbatch \
  --parquet $NIBI_PARQUET --sim-date $SIM_DATE" 2>&1)
log "sbatch output: $JOB1_OUT"
JOB1_ID=$(echo "$JOB1_OUT" | grep -oP '(?<=Submitted batch job )\d+')
log "Base train job ID: $JOB1_ID"
write_status "submit_base_train" "ok" "Base-train job submitted" "$JOB1_ID"

# ── STEP 6: Poll until base train done ───────────────────────────────────────
log "=== STEP 6: Polling base train job $JOB1_ID ==="
write_status "base_train" "running" "Waiting for base-train job to finish" "$JOB1_ID"
while true; do
    STATE=$($SSH "sacct -j $JOB1_ID --format=State --noheader 2>/dev/null | head -1 | tr -d ' '" || echo "UNKNOWN")
    log "  Job $JOB1_ID state: $STATE"
    case "$STATE" in
        COMPLETED) log "Base train COMPLETED"; write_status "base_train" "ok" "Base-train job completed" "$JOB1_ID"; break ;;
        FAILED|CANCELLED|TIMEOUT|NODE_FAIL|OUT_OF_ME*|OUT_OF_MEMORY)
            log "ERROR: Base train job failed with state: $STATE"
            write_status "base_train" "error" "Base-train job failed with state $STATE" "$JOB1_ID"
            $SSH "cat $NIBI_SIM_DIR/logs/sim_base_*$JOB1_ID*.err 2>/dev/null | tail -30" | tee -a "$LOG"
            exit 1 ;;
    esac
    sleep 60
done

# ── STEP 7: Verify base model exists ─────────────────────────────────────────
log "=== STEP 7: Verify run_root/current/ on NIBI ==="
write_status "verify_remote_artifacts" "running" "Verifying remote base model artifacts" "$JOB1_ID"
$SSH "ls $NIBI_RUN_ROOT/current/models/ 2>&1" | tee -a "$LOG"
write_status "verify_remote_artifacts" "ok" "Remote base model artifacts present" "$JOB1_ID"

# ── STEP 8: Rsync warm model bundle back ─────────────────────────────────────
log "=== STEP 8: Rsync base model artifacts back ==="
write_status "download_artifacts" "running" "Syncing base artifacts back from NIBI" "$JOB1_ID"
mkdir -p "$WARM_DEST"

rsync -az --compress \
  -e "$SSH_E" \
  "$NIBI_USER@$NIBI_HOST:$NIBI_RUN_ROOT/current/" \
  "$WARM_DEST/"
log "Base model rsynced to $WARM_DEST"
write_status "download_artifacts" "ok" "Base artifacts synced back from NIBI" "$JOB1_ID"

# ── STEP 9: Promote ──────────────────────────────────────────────────────────
log "=== STEP 9: Promote current_base and current_base_eod ==="
write_status "promote_base" "running" "Promoting base model symlinks" "$JOB1_ID"
if [[ ! -f "$WARM_DEST/models/model_manifest.json" ]]; then
    log "ERROR: model_manifest.json not found in $WARM_DEST/models/ — refusing to promote"
    write_status "promote_base" "error" "Promotion refused because model_manifest.json is missing" "$JOB1_ID"
    exit 1
fi

# Atomic relative symlink swap
ln -sfn "$(basename "$WARM_DEST")" "$ARTIFACTS/current_base.new"
mv -T "$ARTIFACTS/current_base.new" "$ARTIFACTS/current_base"
log "Promoted: current_base → $(basename "$WARM_DEST")"

ln -sfn "$(basename "$WARM_DEST")" "$ARTIFACTS/current_base_eod.new"
mv -T "$ARTIFACTS/current_base_eod.new" "$ARTIFACTS/current_base_eod"
log "Promoted: current_base_eod → $(basename "$WARM_DEST")"
write_status "promote_base" "ok" "Base model promoted to current_base and current_base_eod" "$JOB1_ID"

# ── STEP 10: Reload backend ──────────────────────────────────────────────────
log "=== STEP 10: Reload backend ==="
write_status "reload_backend" "running" "Reloading backend model caches" "$JOB1_ID"
curl -s -X POST http://localhost:8000/api/v1/inference/admin/reload-model | tee -a "$LOG"
echo ""
curl -s -X POST http://localhost:8000/api/v1/inference/admin/reload-base-model | tee -a "$LOG"
echo ""
write_status "reload_backend" "ok" "Backend model caches reloaded" "$JOB1_ID"

log "=== Base-only pipeline complete ==="
log "Base model: $WARM_DEST"
write_status "completed" "ok" "Base-only pipeline completed" "$JOB1_ID"
