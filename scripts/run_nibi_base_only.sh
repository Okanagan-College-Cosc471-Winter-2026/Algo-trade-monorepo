#!/usr/bin/env bash
# Base-model-only NIBI pipeline: export → sync → base train → rsync back → promote → reload
# No warm windows, no simulation. Run this nightly before market open.
set -euo pipefail

REPO=/data/projects/Algo-trade-monorepo
DATE=${1:-$(date +%Y-%m-%d)}   # pass date as arg or default to today
DATASETS="$REPO/datasets"
ARTIFACTS="$REPO/model_artifacts"
LOG="$REPO/logs/nibi_base_only_$DATE.log"

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

log "=== Base-only pipeline: sim-date=$DATE ==="

# ── STEP 1: Export parquet ────────────────────────────────────────────────────
log "=== STEP 1: Export parquet from local DB ==="
if [[ -f "$LOCAL_PARQUET" ]]; then
    MB=$(du -m "$LOCAL_PARQUET" | cut -f1)
    log "Parquet already exists (${MB} MB) — skipping export"
else
    /home/ubuntu/env/bin/python3 - <<PYEOF 2>&1 | tee -a "$LOG"
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

# ── STEP 2: Sync ML code to NIBI ─────────────────────────────────────────────
log "=== STEP 2: Rsync ml/ code to NIBI ==="
$SSH "mkdir -p $NIBI_SIM_DIR/ml/ml"
rsync -az --delete \
  -e "$SSH_E" \
  "$REPO/ml/ml/" \
  "$NIBI_USER@$NIBI_HOST:$NIBI_SIM_DIR/ml/ml/"
log "Code synced"

# ── STEP 3: SCP parquet to NIBI ──────────────────────────────────────────────
log "=== STEP 3: SCP parquet to NIBI ==="
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

# ── STEP 4: Clean run_root on NIBI ───────────────────────────────────────────
log "=== STEP 4: Clean NIBI run_root ==="
$SSH "rm -rf $NIBI_RUN_ROOT && mkdir -p $NIBI_RUN_ROOT"
log "run_root cleared"

# ── STEP 5: Submit base train job (base-only, no warm windows) ───────────────
log "=== STEP 5: Submit sim_base_train.sbatch (sim-date=$DATE) ==="
JOB1_OUT=$($SSH "cd $NIBI_SIM_DIR && sbatch ml/ml/nibi/sim_base_train.sbatch \
  --parquet $NIBI_PARQUET --sim-date $DATE" 2>&1)
log "sbatch output: $JOB1_OUT"
JOB1_ID=$(echo "$JOB1_OUT" | grep -oP '(?<=Submitted batch job )\d+')
log "Base train job ID: $JOB1_ID"

# ── STEP 6: Poll until base train done ───────────────────────────────────────
log "=== STEP 6: Polling base train job $JOB1_ID ==="
while true; do
    STATE=$($SSH "sacct -j $JOB1_ID --format=State --noheader 2>/dev/null | head -1 | tr -d ' '" || echo "UNKNOWN")
    log "  Job $JOB1_ID state: $STATE"
    case "$STATE" in
        COMPLETED) log "Base train COMPLETED"; break ;;
        FAILED|CANCELLED|TIMEOUT|NODE_FAIL|OUT_OF_ME*|OUT_OF_MEMORY)
            log "ERROR: Base train job failed with state: $STATE"
            $SSH "cat $NIBI_SIM_DIR/logs/sim_base_*$JOB1_ID*.err 2>/dev/null | tail -30" | tee -a "$LOG"
            exit 1 ;;
    esac
    sleep 60
done

# ── STEP 7: Verify base model exists ─────────────────────────────────────────
log "=== STEP 7: Verify run_root/current/ on NIBI ==="
$SSH "ls $NIBI_RUN_ROOT/current/models/ 2>&1" | tee -a "$LOG"

# ── STEP 8: Rsync warm model bundle back ─────────────────────────────────────
log "=== STEP 8: Rsync base model artifacts back ==="
mkdir -p "$WARM_DEST"

rsync -az --compress \
  -e "$SSH_E" \
  "$NIBI_USER@$NIBI_HOST:$NIBI_RUN_ROOT/current/" \
  "$WARM_DEST/"
log "Base model rsynced to $WARM_DEST"

# ── STEP 9: Promote ──────────────────────────────────────────────────────────
log "=== STEP 9: Promote current_base and current_base_eod ==="
if [[ ! -f "$WARM_DEST/models/model_manifest.json" ]]; then
    log "ERROR: model_manifest.json not found in $WARM_DEST/models/ — refusing to promote"
    exit 1
fi

# Atomic relative symlink swap
ln -sfn "$(basename "$WARM_DEST")" "$ARTIFACTS/current_base.new"
mv -T "$ARTIFACTS/current_base.new" "$ARTIFACTS/current_base"
log "Promoted: current_base → $(basename "$WARM_DEST")"

ln -sfn "$(basename "$WARM_DEST")" "$ARTIFACTS/current_base_eod.new"
mv -T "$ARTIFACTS/current_base_eod.new" "$ARTIFACTS/current_base_eod"
log "Promoted: current_base_eod → $(basename "$WARM_DEST")"

# ── STEP 10: Reload backend ──────────────────────────────────────────────────
log "=== STEP 10: Reload backend ==="
curl -s -X POST http://localhost:8000/api/v1/inference/admin/reload-model | tee -a "$LOG"
echo ""
curl -s -X POST http://localhost:8000/api/v1/inference/admin/reload-base-model | tee -a "$LOG"
echo ""

log "=== Base-only pipeline complete ==="
log "Base model: $WARM_DEST"
