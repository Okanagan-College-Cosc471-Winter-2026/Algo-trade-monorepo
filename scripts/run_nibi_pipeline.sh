#!/usr/bin/env bash
# Full NIBI pipeline: export → sync → base train → warm windows → rsync back → promote → reload
set -euo pipefail

REPO=/data/projects/Algo-trade-monorepo
DATE=2026-04-29
DATASETS="$REPO/datasets"
ARTIFACTS="$REPO/model_artifacts"
LOG="$REPO/logs/nibi_pipeline_$DATE.log"

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
SIM_DEST=$ARTIFACTS/simulation_$DATE

SSH_E="ssh -i $NIBI_KEY -o ControlPath=$NIBI_SOCKET -o ControlMaster=no -o BatchMode=yes -o StrictHostKeyChecking=accept-new"
SSH="ssh -i $NIBI_KEY -o ControlPath=$NIBI_SOCKET -o ControlMaster=no -o BatchMode=yes -o ConnectTimeout=15 $NIBI_USER@$NIBI_HOST"

mkdir -p "$DATASETS" "$REPO/logs"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

# ── STEP 1: Export parquet ────────────────────────────────────────────────────
log "=== STEP 1: Export parquet from local DB ==="
if [[ -f "$LOCAL_PARQUET" ]]; then
    MB=$(du -m "$LOCAL_PARQUET" | cut -f1)
    log "Parquet already exists (${MB} MB) — skipping export"
else
    /home/ubuntu/env/bin/python3 - <<'PYEOF' 2>&1 | tee -a "$LOG"
import pandas as pd, pyarrow as pa, pyarrow.parquet as pq
from sqlalchemy import create_engine, text
from pathlib import Path

out = Path("/data/projects/Algo-trade-monorepo/datasets/snapshot_2026-04-28.parquet")
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

# ── STEP 5: Submit base train job ─────────────────────────────────────────────
log "=== STEP 5: Submit sim_base_train.sbatch (--fast, sim-date=$DATE) ==="
JOB1_OUT=$($SSH "cd $NIBI_SIM_DIR && sbatch ml/ml/nibi/sim_base_train.sbatch \
  --parquet $NIBI_PARQUET --sim-date $DATE --fast" 2>&1)
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

# ── STEP 8: Submit warm windows job ──────────────────────────────────────────
log "=== STEP 8: Submit sim_warm_windows.sbatch (sim-date=$DATE) ==="
JOB2_OUT=$($SSH "cd $NIBI_SIM_DIR && sbatch ml/ml/nibi/sim_warm_windows.sbatch \
  --parquet $NIBI_PARQUET --sim-date $DATE" 2>&1)
log "sbatch output: $JOB2_OUT"
JOB2_ID=$(echo "$JOB2_OUT" | grep -oP '(?<=Submitted batch job )\d+')
log "Warm windows job ID: $JOB2_ID"

# ── STEP 9: Poll until warm windows done ─────────────────────────────────────
log "=== STEP 9: Polling warm windows job $JOB2_ID ==="
while true; do
    STATE=$($SSH "sacct -j $JOB2_ID --format=State --noheader 2>/dev/null | head -1 | tr -d ' '" || echo "UNKNOWN")
    log "  Job $JOB2_ID state: $STATE"
    case "$STATE" in
        COMPLETED) log "Warm windows COMPLETED"; break ;;
        FAILED|CANCELLED|TIMEOUT|NODE_FAIL|OUT_OF_ME*|OUT_OF_MEMORY)
            log "ERROR: Warm windows job failed with state: $STATE"
            $SSH "cat $NIBI_SIM_DIR/logs/sim_warm_*$JOB2_ID*.err 2>/dev/null | tail -30" | tee -a "$LOG"
            exit 1 ;;
    esac
    sleep 60
done

# ── STEP 10: Verify SIMULATION_DONE ─────────────────────────────────────────
log "=== STEP 10: Verify SIMULATION_DONE and step dirs ==="
$SSH "ls $NIBI_RUN_ROOT/SIMULATION_DONE 2>&1 && ls $NIBI_RUN_ROOT/ | grep step_ | wc -l" | tee -a "$LOG"

# ── STEP 11: Rsync artifacts back ────────────────────────────────────────────
log "=== STEP 11: Rsync warm model and simulation artifacts back ==="
mkdir -p "$WARM_DEST" "$SIM_DEST"

# Warm model bundle: run_root/current/ → warm_DATE/
rsync -az --compress \
  -e "$SSH_E" \
  "$NIBI_USER@$NIBI_HOST:$NIBI_RUN_ROOT/current/" \
  "$WARM_DEST/"
log "Warm model rsynced to $WARM_DEST"

# Step predictions: run_root/step_XX/ → simulation_DATE/step_XX/
for i in $(seq -w 0 25); do
    STEP="step_$i"
    mkdir -p "$SIM_DEST/$STEP/predictions"
    rsync -az --compress -e "$SSH_E" \
      "$NIBI_USER@$NIBI_HOST:$NIBI_RUN_ROOT/$STEP/predictions/" \
      "$SIM_DEST/$STEP/predictions/"
    rsync -az -e "$SSH_E" \
      "$NIBI_USER@$NIBI_HOST:$NIBI_RUN_ROOT/$STEP/metadata.json" \
      "$SIM_DEST/$STEP/" 2>/dev/null || true
done
log "Simulation steps rsynced to $SIM_DEST"

# ── STEP 12: Promote ─────────────────────────────────────────────────────────
log "=== STEP 12: Promote current_base and current_simulation ==="
if [[ ! -f "$WARM_DEST/models/model_manifest.json" ]]; then
    log "ERROR: model_manifest.json not found in $WARM_DEST/models/ — refusing to promote"
    exit 1
fi

# Atomic symlink swap — use relative paths so symlink resolves inside Docker container
ln -sfn "$(basename "$WARM_DEST")" "$ARTIFACTS/current_base.new"
mv -T "$ARTIFACTS/current_base.new" "$ARTIFACTS/current_base"
log "Promoted: current_base → $(basename "$WARM_DEST")"

# Also update current_base_eod — this is the static EOD model used for all-day base predictions
ln -sfn "$(basename "$WARM_DEST")" "$ARTIFACTS/current_base_eod.new"
mv -T "$ARTIFACTS/current_base_eod.new" "$ARTIFACTS/current_base_eod"
log "Promoted: current_base_eod → $(basename "$WARM_DEST")"

N_STEPS=$(ls "$SIM_DEST" | grep -c "step_" || true)
if [[ "$N_STEPS" -ge 26 ]]; then
    ln -sfn "$(basename "$SIM_DEST")" "$ARTIFACTS/current_simulation.new"
    mv -T "$ARTIFACTS/current_simulation.new" "$ARTIFACTS/current_simulation"
    log "Promoted: current_simulation → $SIM_DEST"
else
    log "WARNING: only $N_STEPS/26 step dirs — skipping current_simulation promotion"
fi

# ── STEP 13: Reload backend ──────────────────────────────────────────────────
log "=== STEP 13: Reload backend ==="
curl -s -X POST http://localhost:8000/api/v1/inference/admin/reload-model | tee -a "$LOG"
echo ""
curl -s -X POST http://localhost:8000/api/v1/inference/admin/reload-base-model | tee -a "$LOG"
echo ""
curl -s -X POST http://localhost:8000/api/v1/simulation/admin/reload-simulation 2>/dev/null | tee -a "$LOG" || true
echo ""

log "=== Pipeline complete ==="
log "Warm model : $WARM_DEST"
log "Simulation : $SIM_DEST ($N_STEPS steps)"
