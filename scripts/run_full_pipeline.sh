#!/usr/bin/env bash
# Full local pipeline: bootstrap base model → promote → reload backend
set -euo pipefail

REPO=/data/projects/Algo-trade-monorepo
RUN_ROOT="$REPO/model_artifacts/local_run_root"
ARTIFACTS="$REPO/model_artifacts"
DATE=$(date +%Y-%m-%d)
DEST="$ARTIFACTS/base_$DATE"
VENV="/data/env"
LOG="$REPO/logs/bootstrap_$DATE.log"
BACKEND_URL="http://localhost:8000"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

log "=== Full pipeline start: $DATE ==="
log "Run root : $RUN_ROOT"
log "Dest     : $DEST"

# ── 1. Bootstrap base model ────────────────────────────────────────────────
log "--- STEP 1: bootstrap training ---"
"$VENV/bin/python3" "$REPO/ml/ml/XG_boost_3_multigpu_final.py" \
  --mode cycle \
  --db-url "postgresql+psycopg2://appuser:changeme@localhost:5433/algotrade" \
  --source-table ml.market_data_15m \
  --run-root "$RUN_ROOT" \
  --device auto \
  --base-window-months 18 \
  --log-level INFO \
  2>&1 | tee -a "$LOG"

log "--- STEP 1 complete ---"

# ── 2. Verify output ───────────────────────────────────────────────────────
if [ ! -d "$RUN_ROOT/current" ]; then
  log "ERROR: $RUN_ROOT/current not found after training — aborting"
  exit 1
fi
if [ ! -f "$RUN_ROOT/current/metadata.json" ]; then
  log "ERROR: metadata.json missing in run_root/current — aborting"
  exit 1
fi
log "Artifacts verified at $RUN_ROOT/current"

# ── 3. Copy to dated artifact dir ──────────────────────────────────────────
log "--- STEP 2: copy to $DEST ---"
rm -rf "$DEST"
cp -r "$RUN_ROOT/current" "$DEST"
log "Copied to $DEST"

# ── 4. Atomic symlink swap current_base → base_YYYY-MM-DD ──────────────────
log "--- STEP 3: promote current_base ---"
LINK="$ARTIFACTS/current_base"
TMP_LINK="$ARTIFACTS/.current_base_tmp_$$"
ln -sfn "$DEST" "$TMP_LINK"
mv -T "$TMP_LINK" "$LINK"
log "current_base → $DEST"

# ── 5. Reload backend ──────────────────────────────────────────────────────
log "--- STEP 4: reload backend ---"
RESPONSE=$(curl -s -X POST "$BACKEND_URL/api/v1/inference/admin/reload-model" \
  -H "Content-Type: application/json" || echo '{"error":"curl failed"}')
log "Backend response: $RESPONSE"

log "=== Pipeline complete ==="
log "Model ID: $(python3 -c "import json; d=json.load(open('$DEST/metadata.json')); print(d.get('model_id','?'))" 2>/dev/null || echo '?')"
