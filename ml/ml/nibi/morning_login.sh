#!/usr/bin/env bash
# morning_login.sh — Run once each morning to establish the SSH ControlMaster.
# After this, all automated SSH/SCP/rsync to NIBI works without MFA prompts
# for up to 24 hours (ControlPersist 24h).
#
# ── Authentication modes ──────────────────────────────────────────────────────
#
# Option A — Interactive (default, no setup required)
#   bash ml/ml/nibi/morning_login.sh
#   → prompts for password + Duo push once, then socket persists
#
# Option B — Headless Duo push (pexpect, password from env)
#   NIBI_PASSWORD='...' python3 ml/ml/nibi/auto_login.py
#   → supplies password automatically, still needs phone for Duo push
#
# Option C — Fully headless TOTP (no phone needed, requires CCDB setup)
#   NIBI_PASSWORD='...' NIBI_TOTP_SECRET='BASE32...' python3 ml/ml/nibi/auto_login.py
#   → completely automated, ideal for cron / Airflow
#   → one-time setup: register TOTP token at ccdb.alliancecan.ca → My Account → MFA
#
# ── Keepalive ─────────────────────────────────────────────────────────────────
# Cron entry (keeps socket alive — add with: crontab -e):
#   */20 * * * * ssh -O check nibi >> ~/.ssh/cm/keepalive.log 2>&1
#
# ─────────────────────────────────────────────────────────────────────────────

set -e

SSH_ALIAS="${NIBI_SSH_ALIAS:-nibi}"

# Idempotent — skip if socket already alive
if ssh -O check "$SSH_ALIAS" 2>/dev/null; then
  echo "ControlMaster to $SSH_ALIAS is already active — nothing to do."
  exit 0
fi

echo "Establishing SSH ControlMaster to $SSH_ALIAS ..."
echo "(You will be prompted for MFA — this happens ONCE for the day)"
echo ""

# -M = become ControlMaster, -N = no remote command
# -f = background after authentication succeeds
ssh -M -N -f "$SSH_ALIAS"

echo ""
echo "ControlMaster socket established. Valid for 24 hours."
echo "All pipeline SSH/SCP/rsync commands will now work without MFA."
echo ""
echo "Verify:"
echo "  ssh -O check $SSH_ALIAS"
