#!/usr/bin/env bash
# morning_login.sh — Run once each morning to establish the SSH ControlMaster.
# After this, all automated SSH/SCP/rsync to NIBI works without MFA prompts
# for the next 10 hours (covers full market day).
#
# Usage:
#   bash ml/ml/nibi/morning_login.sh
#
# You will be prompted for your MFA code once. After that, the socket at
# ~/.ssh/cm/nibi-* persists and all pipeline scripts use it silently.

set -e

echo "Establishing SSH ControlMaster to NIBI..."
echo "(You will be prompted for MFA — this happens ONCE for the day)"
echo ""

# -N = no remote command, just open the tunnel
# -f = background after auth succeeds
ssh -N -f nibi

echo ""
echo "ControlMaster socket established. Valid for 10 hours."
echo "You can now run automated pipeline scripts without MFA prompts."
echo ""
echo "To verify the socket is active:"
echo "  ssh -O check nibi"
