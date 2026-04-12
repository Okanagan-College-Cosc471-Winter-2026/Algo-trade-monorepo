#!/usr/bin/env bash
# ==============================================================================
# upload_to_gdrive.sh
# ──────────────────────────────────────────────────────────────────────────────
# Uploads the latest Parquet and CSV snapshots from the VPS datasets folder
# to Google Drive via rclone, and dynamically prints the gdown commands for DRAC.
#
# Usage:
#   ./upload_to_gdrive.sh
# ==============================================================================

set -e

DATA_DIR="/home/cosc-admin/the-project-maverick/datasets"
GDRIVE_REMOTE="gdrive:MaverickData"

# 1. Find the newest Parquet snapshot
LATEST_PARQUET=$(ls -t "${DATA_DIR}"/*.parquet 2>/dev/null | head -n 1)

if [ -z "$LATEST_PARQUET" ]; then
    echo "❌ ERROR: No .parquet files found in ${DATA_DIR}"
    exit 1
fi

FILENAME=$(basename "$LATEST_PARQUET")
echo "📦 Found latest snapshot: ${FILENAME}"
echo "⏳ Uploading to Google Drive (${GDRIVE_REMOTE}) ..."

# 2. Upload with rclone showing progress
rclone copy "${LATEST_PARQUET}" "${GDRIVE_REMOTE}/" -P

# 3. Get the Google Drive link ID
echo "🔗 Fetching public Google Drive link..."
RAW_LINK=$(rclone link "${GDRIVE_REMOTE}/${FILENAME}")

# The link looks like: https://drive.google.com/open?id=1mafi29exWIFRjPY36kFExEnYLIzlCIBl
# We need to extract just the `id=` part for gdown
FILE_ID=$(echo "$RAW_LINK" | grep -oP 'id=\K.*')

# 4. Print the exact commands needed for DRAC
echo ""
echo "✅ UPLOAD COMPLETE!"
echo "====================================================================="
echo "💻 Run this in your DRAC terminal to download it instantly:"
echo ""
echo "pip install --quiet gdown"
echo "gdown ${FILE_ID}"
echo ""
echo "====================================================================="
