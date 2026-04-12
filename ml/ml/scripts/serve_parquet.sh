#!/bin/bash

PARQUET_DIR="/home/cosc-admin/the-project-maverick/ml/data/processed_parquet"
PORT=8000
TUNNEL_LOG="/tmp/parquet_tunnel.log"

# Kill any existing instances
pkill -f "http.server $PORT" 2>/dev/null
pkill -f "serveo.net" 2>/dev/null
sleep 1

# Start HTTP server
echo "Starting HTTP server on port $PORT..."
python3 -m http.server $PORT --directory "$PARQUET_DIR" &
HTTP_PID=$!

# Start Serveo tunnel (SSH-based, no install needed)
echo "Starting Serveo tunnel..."
ssh -o StrictHostKeyChecking=no -o ServerAliveInterval=60 \
    -R 80:localhost:$PORT serveo.net > "$TUNNEL_LOG" 2>&1 &
TUNNEL_PID=$!

# Wait for tunnel URL
sleep 5
URL=$(grep -o 'https://[^ ]*' "$TUNNEL_LOG" | head -1)

echo ""
echo "========================================"
echo "  Parquet server is live!"
echo "  Public URL: $URL"
echo "  Files: $(ls $PARQUET_DIR/*.parquet | wc -l) tickers"
echo ""
echo "  HTTP PID  : $HTTP_PID"
echo "  Tunnel PID: $TUNNEL_PID"
echo ""
echo "  To stop: kill $HTTP_PID $TUNNEL_PID"
echo "========================================"
echo ""
echo "Download snippet for DRAC notebook:"
echo "  BASE_URL = \"$URL\""
