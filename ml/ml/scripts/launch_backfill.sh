#!/usr/bin/env zsh
cd /home/cosc-admin/the-project-maverick
LOG="ml/data/quality_refetch_reports/missing_symbols_backfill.log"
PID_FILE="ml/data/quality_refetch_reports/missing_symbols_backfill.pid"
echo $$ > "$PID_FILE"
exec python -u ml/scripts/refetch_market_data_15m_quality.py \
  --mode simple-fmp \
  --start-date 2024-03-25 \
  --end-date 2026-04-07 \
  --symbols ABBV,ABNB,ABT,ACGL,ACN,ADBE,ADI,ADM,ADP,ADSK,AEE,AEP,AES,AFL,AIG,AIZ,AJG,AKAM,ALB,ALGN,ALL \
  --skip-treasuries \
  >> "$LOG" 2>&1 < /dev/null
