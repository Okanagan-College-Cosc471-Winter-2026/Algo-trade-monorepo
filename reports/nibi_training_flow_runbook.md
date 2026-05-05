# NIBI Training Flow Runbook

This note captures the checks and recovery flow used on May 1, 2026 when an April 30 base-model run was found to be using a bad snapshot and the wrong cutoff mapping.

## Key rule

In `ml/ml/nibi/run_simulation_day.py`, base training uses:

- `cutoff_date = sim_date - 1 day`

That means:

- if you want a base model trained through `2026-04-30`
- you must submit the NIBI base job with `--sim-date 2026-05-01`

Submitting `--sim-date 2026-04-30` trains through `2026-04-29`.

## Checks before base training

1. Confirm the local snapshot exists:
   - `datasets/snapshot_<cutoff>.parquet`

2. Validate snapshot coverage for the cutoff day:
   - snapshot contains rows for `trade_date == cutoff`
   - open bar `13:30 UTC` has full symbol coverage
   - close bar `19:30 UTC` has full symbol coverage
   - the cutoff-day symbol count matches the open/close bar symbol counts

3. Confirm there is no stale running warm-refresh/full-day job consuming the GPU when the goal is only base training.

4. Clear remote `run_root` before a clean re-run.

5. Submit the base job with `sim_date = cutoff + 1 day`.

## What was wrong in the failed run

The original `datasets/snapshot_2026-04-30.parquet` was incomplete for April 30:

- only `6` rows on `2026-04-30`
- only `2` symbols
- only early timestamps: `08:00`, `08:15`, `08:30 UTC`

That snapshot was not safe to train from.

## Fix applied

1. Cancelled the bad base-train job.
2. Rebuilt `datasets/snapshot_2026-04-30.parquet` from `ml.market_data_15m`.
3. Verified the rebuilt snapshot had full April 30 market-session coverage:
   - `504` symbols at `13:30 UTC`
   - `504` symbols at `19:30 UTC`
4. Re-uploaded the rebuilt parquet to NIBI.
5. Resubmitted the base-only job with `--sim-date 2026-05-01`.

## Automation added

`scripts/run_nibi_base_only.sh` now:

- computes `SIM_DATE = cutoff_date + 1 day`
- validates cutoff-day snapshot coverage before upload/submission
- writes `datasets/snapshot_<cutoff>.meta.json`
- writes live pipeline status to `logs/nibi_base_only_status.json`

## API / Frontend visibility

The ops API now exposes a `training_flow` block inside:

- `GET /api/v1/ops/status`

The frontend ops page renders this as a stage view showing:

- SSH socket
- snapshot validation
- base train
- base promotion
- warm refresh

Relevant files:

- `scripts/run_nibi_base_only.sh`
- `services/backend/app/modules/ops/api.py`
- `services/frontend/app.py`
