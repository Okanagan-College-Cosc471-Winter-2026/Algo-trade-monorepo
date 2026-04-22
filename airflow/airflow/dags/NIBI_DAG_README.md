# NIBI Daily Warm-Refresh DAG

**File:** `nibi_daily_training_dag.py`  
**DAG ID:** `nibi_daily_warm_refresh`  
**Schedule:** Mon–Fri at 10:00 UTC (06:00 ET)

This DAG drives the full daily ML training cycle — from exporting market data out of Postgres, through GPU training on the NIBI HPC cluster, to promoting the finished model into production and reloading the backend.

---

## What Problem This Solves

The XGBoost warm-refresh model needs to be retrained every trading day. Training 26 horizon-specific boosters on 505 symbols of 15-minute data requires a GPU that the VPS does not have. NIBI (Narval/nibi.sharcnet.ca, Alliance Canada) provides H100 GPU nodes via Slurm job submission.

The challenge is that NIBI is a remote HPC cluster:
- Jobs sit in a queue for an unpredictable amount of time before running
- SSH sessions are time-limited (MFA required per session)
- Failures can happen at any stage: SSH, Slurm, Python imports, out-of-memory
- Artifacts need to travel: DB → local → NIBI → local → production

This DAG orchestrates all of that with proper retry logic, failure detection, and atomic model promotion.

---

## Pipeline Diagram

```
                          ┌─────────────────────────┐
                          │   ssh_health_check       │  verify NIBI reachable + Slurm up
                          └────────────┬────────────┘
                    ┌───────────────────┼───────────────────┐
                    ▼                   ▼                   ▼
          ┌──────────────┐   ┌──────────────────┐  ┌────────────────┐
          │export_parquet│   │ sync_code_to_nibi│  │sync_base_model │
          │  DB → local  │   │  rsync ml/ tree  │  │ rsync current/ │
          └──────┬───────┘   └────────┬─────────┘  └───────┬────────┘
                 └───────────────────►│◄───────────────────┘
                                      ▼
                          ┌─────────────────────────┐
                          │  sync_parquet_to_nibi   │  SCP 427 MB → NIBI
                          └────────────┬────────────┘
                                       ▼
                          ┌─────────────────────────┐
                          │   submit_slurm_job       │  sbatch → job_id via XCom
                          └────────────┬────────────┘
                                       ▼
                          ┌─────────────────────────┐
                          │  poll_job_until_done     │  Sensor: poke every 2 min
                          │  (up to 9 hours)         │  releases worker between pokes
                          └────────────┬────────────┘
                                       ▼
                          ┌─────────────────────────┐
                          │   validate_artifacts     │  SIMULATION_DONE + 26 step dirs
                          └────────────┬────────────┘
                                       ▼
                          ┌─────────────────────────┐
                          │  rsync_artifacts_back   │  NIBI run_root/ → local
                          └────────────┬────────────┘
                                       ▼
                          ┌─────────────────────────┐
                          │    promote_model         │  atomic symlink swap
                          └────────────┬────────────┘
                                       ▼
                          ┌─────────────────────────┐
                          │    reload_backend        │  POST /admin/reload-model
                          └─────────────────────────┘
```

**Tasks 2, 3, 5 run in parallel** after the health check — exporting parquet, syncing code, and checking the base model are independent of each other. This saves ~3 minutes off wall-clock time before the job even submits.

---

## Task Reference

| # | Task ID | What It Does | Key Edge Cases |
|---|---------|-------------|----------------|
| 1 | `ssh_health_check` | SSH to NIBI, verify `echo pong` and `sbatch --version` both succeed | MFA expired → fail immediately with actionable message |
| 2 | `export_parquet` | Query `ml.market_data_15m` → write `snapshot_YYYY-MM-DD.parquet` locally | Skip if file already exists; write to `.tmp` then rename to prevent partial files |
| 3 | `sync_code_to_nibi` | rsync `ml/ml/` code tree to NIBI | Only transfers changed files; handles first-run missing directories |
| 4 | `sync_parquet_to_nibi` | SCP parquet to `NIBI:test_simulation/data/` | Skips if remote size matches local (saves 30s); detects partial remote write |
| 5 | `sync_base_model` | rsync local base model to `NIBI:run_root/current/` | Fails fast if base model missing locally; handles first-run |
| 6 | `submit_slurm_job` | `sbatch simulate_full_day.sbatch --skip-base` | Deduplication guard: reuses job_id if already submitted today; surfaces sbatch errors |
| 7 | `poll_job_until_done` | Sensor: check `squeue` then `sacct` every 2 min | See Sensor section below |
| 8 | `validate_artifacts` | Check `SIMULATION_DONE` sentinel + all 26 `step_XX/` directories | Slurm COMPLETED ≠ training succeeded; shows per-window failure detail |
| 9 | `rsync_artifacts_back` | rsync `NIBI:run_root/` → `local model_artifacts/sim_YYYY-MM-DD/` | Dated dir prevents overwriting; 30 min timeout for large bundles |
| 10 | `promote_model` | Atomic symlink: `current_base → sim_YYYY-MM-DD/` | Atomic rename prevents backend seeing missing symlink; cleans up artifacts >7 days old |
| 11 | `reload_backend` | `POST /api/v1/admin/reload-model` | Non-fatal: catches all exceptions, warns but never fails the DAG |

---

## Core Concepts

### Fail-Fast Gate (Task 1)

Every pipeline should start with the cheapest check that proves the environment is ready. If NIBI is unreachable or Slurm is down, you want to know in 5 seconds — not after wasting 3 minutes exporting a 427 MB parquet that can't be sent anywhere.

`BatchMode=yes` in the SSH command tells SSH to never prompt for credentials. If the ControlMaster socket is gone (MFA session expired), SSH returns exit code 255 immediately instead of hanging indefinitely waiting for keyboard input. This turns a silent hang into a loud, actionable failure.

```
Without BatchMode=yes:    ssh nibi "echo pong"  →  hangs waiting for Duo push
With BatchMode=yes:       ssh nibi "echo pong"  →  exits rc=255 in <1 second
```

---

### Idempotency (Tasks 2, 4, 5)

An idempotent task produces the same result whether run once or ten times. This matters because Airflow re-runs tasks on retry, manual clear, or backfill. Each of these tasks checks whether the expected output already exists before doing work:

- Task 2 checks if the parquet file exists before querying 6.6M rows from Postgres
- Task 4 compares remote file size before sending 427 MB over the network
- Task 5 uses rsync which transfers 0 bytes if nothing changed

The parquet export also writes to a `.tmp` file first, then renames atomically. This guarantees that a partial file from a previous crashed run is never mistaken for a complete one (a partial `.parquet` and a complete `.parquet` are different sizes — the size check would catch it, but the atomic write prevents the ambiguity entirely).

---

### XCom: Passing Values Between Tasks

Airflow tasks are isolated Python functions that cannot share variables directly. XCom (Cross-Communication) is Airflow's built-in key-value store for small inter-task values:

```python
# Task 6: push
ctx["ti"].xcom_push(key="job_id", value="12144848")

# Task 7: pull
job_id = ctx["ti"].xcom_pull(task_ids="submit_slurm_job", key="job_id")
```

XCom values are stored in the Airflow metadata database and survive task retries. If Task 6 succeeds but Task 8 fails and is retried, Task 7 can still pull the original `job_id` from XCom — no need to re-submit the Slurm job.

**What not to put in XCom:** large data (files, DataFrames). XCom is for small values like IDs, paths, counts. The parquet path is a string — fine. The parquet contents are not.

---

### Sensors vs Operators — The Most Important Design Choice

The original DRAC DAG used a single `SSHOperator` with an 8-hour bash `while` loop:

```bash
# OLD APPROACH — fragile
while true; do
    STATE=$(squeue -j "$JOB_ID" ...)
    [ "$STATE" = "COMPLETED" ] && exit 0
    sleep 120
done
```

**Problems with this:**
1. One SSH TCP connection held open for up to 8 hours. One network hiccup → task fails, even if the Slurm job is fine.
2. One Airflow worker slot is locked the entire time. Other DAG tasks cannot use it.

**The Sensor approach:**

```
poke() called every 120s:
  ┌─── open SSH ──► check squeue/sacct ──► close SSH ───┐
  │                                                       │
  └── returns False (keep waiting) or True (done) ───────┘
      between pokes: worker slot is RELEASED
```

Each `poke()` call is 2–3 seconds. Between pokes, `mode="reschedule"` releases the Airflow worker slot entirely. The sensor re-acquires a slot only for the next poke. This means:
- A dropped SSH connection only fails one poke, not the whole wait
- Worker slots are free for other tasks in other DAGs
- The sensor can wait 9 hours with essentially zero resource usage between pokes

**`mode="poke"` vs `mode="reschedule"`:**

| Mode | Worker slot | Use case |
|------|------------|---------|
| `poke` | Held continuously | Short waits (<5 min) |
| `reschedule` | Released between pokes | Long waits (minutes to hours) |

For an 8-hour GPU job, always use `reschedule`.

---

### squeue vs sacct

Slurm has two job query tools with different scopes:

```
squeue  → shows ONLY currently queued or running jobs
          returns nothing once a job finishes

sacct   → shows historical accounting records
          works even after a job has been gone for days
          but can have a brief delay (~30s) after job exits
```

The sensor checks both in sequence:

```python
# Step 1: is job still active?
squeue -j 12144848 -h -o '%T'
# → "RUNNING", "PENDING", "COMPLETING", or "" (finished)

# Step 2: if empty, what was the final state?
sacct -j 12144848 --format=State --noheader
# → "COMPLETED", "FAILED", "CANCELLED+", "TIMEOUT", etc.
```

If both return empty (race condition — job just finished, sacct not updated yet), the sensor returns `False` and retries next poke.

---

### Terminal State Detection (Task 7)

When `sacct` shows a non-COMPLETED terminal state, the sensor immediately pulls the `.err` log from NIBI and embeds it in the `AirflowException` message:

```python
_, err_content, _ = _ssh(
    f"tail -30 {NIBI_SIM_DIR}/logs/sim_full_day_{job_id}.err"
)
raise AirflowException(
    f"Job {job_id} ended with state: {final_state}\n"
    f"--- Last 30 lines of .err log ---\n{err_content}"
)
```

This is what would have caught the `No module named 'seaborn'` crash from job 12115066 immediately, rather than requiring a manual SSH session to read the logs.

**Terminal states and meanings:**

| State | Meaning | Action |
|-------|---------|--------|
| `COMPLETED` | Script exited 0 | Proceed to Task 8 |
| `FAILED` | Script exited non-zero | Raise with .err log |
| `CANCELLED` | Killed by admin or `scancel` | Raise — do not re-submit automatically |
| `TIMEOUT` | Hit `--time=08:00:00` wall | Raise — consider `--fast` mode for next run |
| `NODE_FAIL` | Hardware failure on compute node | Raise — re-submit is safe |
| `OUT_OF_MEMORY` | Exceeded `--mem=32G` | Raise — model too large, investigate |

---

### Post-Condition Validation (Task 8)

Slurm `COMPLETED` means the bash script exited with code 0. It does **not** mean the Python training script succeeded. The sbatch script uses `set -euo pipefail` so a Python crash will propagate, but network issues or partial runs can still produce a 0 exit code.

Two checks confirm the simulation actually ran fully:

**1. SIMULATION_DONE sentinel** — `run_simulation_day.py` writes this file only after all 26 windows complete successfully:
```python
sentinel = run_root / "SIMULATION_DONE"
sentinel.write_text(f"finished_at={...}\nsteps=26\nstatus=success\n")
```
If it's missing, we pull `simulation_progress.json` to show exactly how many windows completed and which ones failed.

**2. All 26 `step_XX/` directories** — each warm-refresh window writes its snapshot here. Missing directories mean partial training that would produce an incomplete model bundle.

---

### Atomic Symlink Swap (Task 10)

The backend reads the active model from a symlink:
```
model_artifacts/current_base  →  model_artifacts/sim_2026-04-07/
```

A naive approach would be:
```bash
rm current_base               # ← backend crashes here (symlink gone)
ln -s sim_2026-04-07 current_base
```

The atomic approach uses the Linux `rename()` syscall, which is guaranteed to be atomic — no process can observe an intermediate state:

```bash
ln -sfn sim_2026-04-07  current_base.new   # create with temp name
mv -f   current_base.new  current_base      # atomic rename into place
```

At every instant, `current_base` points to either the old model or the new model — never nothing. The backend is never in a state where the symlink doesn't exist.

---

### Non-Fatal Tail Task (Task 11)

Once the model is promoted (Task 10), the value of the pipeline is complete. The backend reload is a cache-warming optimization — it tells the backend to load the new model now rather than waiting for the next inference request.

If the backend is temporarily down, being redeployed, or the reload endpoint doesn't exist yet, we don't want the entire DAG to show `FAILED` — the model is already live. Task 11 catches all exceptions and prints them as warnings:

```python
except Exception as exc:
    print(f"WARNING: backend reload failed (non-fatal): {exc}")
    # does NOT re-raise — task still shows SUCCESS
```

---

### Duplicate Submission Guard (Task 6)

If the DAG succeeds through Task 6 (job submitted) but fails at Task 8 (validation) and is manually re-triggered, Task 6 would normally submit a second Slurm job — wasting an H100 allocation on work that's already done.

The guard writes a `logs/nibi_job_YYYY-MM-DD.json` file on submission:
```json
{"job_id": "12144848", "sim_date": "2026-04-13", "submitted_at": "...", "status": "submitted"}
```

On re-trigger, Task 6 checks this file first. If a job was already submitted today, it pushes the existing `job_id` to XCom and returns without calling `sbatch`. The sensor in Task 7 then picks up the existing job and polls it.

To force a re-submission (e.g. because the previous job failed), delete the record file:
```bash
rm logs/nibi_job_2026-04-13.json
```

---

## Configuration

All paths and connection settings come from environment variables, with sensible defaults:

| Variable | Default | Purpose |
|----------|---------|---------|
| `NIBI_USER` | `harshsaw` | NIBI username |
| `NIBI_SIM_DIR` | `/home/harshsaw/projects/def-youry/test_simulation` | Root on NIBI |
| `OLD_DB_HOST` | `localhost` | Source Postgres host |
| `OLD_DB_NAME` | `market_data` | Source database |
| `BASE_MODEL_DIR` | `/data/projects/the-project-maverick/model_artifacts/base_2026-04-07` | Pre-trained base model |
| `BACKEND_URL` | `http://localhost:8000` | Backend for reload call |

The SSH connection uses the `nibi` alias from `~/.ssh/config`. The alias must have an active ControlMaster socket (opened by `morning_login.sh`) before this DAG fires at 06:00 ET.

---

## Prerequisites

Before this DAG can run daily:

1. **morning_login.sh must run before 06:00 ET** — opens the SSH ControlMaster socket and authenticates the Duo MFA session. This is currently a manual step.

2. **Base model must exist** at `BASE_MODEL_DIR` — the daily job runs `--skip-base`, meaning it warm-refreshes from an existing trained model rather than training from scratch. The base model is retrained separately when needed.

3. **Airflow SSH connection** — `nibi` alias in `~/.ssh/config` on the Airflow worker, with `ControlPath` pointing to the socket opened by `morning_login.sh`.

---

## Failure Runbook

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| Task 1 fails: `Permission denied` | MFA session expired | `bash ml/ml/nibi/morning_login.sh` |
| Task 6 fails: `sbatch failed` | Quota exceeded or bad script syntax | Check `ssh nibi "quota"` and sbatch script |
| Task 7 raises `CANCELLED` | Slurm admin or scheduler killed it | Re-trigger DAG; check NIBI maintenance window |
| Task 7 raises `FAILED` with `.err` log | Python crash inside the job | Read the embedded error; fix the issue; re-trigger |
| Task 7 raises `TIMEOUT` | 8h wall time exceeded | Re-submit with `--fast` flag for a shorter run |
| Task 8 fails: `SIMULATION_DONE not found` | Some windows errored out | Check `simulation_progress.json` on NIBI |
| Task 10 fails: `Only N/26 step dirs` | Partial rsync or partial training | Re-run Task 9 + 10 manually after investigating |

---

## Relationship to the Cron Scheduler

This DAG replaces `services/collector/src/nibi_orchestrator.py` in the scheduler crontab for the NIBI training step. The cron scheduler is still used for:

- `*/15` during market hours → `run_15min_pipeline.py` (too frequent for Airflow overhead)
- `20:05 ET` nightly → `run_scheduled_operations.py` (short-running, no polling needed)

The NIBI job is the only step that needs Airflow — it's the only one that requires long polling, retry logic, artifact validation, and multi-step orchestration with dependencies.
