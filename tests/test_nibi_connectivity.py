"""
TEST — NIBI HPC Connectivity
──────────────────────────────
Tests the full manual workflow path before we automate it:

  Step 1  SSH reachable (can we log in?)
  Step 2  Remote dirs can be created
  Step 3  Small file can be SCP'd to NIBI
  Step 4  sbatch is available on NIBI
  Step 5  A trivial Slurm job can be submitted and polled to completion

Run on the VPS (NOT inside Docker, needs real SSH key):
    export NIBI_USER=youruser
    export NIBI_HOST=nibi.ok.ubc.ca
    export NIBI_SCRATCH=/scratch/youruser
    export NIBI_SSH_KEY=~/.ssh/nibi_key

    python tests/test_nibi_connectivity.py

Or with pytest:
    pytest tests/test_nibi_connectivity.py -v -s
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pytest

# ── Env vars ─────────────────────────────────────────────────────────────────
NIBI_USER    = os.getenv("NIBI_USER")
NIBI_HOST    = os.getenv("NIBI_HOST", "nibi.ok.ubc.ca")
NIBI_SCRATCH = os.getenv("NIBI_SCRATCH")
NIBI_SSH_KEY = os.path.expanduser(os.getenv("NIBI_SSH_KEY", "~/.ssh/nibi_key"))
NIBI_ALIAS   = "nibi"  # matches Host alias in ~/.ssh/config — no MFA after ControlMaster is up

_all_set = all([NIBI_USER, NIBI_HOST, NIBI_SCRATCH])
skip_msg = "NIBI_USER / NIBI_HOST / NIBI_SCRATCH env vars not set"


def ssh(cmd: str, timeout: int = 30) -> tuple[int, str, str]:
    """Run a command on NIBI via SSH.
    Uses the 'nibi' alias from ~/.ssh/config which has ControlMaster set.
    After the morning login (MFA once), all calls reuse the existing socket.
    """
    full = [
        "ssh",
        "-o", "BatchMode=yes",
        "-o", f"ConnectTimeout={timeout}",
        NIBI_ALIAS,
        cmd,
    ]
    result = subprocess.run(full, capture_output=True, text=True, timeout=timeout + 5)
    return result.returncode, result.stdout.strip(), result.stderr.strip()


def scp_to(local: str, remote: str, timeout: int = 60) -> tuple[int, str]:
    """SCP a local file to NIBI via the 'nibi' alias (reuses ControlMaster socket)."""
    full = [
        "scp",
        "-o", "BatchMode=yes",
        local,
        f"{NIBI_ALIAS}:{remote}",
    ]
    result = subprocess.run(full, capture_output=True, text=True, timeout=timeout)
    return result.returncode, result.stderr.strip()


# ── Tests ─────────────────────────────────────────────────────────────────────

@pytest.mark.skipif(not _all_set, reason=skip_msg)
class TestNibiSSH:

    def test_step1_ssh_login(self):
        """Can we SSH to NIBI and run a trivial command?"""
        rc, out, err = ssh("echo hello_nibi")
        assert rc == 0, f"SSH failed (rc={rc}): {err}"
        assert "hello_nibi" in out, f"Unexpected output: {out!r}"
        print(f"\n  SSH OK — {NIBI_USER}@{NIBI_HOST}")

    def test_step2_scratch_dir_exists_or_can_create(self):
        """Is $NIBI_SCRATCH accessible?"""
        rc, out, err = ssh(f"mkdir -p {NIBI_SCRATCH}/data {NIBI_SCRATCH}/ml/logs && echo OK")
        assert rc == 0, f"mkdir failed: {err}"
        assert "OK" in out
        print(f"  Scratch dir ready: {NIBI_SCRATCH}")

    def test_step3_scp_small_file(self):
        """Can we copy a small file to NIBI?"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("nibi_test_payload\n")
            local_path = f.name

        remote_path = f"{NIBI_SCRATCH}/data/_test_scp.txt"
        rc, err = scp_to(local_path, remote_path)
        assert rc == 0, f"SCP failed: {err}"

        # Verify the file arrived
        rc2, out2, err2 = ssh(f"cat {remote_path}")
        assert rc2 == 0
        assert "nibi_test_payload" in out2

        # Cleanup
        ssh(f"rm -f {remote_path}")
        os.unlink(local_path)
        print("  SCP OK — file transferred and verified")

    def test_step4_sbatch_available(self):
        """Is sbatch on PATH on NIBI?"""
        rc, out, err = ssh("which sbatch && sbatch --version")
        assert rc == 0, f"sbatch not found: {err}"
        print(f"  sbatch OK — {out.splitlines()[0] if out else '?'}")

    def test_step5_submit_trivial_job_and_poll(self):
        """Submit a 1-line hello-world job and wait for COMPLETED (max 5 min)."""
        # Write a minimal sbatch script to NIBI
        script = (
            "#!/bin/bash\\n"
            "#SBATCH --job-name=nibi_conn_test\\n"
            "#SBATCH --time=00:01:00\\n"
            "#SBATCH --cpus-per-task=1\\n"
            "#SBATCH --mem=256M\\n"
            "echo 'NIBI_CONN_TEST_OK'\\n"
        )
        remote_script = f"{NIBI_SCRATCH}/_conn_test.sbatch"
        rc, out, err = ssh(f"printf '{script}' > {remote_script} && echo WRITTEN")
        assert rc == 0 and "WRITTEN" in out, f"Could not write test script: {err}"

        # Submit
        rc, out, err = ssh(f"sbatch {remote_script}")
        assert rc == 0, f"sbatch submit failed: {err}"

        # Parse job ID from "Submitted batch job 12345"
        job_id = None
        for token in out.split():
            if token.isdigit():
                job_id = token
                break
        assert job_id, f"Could not parse job ID from: {out!r}"
        print(f"  Submitted job {job_id}")

        # Poll for up to 5 minutes
        deadline = time.time() + 300
        final_state = None
        while time.time() < deadline:
            rc, out, err = ssh(f"squeue -j {job_id} -h -o '%T' 2>/dev/null || echo DONE")
            state = out.strip()
            if not state or state == "DONE":
                # Job no longer in queue — check sacct
                rc2, out2, _ = ssh(
                    f"sacct -j {job_id} --noheader --format=State | head -1"
                )
                final_state = out2.strip().split()[0] if out2.strip() else "COMPLETED"
                break
            if state in ("FAILED", "CANCELLED", "TIMEOUT", "NODE_FAIL"):
                final_state = state
                break
            print(f"  [{state}] polling...", flush=True)
            time.sleep(20)
        else:
            pytest.fail(f"Job {job_id} did not finish within 5 minutes")

        assert final_state in ("COMPLETED", "COMPLETING"), (
            f"Job ended in unexpected state: {final_state}"
        )
        print(f"  Job {job_id} finished: {final_state}")

        # Cleanup
        ssh(f"rm -f {remote_script}")


# ── Standalone run ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if not _all_set:
        print("ERROR: Set NIBI_USER, NIBI_HOST, NIBI_SCRATCH before running.")
        print()
        print("  export NIBI_USER=youruser")
        print("  export NIBI_HOST=nibi.ok.ubc.ca")
        print("  export NIBI_SCRATCH=/scratch/youruser")
        print("  export NIBI_SSH_KEY=~/.ssh/nibi_key   # default")
        sys.exit(1)

    print(f"Testing NIBI connectivity: {NIBI_USER}@{NIBI_HOST}")
    print(f"  Scratch : {NIBI_SCRATCH}")
    print(f"  SSH key : {NIBI_SSH_KEY}")
    print()

    steps = [
        ("Step 1  SSH login",             TestNibiSSH().test_step1_ssh_login),
        ("Step 2  Scratch dirs",          TestNibiSSH().test_step2_scratch_dir_exists_or_can_create),
        ("Step 3  SCP small file",        TestNibiSSH().test_step3_scp_small_file),
        ("Step 4  sbatch available",      TestNibiSSH().test_step4_sbatch_available),
        ("Step 5  Submit + poll job",     TestNibiSSH().test_step5_submit_trivial_job_and_poll),
    ]

    passed, failed = 0, 0
    for name, fn in steps:
        print(f"[RUN] {name}")
        try:
            fn()
            print(f"[PASS] {name}\n")
            passed += 1
        except Exception as exc:
            print(f"[FAIL] {name}: {exc}\n")
            failed += 1

    print("=" * 50)
    print(f"Results: {passed} passed, {failed} failed")
    if failed:
        sys.exit(1)
