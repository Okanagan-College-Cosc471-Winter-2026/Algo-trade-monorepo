#!/usr/bin/env python3
"""
auto_login.py — Headless SSH ControlMaster login to NIBI using pexpect.

What it does:
  1. Checks if ControlMaster socket is already alive (ssh -O check).
     If alive → exits immediately (idempotent).
  2. Spawns an interactive SSH session, drives it through:
     - Password prompt (from NIBI_PASSWORD env var or keyring)
     - Duo MFA prompt (sends push "1", or TOTP if NIBI_TOTP_SECRET is set)
  3. Detaches the session as a background ControlMaster (-N -f equivalent).

After this runs, all BatchMode=yes SSH/SCP/rsync to NIBI work without MFA.

Usage:
  # Option A — Duo push (approve on your phone)
  NIBI_PASSWORD='...' python3 ml/ml/nibi/auto_login.py

  # Option B — TOTP (requires TOTP secret from CCDB, no phone needed)
  NIBI_PASSWORD='...' NIBI_TOTP_SECRET='BASE32SECRET...' python3 ml/ml/nibi/auto_login.py

  # Option C — pipe password, use Duo push
  echo "$NIBI_PASSWORD" | python3 ml/ml/nibi/auto_login.py --stdin-password

Environment variables:
  NIBI_PASSWORD      SSH password for harshsaw@nibi (required)
  NIBI_TOTP_SECRET   TOTP shared secret from CCDB portal (optional)
                     If set, uses TOTP instead of Duo push — fully headless.
  NIBI_SSH_ALIAS     SSH alias to use (default: nibi)
  NIBI_TIMEOUT       Seconds to wait for Duo push (default: 60)

Dependencies:
  pip install pexpect pyotp  (pyotp only needed for TOTP mode)
"""

from __future__ import annotations

import os
import subprocess
import sys
import time

# ── Config ────────────────────────────────────────────────────────────────────
SSH_ALIAS   = os.getenv("NIBI_SSH_ALIAS", "nibi")
PASSWORD    = os.getenv("NIBI_PASSWORD", "")
TOTP_SECRET = os.getenv("NIBI_TOTP_SECRET", "")
TIMEOUT     = int(os.getenv("NIBI_TIMEOUT", "60"))


# ── Helpers ───────────────────────────────────────────────────────────────────

def socket_alive() -> bool:
    """Return True if ControlMaster socket is already alive."""
    r = subprocess.run(
        ["ssh", "-O", "check", SSH_ALIAS],
        capture_output=True, text=True,
    )
    return r.returncode == 0


def get_totp_code() -> str:
    """Generate current TOTP code from secret (requires pyotp)."""
    try:
        import pyotp  # type: ignore
    except ImportError:
        print("[auto_login] ERROR: pyotp not installed. Run: pip install pyotp")
        sys.exit(1)
    return pyotp.TOTP(TOTP_SECRET).now()


def run_login(password: str) -> None:
    """
    Drive the interactive SSH session using pexpect.
    Handles:
      - Password prompt
      - Duo "Enter a passcode or select one of the following options" prompt
      - Keyboard-interactive auth variants
    """
    try:
        import pexpect  # type: ignore
    except ImportError:
        print("[auto_login] ERROR: pexpect not installed. Run: pip install pexpect")
        sys.exit(1)

    print(f"[auto_login] Spawning SSH to {SSH_ALIAS} ...")

    # -M = become ControlMaster, -N = no command, -o ServerAliveInterval etc
    # already in ~/.ssh/config, but we add -o ControlMaster=yes to be explicit
    cmd = f"ssh -M -N -o ControlMaster=yes {SSH_ALIAS}"
    child = pexpect.spawn(cmd, encoding="utf-8", timeout=TIMEOUT)

    # Uncomment to see raw session output for debugging:
    # child.logfile = sys.stdout

    patterns = [
        # 0 — standard password prompt
        r"[Pp]assword:",
        # 1 — Duo MFA menu
        r"(Enter a passcode or select one of the following options|Passcode or option)",
        # 2 — TOTP / passcode prompt (after choosing option or direct)
        r"[Pp]asscode:",
        # 3 — already authenticated / mux already running
        r"(Master running|Control socket exists|Already connected)",
        # 4 — connection closed / auth failure
        r"(Connection closed|Permission denied|Authentication failed)",
        # 5 — EOF
        pexpect.EOF,
        # 6 — timeout
        pexpect.TIMEOUT,
    ]

    mfa_sent = False
    password_sent = False

    while True:
        idx = child.expect(patterns, timeout=TIMEOUT)

        if idx == 0:  # password prompt
            if password_sent:
                print("[auto_login] ERROR: Password rejected.")
                child.close(force=True)
                sys.exit(1)
            print("[auto_login] Sending password ...")
            child.sendline(password)
            password_sent = True

        elif idx == 1:  # Duo MFA menu
            if TOTP_SECRET:
                # Generate TOTP and type it directly
                code = get_totp_code()
                print(f"[auto_login] Sending TOTP code {code} ...")
                child.sendline(code)
            else:
                # Select option 1 = "Duo Push" (send push to registered device)
                print("[auto_login] Sending Duo push (option 1) — approve on your phone ...")
                child.sendline("1")
            mfa_sent = True

        elif idx == 2:  # bare passcode prompt (some NIBI login node variants)
            if TOTP_SECRET:
                code = get_totp_code()
                print(f"[auto_login] Sending TOTP code {code} ...")
                child.sendline(code)
            else:
                print("[auto_login] Waiting for Duo push approval ...")
                # Nothing to send for push — just wait for it to proceed
                # If prompt appears again, we've timed out
                time.sleep(2)

        elif idx == 3:  # master already running
            print("[auto_login] ControlMaster socket is already active.")
            child.close()
            return

        elif idx == 4:  # auth failure
            print(f"[auto_login] ERROR: Authentication failed — {child.match.group(0)!r}")
            child.close(force=True)
            sys.exit(1)

        elif idx == 5:  # EOF — session ended
            if mfa_sent:
                # Normal: -N session backgrounded itself or ControlMaster forked
                # Check if socket is now alive
                time.sleep(1)
                if socket_alive():
                    print("[auto_login] ControlMaster socket established (EOF after auth).")
                    return
                else:
                    print("[auto_login] ERROR: Session ended but socket not alive.")
                    sys.exit(1)
            else:
                print("[auto_login] ERROR: Connection closed before auth completed.")
                sys.exit(1)

        elif idx == 6:  # timeout
            if mfa_sent and not TOTP_SECRET:
                print(f"[auto_login] ERROR: Timed out waiting for Duo push ({TIMEOUT}s).")
                print("[auto_login] Did you approve the push on your phone?")
            else:
                print(f"[auto_login] ERROR: Timed out waiting for SSH prompt ({TIMEOUT}s).")
            child.close(force=True)
            sys.exit(1)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    # --stdin-password: read password from stdin (for pipe usage)
    if "--stdin-password" in sys.argv:
        password = sys.stdin.readline().rstrip("\n")
    else:
        password = PASSWORD

    if not password:
        print("[auto_login] ERROR: No password. Set NIBI_PASSWORD env var or use --stdin-password.")
        print("  export NIBI_PASSWORD='your_password'")
        sys.exit(1)

    # Idempotent — skip if already alive
    if socket_alive():
        print("[auto_login] ControlMaster already alive — nothing to do.")
        return

    if TOTP_SECRET:
        print("[auto_login] Mode: TOTP (fully headless — no phone needed)")
    else:
        print("[auto_login] Mode: Duo push (approve on your phone when prompted)")

    run_login(password)

    # Final check
    if socket_alive():
        print(f"[auto_login] SUCCESS — ControlMaster to {SSH_ALIAS} is active.")
        print("[auto_login] All pipeline SSH commands will now work without MFA.")
    else:
        print("[auto_login] WARNING: Could not confirm socket is alive after login.")
        sys.exit(1)


if __name__ == "__main__":
    main()
