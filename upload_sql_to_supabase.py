"""
Upload SQL batch files to Supabase.

Two modes:
  1. RPC mode (default): Uses the exec_sql() database function via PostgREST.
     Requires only the anon key (hardcoded below).
  2. Management API mode (--mgmt): Uses the Supabase Management API.
     Requires SUPABASE_ACCESS_TOKEN env var.

Usage:
  python upload_sql_to_supabase.py             # RPC mode (recommended)
  python upload_sql_to_supabase.py --mgmt      # Management API mode
  python upload_sql_to_supabase.py --dry-run   # List files without executing
"""
import os
import sys
import json
import time
import requests
import glob

# --- Configuration ---
PROJECT_ID = "zdkoniqnvbklxtsviikl"
SUPABASE_URL = f"https://{PROJECT_ID}.supabase.co"
ANON_KEY = (
    "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9."
    "eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Inpka29uaXFudmJrbHh0c3ZpaWtsIiwi"
    "cm9sZSI6ImFub24iLCJpYXQiOjE3NzU2ODY4ODEsImV4cCI6MjA5MTI2Mjg4MX0."
    "3um6tE3eggaD1DSSEOYGWweSeUrzbzUiBVNxlCulgHY"
)
SQL_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "local_pipeline", "_upload_sql",
)


def execute_via_rpc(sql_content: str) -> tuple[bool, str]:
    """Execute SQL via the exec_sql() database function (PostgREST RPC)."""
    url = f"{SUPABASE_URL}/rest/v1/rpc/exec_sql"
    headers = {
        "apikey": ANON_KEY,
        "Authorization": f"Bearer {ANON_KEY}",
        "Content-Type": "application/json",
    }
    resp = requests.post(url, headers=headers, json={"query": sql_content}, timeout=300)
    if resp.status_code in (200, 204):
        return True, "OK"
    else:
        return False, f"HTTP {resp.status_code}: {resp.text[:500]}"


def execute_via_mgmt(sql_content: str, token: str) -> tuple[bool, str]:
    """Execute SQL via Supabase Management API."""
    url = f"https://api.supabase.com/v1/projects/{PROJECT_ID}/database/query"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    resp = requests.post(url, headers=headers, json={"query": sql_content}, timeout=300)
    if resp.status_code in (200, 201):
        return True, "OK"
    else:
        return False, f"HTTP {resp.status_code}: {resp.text[:500]}"


def main():
    use_mgmt = "--mgmt" in sys.argv
    dry_run = "--dry-run" in sys.argv

    mgmt_token = None
    if use_mgmt:
        mgmt_token = os.environ.get("SUPABASE_ACCESS_TOKEN")
        if not mgmt_token:
            print("ERROR: --mgmt requires SUPABASE_ACCESS_TOKEN env var.")
            sys.exit(1)
        print("Mode: Management API")
    else:
        print("Mode: RPC (exec_sql function)")

    print(f"SQL directory: {SQL_DIR}")
    print(f"Project: {PROJECT_ID}")

    # Collect all target files
    patterns = ["lstm_*.sql", "s1_*.sql", "s2_*.sql", "state_summary.sql"]
    all_files = []
    for pattern in patterns:
        matches = sorted(glob.glob(os.path.join(SQL_DIR, pattern)))
        all_files.extend(matches)

    print(f"Found {len(all_files)} SQL files to execute\n")

    if dry_run:
        for f in all_files:
            sz = os.path.getsize(f)
            print(f"  {os.path.basename(f):>20s}  {sz:>10,} bytes")
        print(f"\nTotal: {len(all_files)} files")
        return

    success_count = 0
    fail_count = 0
    t0 = time.time()

    for i, filepath in enumerate(all_files):
        filename = os.path.basename(filepath)
        filesize = os.path.getsize(filepath)

        with open(filepath, "r", encoding="utf-8") as f:
            sql_content = f.read()

        print(
            f"[{i+1:>3}/{len(all_files)}] {filename:<20s} ({filesize:>8,} bytes) ... ",
            end="",
            flush=True,
        )

        try:
            if use_mgmt:
                ok, msg = execute_via_mgmt(sql_content, mgmt_token)
            else:
                ok, msg = execute_via_rpc(sql_content)

            if ok:
                print("OK")
                success_count += 1
            else:
                print(f"FAILED  {msg}")
                fail_count += 1
        except requests.exceptions.Timeout:
            print("TIMEOUT")
            fail_count += 1
        except Exception as e:
            print(f"ERROR  {e}")
            fail_count += 1

        # Rate-limit protection
        if (i + 1) % 5 == 0:
            time.sleep(0.5)

    elapsed = time.time() - t0
    print(f"\n{'=' * 60}")
    print(f"Results: {success_count} succeeded, {fail_count} failed, {len(all_files)} total")
    print(f"Elapsed: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
