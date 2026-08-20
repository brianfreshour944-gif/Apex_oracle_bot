#!/usr/bin/env python3
"""
Deployment Hygiene — Step 1 of Foundation Hardening.

Provides a single source of truth for what bot processes are running,
which accounts they're trading, and their current status.
Writes a machine-parseable JSON file that can be checked at a glance.

Run: python scripts/deployment_registry.py [--register|--heartbeat|--status|--cleanup]
"""

import os
import sys
import json
import time
import socket
import argparse
import subprocess
from pathlib import Path
from datetime import datetime, timezone

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REGISTRY_PATH = PROJECT_ROOT / "data" / "deployment_registry.json"
HEARTBEAT_TTL_SECONDS = 120  # Process considered dead after 2 missed heartbeats (60s interval)

def print_section(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")

def load_registry() -> dict:
    if REGISTRY_PATH.exists():
        try:
            with open(REGISTRY_PATH) as f:
                return json.load(f)
        except Exception:
            return {"processes": {}}
    return {"processes": {}}

def save_registry(data: dict):
    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = REGISTRY_PATH.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    tmp.replace(REGISTRY_PATH)

def get_git_info() -> dict:
    """Get current git commit and branch for version tracking."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=PROJECT_ROOT, stderr=subprocess.DEVNULL, text=True
        ).strip()
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=PROJECT_ROOT, stderr=subprocess.DEVNULL, text=True
        ).strip()
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=PROJECT_ROOT, stderr=subprocess.DEVNULL, text=True
        ).strip()
        return {"commit": commit, "branch": branch, "dirty": bool(dirty)}
    except Exception:
        return {"commit": "unknown", "branch": "unknown", "dirty": False}

def get_account_from_env() -> str:
    """Determine which Alpaca account this process is configured for."""
    base_url = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
    if "paper-api" in base_url:
        return "paper"
    return "live"

def register_process(args):
    """Register this process in the deployment registry."""
    pid = os.getpid()
    hostname = socket.gethostname()
    account = get_account_from_env()
    git_info = get_git_info()
    
    registry = load_registry()
    key = f"{hostname}:{pid}"
    
    registry["processes"][key] = {
        "pid": pid,
        "hostname": hostname,
        "account": account,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "last_heartbeat": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_info["commit"],
        "git_branch": git_info["branch"],
        "git_dirty": git_info["dirty"],
        "role": args.role,
        "symbols": args.symbols.split(",") if args.symbols else [],
        "status": "running"
    }
    
    save_registry(registry)
    print(f"Registered: {key} (account={account}, role={args.role})")

def heartbeat_process(args):
    """Update heartbeat for this process."""
    pid = os.getpid()
    hostname = socket.gethostname()
    key = f"{hostname}:{pid}"
    
    registry = load_registry()
    if key not in registry["processes"]:
        print(f"Process {key} not registered. Run with --register first.")
        sys.exit(1)
    
    registry["processes"][key]["last_heartbeat"] = datetime.now(timezone.utc).isoformat()
    registry["processes"][key]["status"] = "running"
    save_registry(registry)
    print(f"Heartbeat: {key}")

def cleanup_stale():
    """Remove processes that haven't heartbeated within TTL."""
    registry = load_registry()
    now = datetime.now(timezone.utc)
    removed = []
    
    for key, proc in list(registry["processes"].items()):
        last_hb = datetime.fromisoformat(proc["last_heartbeat"].replace("Z", "+00:00"))
        if (now - last_hb).total_seconds() > HEARTBEAT_TTL_SECONDS:
            proc["status"] = "stale"
            removed.append(key)
    
    if removed:
        print(f"Marked stale: {removed}")
        save_registry(registry)
    else:
        print("No stale processes found.")

def show_status():
    """Display current deployment status."""
    registry = load_registry()
    cleanup_stale()
    registry = load_registry()  # Reload after cleanup
    
    print_section("DEPLOYMENT REGISTRY STATUS")
    print(f"Registry file: {REGISTRY_PATH}")
    print(f"Total processes: {len(registry['processes'])}")
    
    has_stale = False
    has_dirty = False
    
    # Group by account
    by_account = {}
    for key, proc in registry["processes"].items():
        acct = proc.get("account", "unknown")
        by_account.setdefault(acct, []).append((key, proc))
    
    for account, procs in by_account.items():
        print(f"\n  Account: {account} ({len(procs)} process(es))")
        for key, proc in procs:
            last_hb = datetime.fromisoformat(proc["last_heartbeat"].replace("Z", "+00:00"))
            age = (datetime.now(timezone.utc) - last_hb).total_seconds()
            status = proc.get("status", "unknown")
            role = proc.get("role", "unknown")
            symbols = ", ".join(proc.get("symbols", [])) or "N/A"
            print(f"    {key}")
            print(f"      Role: {role} | Status: {status} | Last HB: {age:.0f}s ago")
            print(f"      Symbols: {symbols}")
            git_str = f"      Git: {proc.get('git_commit', '?')} ({proc.get('git_branch', '?')})"
            if proc.get('git_dirty'):
                git_str += " DIRTY"
                has_dirty = True
            print(git_str)
            if status == "stale":
                has_stale = True
    
    # Return non-zero if issues found
    if has_stale or has_dirty:
        return 1
    return 0

def main():
    parser = argparse.ArgumentParser(description="Deployment registry for bot processes")
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    # Register
    reg = subparsers.add_parser("register", help="Register this process")
    reg.add_argument("--role", default="trader", help="Process role (trader, analyzer, etc.)")
    reg.add_argument("--symbols", default="", help="Comma-separated symbols this process trades")
    reg.set_defaults(func=register_process)
    
    # Heartbeat
    hb = subparsers.add_parser("heartbeat", help="Update heartbeat for this process")
    hb.set_defaults(func=heartbeat_process)
    
    # Status
    st = subparsers.add_parser("status", help="Show deployment status")
    st.set_defaults(func=lambda _: sys.exit(show_status()))
    
    # Cleanup
    cl = subparsers.add_parser("cleanup", help="Remove stale processes")
    cl.set_defaults(func=lambda _: cleanup_stale())
    
    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()