#!/usr/bin/env python3
"""
Capability Freeze Enforcer — Step 9 of Foundation Hardening.

Enforces a freeze on new capability until steps 1-8 are verified.
This script can be run in CI to prevent merging new features
while the foundation is still being hardened.

Run: python scripts/enforce_capability_freeze.py
"""

import os
import sys
import subprocess
import json
from pathlib import Path
from datetime import datetime, timezone

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Foundation verification scripts (steps 1-8)
FOUNDATION_CHECKS = [
    ("Deployment Hygiene", "scripts/deployment_registry.py", ["status"]),
    ("Data Integrity", "scripts/verify_data_integrity.py", ["--days", "7"]),
    ("Track Record", "scripts/track_record_status.py", []),
    ("Decision Gate", "python -c \"from src.committee.decision_gate import check_decision_source_gate; print('OK')\"", []),
    ("Data Integrity Tests", "pytest tests/test_data_integrity.py -v --tb=short", []),
    ("Risk Parameter Audit", "scripts/audit_risk_parameters.py", []),
    ("Alerting System", "python -c \"from src.alerting import AlertingEngine; print('OK')\"", []),
    ("Config Dump", "scripts/dump_active_config.py", []),
]

FREEZE_FILE = PROJECT_ROOT / "FOUNDATION_FREEZE.md"
STATUS_FILE = PROJECT_ROOT / "data" / "foundation_status.json"


def run_check(name: str, command: str, args: list) -> dict:
    """Run a foundation check and return result."""
    full_cmd = command if isinstance(command, str) else " ".join([command] + args)
    
    try:
        if command.endswith(".py"):
            cmd = [sys.executable, command] + args
            result = subprocess.run(
                cmd,
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                timeout=120,
                env={**os.environ, "ALPACA_API_KEY": "dummy", "ALPACA_SECRET_KEY": "dummy"}
            )
        else:
            # Shell command
            result = subprocess.run(
                command,
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                timeout=120,
                shell=True,
                env={**os.environ, "ALPACA_API_KEY": "dummy", "ALPACA_SECRET_KEY": "dummy"}
            )
        
        success = result.returncode == 0
        return {
            "name": name,
            "command": full_cmd,
            "success": success,
            "stdout": result.stdout[-2000:] if result.stdout else "",
            "stderr": result.stderr[-2000:] if result.stderr else "",
            "returncode": result.returncode,
        }
    except subprocess.TimeoutExpired:
        return {
            "name": name,
            "command": full_cmd,
            "success": False,
            "stdout": "",
            "stderr": "TIMEOUT after 120s",
            "returncode": -1,
        }
    except Exception as e:
        return {
            "name": name,
            "command": full_cmd,
            "success": False,
            "stdout": "",
            "stderr": str(e),
            "returncode": -1,
        }


def print_section(title: str):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def main():
    print_section(f"FOUNDATION FREEZE STATUS — {datetime.now(timezone.utc).isoformat()}")
    
    results = []
    all_pass = True
    
    for name, cmd, args in FOUNDATION_CHECKS:
        print(f"\n  Checking: {name}...")
        result = run_check(name, cmd, args)
        results.append(result)
        
        status = "PASS" if result["success"] else "FAIL"
        print(f"    {status}: {name}")
        
        if not result["success"]:
            all_pass = False
            if result["stderr"]:
                print(f"    Error: {result['stderr'][:500]}")
    
    # Summary
    print_section("SUMMARY")
    passed = sum(1 for r in results if r["success"])
    total = len(results)
    
    for r in results:
        status = "[PASS]" if r["success"] else "[FAIL]"
        print(f"  {status} {r['name']}")
    
    print(f"\n  Passed: {passed}/{total}")
    
    # Save status
    status_data = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_checks": total,
        "passed": passed,
        "all_pass": all_pass,
        "results": results,
    }
    
    STATUS_FILE.parent.mkdir(exist_ok=True)
    with open(STATUS_FILE, "w") as f:
        json.dump(status_data, f, indent=2)
    print(f"\n  Status saved: {STATUS_FILE}")
    
    # Freeze status
    if all_pass:
        print("\n  [VERIFIED] FOUNDATION VERIFIED — Freeze can be lifted!")
        freeze_status = "VERIFIED"
    else:
        print("\n  [FREEZE ACTIVE] Foundation not ready")
        freeze_status = "ACTIVE"
    
    # Update freeze marker file
    with open(FREEZE_FILE, "w") as f:
        f.write(f"""# Capability Freeze Status

**Status:** {freeze_status}
**Last Check:** {datetime.now(timezone.utc).isoformat()}
**Passed:** {passed}/{total} checks

## Foundation Checks (Steps 1-8)

| Step | Check | Status |
|------|-------|--------|
""")
        for r in results:
            status = "PASS" if r["success"] else "FAIL"
            f.write(f"| {FOUNDATION_CHECKS[results.index(r)][0]} | {r['name']} | {status} |\n")
        
        f.write(f"""
## Policy

While freeze is **ACTIVE**:
- No new capability PRs will be merged
- Only bug fixes, foundation hardening, and documentation allowed
- New features must wait until all checks PASS

## To Lift Freeze

All {total} checks must PASS. Run:
```bash
python scripts/enforce_capability_freeze.py
```
""")
    
    # Exit code for CI
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()