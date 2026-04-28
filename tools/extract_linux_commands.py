#!/usr/bin/env python3
"""
Extract Linux commands from Atomic Red Team YAMLs and join with linux-index.csv.

Requirements:
- PyYAML (yaml)

Usage:
  python tools/extract_linux_commands.py \
    --index linux-index.csv \
    --atomics-dir atomics \
    --out linux-index-with-commands.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, Any, Optional

import yaml


def normalize_whitespace(s: str) -> str:
    # Collapse newlines to ' && ' to keep single-line command context
    lines = [ln.strip() for ln in s.splitlines() if ln.strip()]
    if not lines:
        return ""
    # Prefer ' && ' to indicate ordered steps without accidentally chaining destructive ops
    return " && ".join(lines)


def extract_command_from_test(test: Dict[str, Any]) -> str:
    executor = test.get("executor") or {}
    cmd = executor.get("command")
    if cmd:
        return normalize_whitespace(str(cmd))
    steps = executor.get("steps")
    if steps:
        return normalize_whitespace(str(steps))
    # Some atomics use 'inputs' passed into the executor; if no command/steps, leave blank
    return ""


def test_supports_linux(test: Dict[str, Any]) -> bool:
    plats = test.get("supported_platforms")
    if not plats:
        return False
    return any(str(p).lower() == "linux" for p in plats)


def load_atomic_yaml(technique_id: str, atomics_dir: Path) -> Optional[Dict[str, Any]]:
    yml_path = atomics_dir / technique_id / f"{technique_id}.yaml"
    if not yml_path.exists():
        return None
    try:
        return yaml.safe_load(yml_path.read_text(encoding="utf-8"))
    except Exception:
        return None


def find_command_for_row(row: Dict[str, str], atomics_dir: Path) -> str:
    technique = row.get("Technique #") or row.get("Technique") or ""
    technique = technique.strip()
    if not technique:
        return ""
    atomic = load_atomic_yaml(technique, atomics_dir)
    if not atomic:
        return ""
    tests = atomic.get("atomic_tests") or []
    if not isinstance(tests, list):
        return ""

    # Prefer a specific test by GUID if present
    target_guid = (row.get("Test GUID") or "").strip().lower()

    selected: Optional[Dict[str, Any]] = None
    if target_guid:
        for t in tests:
            guid = str(t.get("auto_generated_guid") or t.get("guid") or "").strip().lower()
            if guid == target_guid and test_supports_linux(t):
                selected = t
                break
    # Fallback to first linux-supported test
    if not selected:
        for t in tests:
            if test_supports_linux(t):
                selected = t
                break
    if not selected:
        return ""
    return extract_command_from_test(selected)


def main() -> None:
    ap = argparse.ArgumentParser(description="Join linux-index.csv with Atomic Red Team commands (Linux-only).")
    ap.add_argument("--index", default="linux-index.csv", help="Path to linux-index.csv")
    ap.add_argument("--atomics-dir", default="atomics", help="Path to Atomic Red Team 'atomics' directory")
    ap.add_argument("--out", default="linux-index-with-commands.csv", help="Output CSV path")
    args = ap.parse_args()

    index_path = Path(args.index)
    atomics_dir = Path(args.atomics_dir)
    out_path = Path(args.out)

    with index_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])

    if "Command" not in fieldnames:
        fieldnames.append("Command")

    for row in rows:
        cmd = find_command_for_row(row, atomics_dir)
        row["Command"] = cmd

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()

