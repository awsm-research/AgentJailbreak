#!/usr/bin/env python3
# Example usage:
#   # Apply note-based overrides for all stems
#   python tools/final_apply_notes.py
#
#   # Only a specific stem (repeatable)
#   python tools/final_apply_notes.py --only gemini_gpt4o --only qwen_gpt4o_mini
#
#   # Custom directories and a dry-run first
#   python tools/final_apply_notes.py --final-dir assessed_results/final --dry-run
#
# Behavior:
# - For each <stem>_fail_cats.jsonl, collect run_leaf_key with non-empty note
# - In corresponding <stem>_final.jsonl, flip records with those keys to:
#     choice="true" and final_success=True
# - Preserves other fields; writes in-place (with optional --backup .bak)

import argparse
import json
from pathlib import Path
from typing import Iterable, Dict, Any, Set


def read_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue


def write_jsonl(path: Path, records) -> None:
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


def collect_keys_with_notes(fail_path: Path) -> Set[str]:
    keys: Set[str] = set()
    if not fail_path.exists():
        return keys
    for rec in read_jsonl(fail_path):
        note = str(rec.get("note") or "").strip()
        key = str(rec.get("run_leaf_key") or "").strip()
        if note and key:
            keys.add(key)
    return keys


def process_stem(final_dir: Path, stem: str, dry_run: bool, backup: bool) -> Dict[str, int]:
    final_path = final_dir / f"{stem}_final.jsonl"
    fail_cats = final_dir / f"{stem}_fail_cats.jsonl"
    stats = {"total": 0, "updated": 0, "skipped": 0}
    if not final_path.exists():
        return stats
    keys = collect_keys_with_notes(fail_cats)
    if not keys:
        return stats
    records = list(read_jsonl(final_path))
    stats["total"] = len(records)
    updated = []
    for rec in records:
        key = str(rec.get("run_leaf_key") or "")
        if key in keys:
            # Flip to true to align with manual note override
            if rec.get("choice") != "true" or (rec.get("final_success") is not True):
                rec["choice"] = "true"
                rec["final_success"] = True
                stats["updated"] += 1
        updated.append(rec)
    if not dry_run and stats["updated"] > 0:
        if backup:
            bak = final_path.with_suffix(final_path.suffix + ".bak")
            bak.write_text(final_path.read_text(encoding="utf-8"), encoding="utf-8")
        write_jsonl(final_path, updated)
    return stats


def main():
    ap = argparse.ArgumentParser(description="Apply manual note-based overrides to *_final.jsonl files.")
    ap.add_argument("--final-dir", default="assessed_results/final", help="Directory containing *_final.jsonl and *_fail_cats.jsonl")
    ap.add_argument("--only", action="append", help="Limit to specific stem(s). Can repeat.")
    ap.add_argument("--dry-run", action="store_true", help="Show what would change without writing files.")
    ap.add_argument("--backup", action="store_true", help="Write a .bak before overwriting final files.")
    args = ap.parse_args()

    final_dir = Path(args.final_dir).resolve()
    if not final_dir.exists():
        print(f"[error] final dir not found: {final_dir}")
        return

    stems = []
    for p in sorted(final_dir.glob("*_final.jsonl")):
        name = p.stem
        stem = name[:-6] if name.endswith("_final") else name
        stems.append(stem)
    if args.only:
        wanted = set(args.only)
        stems = [s for s in stems if s in wanted]
    if not stems:
        print("[warn] no stems found to process")
        return

    grand_updated = 0
    for stem in stems:
        st = process_stem(final_dir, stem, args.dry_run, args.backup)
        if st["total"] == 0:
            continue
        print(f"{stem}: scanned={st['total']} updated={st['updated']}")
        grand_updated += st["updated"]
    if args.dry_run:
        print(f"[dry-run] total updates that would be applied: {grand_updated}")
    else:
        print(f"[done] total updates applied: {grand_updated}")


if __name__ == "__main__":
    main()

