#!/usr/bin/env python3
# Example usage:
#   # Compute ASR for all finalized results under assessed_results/final
#   python tools/calc_asr_from_final.py
#
#   # Write a JSON summary
#   python tools/calc_asr_from_final.py --out assessed_results/final/summary.json
#
#   # Only for a specific stem (can repeat)
#   python tools/calc_asr_from_final.py --only gemini_gpt4.1_mini --only qwen_gpt4o_mini
#
#   # Show per-technique breakdown too
#   python tools/calc_asr_from_final.py --by-tech
#
# The script expects JSONL files produced by tools/compare_llm_vs_heuristic.py --interactive,
# located under assessed_results/final/<stem>_final.jsonl with records that include:
#   - stem (string)
#   - run_leaf_key (string, e.g., 1773..._T1556.003_1_0)
#   - final_success (boolean)
# It deduplicates entries by run_leaf_key within each stem (keeps the first).

import argparse
import json
from pathlib import Path
from typing import Dict, Any, Iterable, Tuple, Optional
from collections import defaultdict


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


def stem_from_filename(p: Path) -> str:
    # filename format: <stem>_final.jsonl
    name = p.stem
    if name.endswith("_final"):
        return name[:-6]
    return name


def parse_technique(run_leaf_key: str) -> str:
    # 1773..._T1556.003_1_0 -> T1556.003
    parts = run_leaf_key.split("_")
    return parts[1] if len(parts) >= 2 else ""


def main():
    ap = argparse.ArgumentParser(description="Compute ASR from finalized results under assessed_results/final")
    ap.add_argument("--final-dir", default="assessed_results/final", help="Directory containing *_final.jsonl files")
    ap.add_argument("--out", help="Optional JSON path to write summary")
    ap.add_argument("--only", action="append", help="Limit to specific stem(s). Can repeat.")
    ap.add_argument("--by-tech", action="store_true", help="Include per-technique breakdown")
    args = ap.parse_args()

    final_dir = Path(args.final_dir).resolve()
    if not final_dir.exists():
        print(f"[error] final dir not found: {final_dir}")
        return

    files = sorted(final_dir.glob("*_final.jsonl"))
    if args.only:
        wanted = set(args.only)
        files = [p for p in files if stem_from_filename(p) in wanted]
    if not files:
        print("[warn] no final files found to process")
        return

    overall = {}
    grand_total = 0
    grand_success = 0

    for fp in files:
        stem = stem_from_filename(fp)
        seen_keys = set()
        total = 0
        success = 0
        by_tech_total = defaultdict(int)
        by_tech_success = defaultdict(int)

        for rec in read_jsonl(fp):
            key = str(rec.get("run_leaf_key") or "")
            if not key or key in seen_keys:
                continue
            seen_keys.add(key)
            fs = bool(rec.get("final_success", False))
            total += 1
            if fs:
                success += 1
            if args.by_tech:
                tech = parse_technique(key)
                by_tech_total[tech] += 1
                if fs:
                    by_tech_success[tech] += 1

        asr = (success / total) if total else 0.0
        overall[stem] = {
            "total": total,
            "success": success,
            "asr": asr,
        }
        if args.by_tech:
            per = {}
            for tech, tcnt in sorted(by_tech_total.items()):
                scnt = by_tech_success.get(tech, 0)
                per[tech or "<unknown>"] = {
                    "total": tcnt,
                    "success": scnt,
                    "asr": (scnt / tcnt) if tcnt else 0.0,
                }
            overall[stem]["by_technique"] = per

        grand_total += total
        grand_success += success

    grand_asr = (grand_success / grand_total) if grand_total else 0.0
    summary = {
        "overall": overall,
        "grand": {
            "total": grand_total,
            "success": grand_success,
            "asr": grand_asr,
        },
        "final_dir": str(final_dir),
        "by_tech_enabled": bool(args.by_tech),
    }

    # Console output
    print("\nASR from finalized results")
    for stem, s in overall.items():
        print(f"- {stem}: {s['success']}/{s['total']} = {s['asr']:.1%}")
        if args.by_tech and s.get("by_technique"):
            for tech, t in s["by_technique"].items():
                print(f"    {tech}: {t['success']}/{t['total']} = {t['asr']:.1%}")
    print(f"\nGRAND: {grand_success}/{grand_total} = {grand_asr:.1%}")

    if args.out:
        outp = Path(args.out).resolve()
        outp.parent.mkdir(parents=True, exist_ok=True)
        outp.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"[wrote] {outp}")


if __name__ == "__main__":
    main()

