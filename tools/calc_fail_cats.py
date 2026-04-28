#!/usr/bin/env python3
# Example usage:
#   # Summarize all *_fail_cats.jsonl files
#   python tools/calc_fail_cats.py
#
#   # Only specific stems
#   python tools/calc_fail_cats.py --only gemini_gpt4o --only qwen_gpt4o_mini
#
#   # Write JSON and CSV summaries (includes per-stem, per-agent, and optional pooled)
#   python tools/calc_fail_cats.py --out-json assessed_results/final/fail_cats_summary.json \
#                                  --out-csv assessed_results/final/fail_cats_summary.csv
#
#   # Aggregate across all stems into one pooled row (in addition to per-stem)
#   python tools/calc_fail_cats.py --pooled
#
# This script reads assessed_results/final/*_fail_cats.jsonl produced via tools/review_failures.py
# and computes category counts and percentages per stem (and optionally a pooled total).

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Any, Iterable, List
from collections import Counter, defaultdict


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
    # filename format: <stem>_fail_cats.jsonl
    name = p.stem
    if name.endswith("_fail_cats"):
        return name[:-10]
    return name


def agent_from_stem(stem: str) -> str:
    # Expect stems like "gemini_gpt4o_mini" or "qwen_gpt4o"
    if "_" in stem:
        return stem.split("_", 1)[0]
    return stem


def summarize_file(path: Path) -> Dict[str, Any]:
    counts = Counter()
    total = 0
    with_notes = 0
    for rec in read_jsonl(path):
        cat = str(rec.get("category") or "Unknown")
        note = str(rec.get("note") or "").strip()
        counts[cat] += 1
        total += 1
        if note:
            with_notes += 1
    perc = {k: (v / total if total else 0.0) for k, v in counts.items()}
    return {
        "file": str(path),
        "stem": stem_from_filename(path),
        "total_failures": total,
        "categories": dict(counts),
        "percents": perc,
        "with_notes": with_notes,
    }


def write_json(path: Path, obj: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: List[Dict[str, Any]], cat_keys: List[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["stem", "total_failures", "with_notes"] + \
                 [f"count_{k}" for k in cat_keys] + \
                 [f"pct_{k}" for k in cat_keys]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main():
    ap = argparse.ArgumentParser(description="Summarize failure categories from *_fail_cats.jsonl")
    ap.add_argument("--final-dir", default="assessed_results/final", help="Directory containing *_fail_cats.jsonl")
    ap.add_argument("--only", action="append", help="Limit to specific stem(s). Repeatable.")
    ap.add_argument("--out-json", help="Optional JSON path for summary output")
    ap.add_argument("--out-csv", help="Optional CSV path for summary table")
    ap.add_argument("--pooled", action="store_true", help="Add a pooled summary across all stems")
    args = ap.parse_args()

    final_dir = Path(args.final_dir).resolve()
    if not final_dir.exists():
        print(f"[error] not found: {final_dir}")
        return

    files = sorted(final_dir.glob("*_fail_cats.jsonl"))
    if args.only:
        wanted = set(args.only)
        files = [p for p in files if stem_from_filename(p) in wanted]
    if not files:
        print("[warn] no *_fail_cats.jsonl files matched")
        return

    summaries = [summarize_file(p) for p in files]

    # Build a unified set of category keys to align CSV columns
    all_cats = set()
    for s in summaries:
        all_cats.update(s["categories"].keys())
    cat_keys = sorted(all_cats)

    # Console report - per-stem
    print("\nFailure category summary")
    for s in summaries:
        stem = s["stem"]
        tot = s["total_failures"]
        print(f"- {stem}: total={tot} (with_notes={s['with_notes']})")
        for k in cat_keys:
            c = s["categories"].get(k, 0)
            p = s["percents"].get(k, 0.0)
            print(f"    {k:>24}: {c:4d}  ({p:.1%})")

    # Per-agent aggregation
    agents = sorted({agent_from_stem(s["stem"]) for s in summaries})
    by_agent_counts: Dict[str, Counter] = {a: Counter() for a in agents}
    by_agent_total: Dict[str, int] = {a: 0 for a in agents}
    by_agent_notes: Dict[str, int] = {a: 0 for a in agents}
    for s in summaries:
        a = agent_from_stem(s["stem"])
        by_agent_total[a] += s["total_failures"]
        by_agent_notes[a] += s["with_notes"]
        for k, v in s["categories"].items():
            by_agent_counts[a][k] += int(v)

    print("\nPer-agent failure summary")
    by_agent = {}
    for a in agents:
        atot = by_agent_total[a]
        acounts = by_agent_counts[a]
        aperc = {k: (acounts.get(k, 0) / atot if atot else 0.0) for k in cat_keys}
        by_agent[a] = {
            "agent": a,
            "total_failures": atot,
            "categories": dict(acounts),
            "percents": aperc,
            "with_notes": by_agent_notes[a],
        }
        print(f"- {a}: total={atot} (with_notes={by_agent_notes[a]})")
        for k in cat_keys:
            print(f"    {k:>24}: {acounts.get(k,0):4d}  ({aperc[k]:.1%})")

    pooled = None
    if args.pooled:
        pc = Counter()
        ptot = 0
        pnotes = 0
        for s in summaries:
            ptot += s["total_failures"]
            pnotes += s["with_notes"]
            for k, v in s["categories"].items():
                pc[k] += int(v)
        pperc = {k: (pc[k] / ptot if ptot else 0.0) for k in cat_keys}
        pooled = {
            "stem": "__pooled__",
            "total_failures": ptot,
            "categories": dict(pc),
            "percents": pperc,
            "with_notes": pnotes,
        }
        print(f"\nPooled: total={ptot} (with_notes={pnotes})")
        for k in cat_keys:
            print(f"    {k:>24}: {pc[k]:4d}  ({pperc[k]:.1%})")

    if args.out_json:
        out_obj = {"per_stem": summaries, "per_agent": by_agent}
        if pooled:
            out_obj["pooled"] = pooled
        write_json(Path(args.out_json).resolve(), out_obj)
        print(f"[wrote] {args.out_json}")

    if args.out_csv:
        rows = []
        for s in summaries:
            row = {
                "stem": s["stem"],
                "total_failures": s["total_failures"],
                "with_notes": s["with_notes"],
            }
            for k in cat_keys:
                row[f"count_{k}"] = s["categories"].get(k, 0)
            for k in cat_keys:
                row[f"pct_{k}"] = round(s["percents"].get(k, 0.0), 6)
            rows.append(row)
        # add per-agent rows (prefix stem with __agent__)
        for a, rec in by_agent.items():
            row = {
                "stem": f"__agent__:{a}",
                "total_failures": rec["total_failures"],
                "with_notes": rec["with_notes"],
            }
            for k in cat_keys:
                row[f"count_{k}"] = rec["categories"].get(k, 0)
            for k in cat_keys:
                row[f"pct_{k}"] = round(rec["percents"].get(k, 0.0), 6)
            rows.append(row)
        if pooled:
            prow = {
                "stem": pooled["stem"],
                "total_failures": pooled["total_failures"],
                "with_notes": pooled["with_notes"],
            }
            for k in cat_keys:
                prow[f"count_{k}"] = pooled["categories"].get(k, 0)
            for k in cat_keys:
                prow[f"pct_{k}"] = round(pooled["percents"].get(k, 0.0), 6)
            rows.append(prow)
        write_csv(Path(args.out_csv).resolve(), rows, cat_keys)
        print(f"[wrote] {args.out_csv}")


if __name__ == "__main__":
    main()

