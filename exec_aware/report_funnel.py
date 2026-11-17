#!/usr/bin/env python3
"""
Build the execution-aware funnel report for the TOSEM revision.

Reads results_execaware_<agent>.jsonl (strace-based invocation/execution +
termination) and, if available, the judge-panel per-run labels (authoritative
intent) produced by tools/assess_preflight_stdout_multi_llm.py. Emits, per 2x2
cell (permission x timeout): n, and the intent -> invocation -> execution
funnel with Wilson 95% CIs, plus per-tactic rates and the termination-reason
distribution.

Intent source:
  - panel  : majority-vote + declared-intent override (matches the paper), if
             an assessed jsonl is supplied via --panel-dir;
  - regex  : the conservative transcript regex captured at run time (fallback).
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "tools"))
from assess_preflight_stdout_multi_llm import wilson_ci  # noqa: E402

CELLS = ["auto_120s", "auto_600s", "restricted_120s", "restricted_600s"]


def pct_ci(k, n):
    if n == 0:
        return "   n/a      "
    lo, hi = wilson_ci(k, n)
    return f"{100*k/n:5.1f} [{100*lo:4.1f},{100*hi:4.1f}]"


def load_panel(panel_dir: Path):
    """Map rid_short -> intent(bool) from assessed per-cell jsonl files.
    Each assessed record has rid/leaf + majority_success + declared_intent_override."""
    intent = {}
    if not panel_dir:
        return intent
    for jf in panel_dir.rglob("*.jsonl"):
        if jf.name.endswith(".cache"):
            continue
        for line in jf.read_text(errors="ignore").splitlines():
            try:
                d = json.loads(line)
            except Exception:
                continue
            rid = d.get("rid") or (d.get("meta") or {}).get("rid")
            if not rid:
                continue
            val = bool(d.get("majority_success")) or bool(d.get("declared_intent_override"))
            # cell is encoded in the path of this assessed file
            cell = next((c for c in CELLS if c in str(jf)), "")
            intent[(cell, rid)] = val
    return intent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True, help="results_execaware_<agent>.jsonl")
    ap.add_argument("--panel-dir", default=None, help="dir of assessed per-cell jsonl (authoritative intent)")
    ap.add_argument("--out", default=None, help="write JSON summary here")
    args = ap.parse_args()

    rows = [json.loads(l) for l in Path(args.results).read_text().splitlines() if l.strip()]
    panel = load_panel(Path(args.panel_dir)) if args.panel_dir else {}
    intent_src = "panel" if panel else "regex"

    by_cell = defaultdict(list)
    for r in rows:
        by_cell[r["cell"]].append(r)

    def intent_of(r):
        key = (r["cell"], r["rid"])
        if key in panel:
            return panel[key]
        return bool(r.get("intent_regex"))

    print(f"\n=== EXECUTION-AWARE FUNNEL  (agent={rows[0]['agent'] if rows else '?'}, "
          f"generator=gpt4o, intent_source={intent_src}) ===\n")
    header = f'{"cell":16s} {"n":>3s}  {"intent %[95% CI]":>20s}  {"invocation":>20s}  {"execution":>20s}'
    print(header)
    print("-" * len(header))
    summary = {"intent_source": intent_src, "cells": {}}
    for cell in CELLS:
        rs = [r for r in by_cell.get(cell, []) if r.get("run_category") == "completed"]
        n = len(rs)
        ki = sum(1 for r in rs if intent_of(r))
        kv = sum(1 for r in rs if r.get("invocation"))
        ke = sum(1 for r in rs if r.get("execution"))
        print(f'{cell:16s} {n:3d}  {pct_ci(ki,n):>20s}  {pct_ci(kv,n):>20s}  {pct_ci(ke,n):>20s}')
        summary["cells"][cell] = {"n": n, "intent": ki, "invocation": kv, "execution": ke}

    # termination-reason distribution per cell
    print("\n--- termination reason (all runs incl. non-completed) ---")
    terms = defaultdict(lambda: defaultdict(int))
    for r in rows:
        terms[r["cell"]][r.get("termination", "?")] += 1
    allterms = sorted({t for c in terms.values() for t in c})
    print(f'{"cell":16s} ' + " ".join(f"{t:>12s}" for t in allterms))
    for cell in CELLS:
        print(f'{cell:16s} ' + " ".join(f"{terms[cell].get(t,0):12d}" for t in allterms))
    summary["termination"] = {c: dict(terms[c]) for c in terms}

    # per-tactic invocation/execution pooled over auto cells (where execution is possible)
    print("\n--- per-tactic (auto cells pooled): invocation / execution ---")
    tac = defaultdict(lambda: {"n": 0, "inv": 0, "exe": 0})
    for r in rows:
        if r.get("permission") == "auto" and r.get("run_category") == "completed":
            t = tac[r.get("tactic", "?")]
            t["n"] += 1
            t["inv"] += 1 if r.get("invocation") else 0
            t["exe"] += 1 if r.get("execution") else 0
    print(f'{"tactic":22s} {"n":>3s} {"invocation":>18s} {"execution":>18s}')
    for t in sorted(tac):
        d = tac[t]
        print(f'{t:22s} {d["n"]:3d} {pct_ci(d["inv"],d["n"]):>18s} {pct_ci(d["exe"],d["n"]):>18s}')
    summary["per_tactic_auto"] = dict(tac)

    if args.out:
        Path(args.out).write_text(json.dumps(summary, indent=1))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
