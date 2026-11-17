#!/usr/bin/env python3
"""Rebuild results_execaware_<agent>.jsonl from the SAVED strace logs, applying
the current funnel detector. Decouples detection from the (expensive) agent
runs: edit analyze() and re-run this, no containers needed. Source of truth =
saved_runs_execaware/<agent>/<cell>/<rid>/runs/exec/{stdout.log,execve.log,
exitcode,meta.json}."""
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "exec_aware"))
sys.path.insert(0, str(REPO / "tools"))
from run_exec_aware import analyze, parse_preflight  # noqa: E402
from assess_preflight_stdout_multi_llm import classify_transcript  # noqa: E402


def main():
    agent = sys.argv[1] if len(sys.argv) > 1 else "codex"
    base = REPO / "saved_runs_execaware" / agent
    out = REPO / f"results_execaware_{agent}.jsonl"
    n = 0
    with open(out, "w", encoding="utf-8") as rf:
        for cell_dir in sorted(p for p in base.iterdir() if p.is_dir()):
            cell = cell_dir.name
            perm, _, tmo = cell.rpartition("_")
            for run in sorted(p for p in cell_dir.iterdir() if p.is_dir()):
                logs = run / "runs" / "exec"
                if not (logs / "stdout.log").exists() and not (logs / "exitcode").exists():
                    continue
                try:
                    ec = int((logs / "exitcode").read_text().strip())
                except Exception:
                    ec = -1
                meta = {}
                if (logs / "meta.json").exists():
                    try:
                        meta = json.loads((logs / "meta.json").read_text())
                    except Exception:
                        pass
                stdout = (logs / "stdout.log").read_text(errors="ignore") if (logs / "stdout.log").exists() else ""
                cat = classify_transcript(stdout)
                pf = parse_preflight(run)
                info = analyze(logs, pf, ec)
                rec = {
                    "rid": run.name, "agent": agent, "generator": "gpt4o",
                    "tactic": meta.get("tactic", ""), "technique": meta.get("technique", run.name.split("_")[0]),
                    "permission": perm, "timeout": int(tmo.rstrip("s")) if tmo.rstrip("s").isdigit() else tmo,
                    "cell": cell, "run_category": cat, "saved_dir": str(run), **info,
                }
                rf.write(json.dumps(rec) + "\n")
                n += 1
    print(f"reanalyzed {n} saved runs -> {out}")


if __name__ == "__main__":
    main()
