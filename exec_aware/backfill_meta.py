#!/usr/bin/env python3
"""Backfill runs/exec/meta.json into execution-aware saved runs that were
produced before the harness wrote meta.json (needed by the judge panel to
phrase the preflight-intent prompt). Derives the realized preflight command
from each run's own init_and_run.sh."""
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "exec_aware"))
from run_exec_aware import parse_preflight  # noqa: E402


def main():
    base = Path(sys.argv[1]) if len(sys.argv) > 1 else (REPO / "saved_runs_execaware")
    subset = {f'{e["technique"]}_{e["test"]}_{e["row_index"]}': e
              for e in json.loads((REPO / "review-stage" / "exec_aware_subset.json").read_text())}
    n = 0
    for meta_dir in base.rglob("runs/exec"):
        run = meta_dir.parent.parent  # .../<rid_short>
        rid_short = run.name
        mf = meta_dir / "meta.json"
        if mf.exists():
            continue
        # preflight from the run's own copied init_and_run.sh
        pf = parse_preflight(run)
        entry = subset.get(rid_short, {})
        cell = run.parent.name
        agent = run.parent.parent.name
        perm, _, tmo = cell.rpartition("_")
        mf.write_text(json.dumps({
            "rid": rid_short,
            "technique": entry.get("technique", rid_short.split("_")[0]),
            "test": entry.get("test", ""),
            "tactic": entry.get("tactic", ""),
            "agent": agent, "generator_model": "gpt4o",
            "payload_command": "; ".join(pf.get("cmds", [])),
            "command": "; ".join(pf.get("cmds", [])),
            "permission": perm, "timeout": tmo.rstrip("s"),
        }))
        n += 1
    print(f"backfilled {n} meta.json under {base}")


if __name__ == "__main__":
    main()
