#!/usr/bin/env python3
"""Build a blind, randomised, stratified RQ1 human-labelling pack (exploited yes/no)
for Gemini CLI + Qwen Code, mirroring tools/build_rq1_review_pack.py (Claude + Codex).

Population : every completed Gemini CLI / Qwen Code legacy run with a Sonnet verdict
             (assessed/panel_final/legacy_*.sonnet.cache, the labels the paper reports),
             minus the 240 runs already in the hard-case gold sample (assessed/gold_splits.csv).
Strata     : agent x Sonnet verdict (exploited / not exploited).
Allocation : proportional to Sonnet's verdict split within each agent, with a floor
             per stratum, same rule as the Claude/Codex pack.
Labeller   : python3 tools/review_label.py rq1legacy

Outputs (review-stage/rq1_legacy_review_pack/):
  transcripts/G###.txt   stdout.log, the same text the judge saw
  items.csv              id + planted command shown to the labeller
  label_config.json      keys for the labeller (identical to the rq1 pack)
  KEY.csv                HIDDEN: agent, base, stratum, sonnet verdict, rid, leaf
  INSTRUCTIONS.md

  python3 tools/build_rq1_legacy_review_pack.py [--per-agent 100] [--floor 20] [--seed 2028]
"""
import argparse, csv, glob, json, random, shutil, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from build_rq1_review_pack import LABEL_CONFIG, INSTRUCTIONS   # same definitions as Claude/Codex

REPO = Path(__file__).resolve().parent.parent
CACHES = REPO / "assessed/panel_final"
GOLD = REPO / "assessed/gold_splits.csv"
PACK = REPO / "review-stage/rq1_legacy_review_pack"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-agent", type=int, default=100)
    ap.add_argument("--floor", type=int, default=20)
    ap.add_argument("--seed", type=int, default=2028)
    ap.add_argument("--force", action="store_true", help="overwrite an existing pack (labels too)")
    a = ap.parse_args()
    if (PACK / "labels.csv").exists() and not a.force:
        raise SystemExit(f"{PACK}/labels.csv exists. Use --force to rebuild (this discards labels).")

    rng = random.Random(a.seed)
    # gold_splits key = <agent>_<gen>::<rid>::<run dir name>; match on the run dir name
    with open(GOLD) as f:
        seen = {r["key"].split("::")[-1] for r in csv.DictReader(f)}

    pop = {}
    for fp in glob.glob(str(CACHES / "legacy_*.sonnet.cache")):
        base = Path(fp).name.removeprefix("legacy_").removesuffix(".sonnet.cache")
        agent = base.split("_")[0]
        for line in open(fp):
            if not line.strip():
                continue
            o = json.loads(line)
            if not o["result"].get("parse_ok") or Path(o["leaf"]).name in seen:
                continue
            stratum = "sonnet_yes" if o["result"]["success"] else "sonnet_no"
            pop.setdefault(agent, {}).setdefault(stratum, {})[o["leaf"]] = base   # dedupe by leaf

    picked = []
    for agent in ("gemini", "qwen"):
        strata = {s: sorted(v.items()) for s, v in pop[agent].items()}
        total = sum(len(v) for v in strata.values())
        n = min(a.per_agent, total)
        alloc = {s: max(min(a.floor, len(v)), round(n * len(v) / total)) for s, v in strata.items()}
        while sum(alloc.values()) > n:
            s = max(alloc, key=lambda k: alloc[k] - min(a.floor, len(strata[k])))
            alloc[s] -= 1
        while sum(alloc.values()) < n:
            s = max(strata, key=lambda k: len(strata[k]) - alloc[k])
            alloc[s] += 1
        print(f"{agent}: population {total}  " +
              "  ".join(f"{s}={len(v)} ({len(v)/total*100:.1f}%) -> {alloc[s]}" for s, v in sorted(strata.items())))
        for s, items in sorted(strata.items()):
            for leaf, base in rng.sample(items, alloc[s]):
                picked.append((agent, base, s, leaf))

    rng.shuffle(picked)
    if PACK.exists():
        shutil.rmtree(PACK)
    (PACK / "transcripts").mkdir(parents=True)
    with open(PACK / "KEY.csv", "w", newline="") as fk, open(PACK / "items.csv", "w", newline="") as fi:
        wk, wi = csv.writer(fk), csv.writer(fi)
        wk.writerow(["id", "agent", "base", "stratum", "sonnet", "rid", "technique", "leaf"])
        wi.writerow(["id", "planted_command"])
        for i, (agent, base, stratum, leaf) in enumerate(picked, 1):
            gid = f"G{i:03d}"
            lp = Path(leaf)
            (PACK / "transcripts" / f"{gid}.txt").write_text((lp / "stdout.log").read_text(errors="ignore"))
            meta = json.loads((lp / "meta.json").read_text()) if (lp / "meta.json").exists() else {}
            wk.writerow([gid, agent, base, stratum, int(stratum == "sonnet_yes"),
                         meta.get("rid", lp.name), meta.get("technique", ""), leaf])
            wi.writerow([gid, meta.get("payload_command") or meta.get("command") or ""])
    (PACK / "label_config.json").write_text(json.dumps(LABEL_CONFIG, indent=1))
    (PACK / "INSTRUCTIONS.md").write_text(
        INSTRUCTIONS.replace("Claude Code + Codex", "Gemini CLI + Qwen Code").replace(" rq1`", " rq1legacy`"))
    print(f"\nwrote {len(picked)} items -> {PACK}")
    print("label with:  python3 tools/review_label.py rq1legacy")


if __name__ == "__main__":
    main()
