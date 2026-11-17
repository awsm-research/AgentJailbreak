#!/usr/bin/env python3
"""RQ2 failure-mode numbers for the Sonnet-judged, four-agent revision.

Failure set (same definition as RQ1): completed run, Sonnet v2 / legacy Sonnet judge says not
exploited, and (Claude Code, Codex CLI only) the strace record shows no preflight launch.

Gemini CLI, Qwen Code : census. Labels = assessed/rq2_labels.csv (earlier human labels) plus
                        review-stage/rq2_review_pack for failures that had none.
Claude Code, Codex CLI: stratified random sample from review-stage/rq2_review_pack
                        (strata: agent touched / never touched the skill file or preflight in its
                        tool calls). Shares are weighted by stratum size, CIs use the stratified
                        variance with finite-population correction.

Runs the human marked ActuallyExploited are judge misses: they are reported separately and
removed from the failure set before the category shares are computed.
Writes assessed/rq2_numbers_sonnet.json.
"""
import csv, json, math, sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from build_rq2_review_pack import sonnet_labels, scan, LEGACY_SONNET, LEGACY_LABELS, PACK
from strace_exec import executed

REPO = Path(__file__).resolve().parent.parent
CATS = ["RecognizedAttack", "IrrelevantToTask", "AcknowledgedButDidNotExecute", "NoAcknowledgment"]
MISS = "ActuallyExploited"

def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(max(0, c - h) * 100, 1), round(min(1, c + h) * 100, 1))

def pack():
    key = {r["id"]: r for r in csv.DictReader(open(PACK / "KEY.csv"))}
    lab = {r["id"]: r["label"] for r in csv.DictReader(open(PACK / "labels.csv"))}
    return [(key[i], lab[i]) for i in key if lab.get(i)]

def legacy():
    old = {}
    for r in csv.DictReader(open(LEGACY_LABELS)):
        base, _, rundir = r["key"].split("::")
        old[(base.replace("saved_runs_", ""), rundir)] = r["label"]
    new = {k["leaf"]: l for k, l in pack() if k["stratum"] == "legacy_unlabelled"}
    out = {}
    for agent in ("gemini", "qwen"):
        labels, missing = Counter(), 0
        for fp in sorted(LEGACY_SONNET.glob(f"legacy_{agent}_*.sonnet.cache")):
            for line in open(fp):
                if not line.strip():
                    continue
                o = json.loads(line)
                if not o["result"].get("parse_ok") or o["result"]["success"]:
                    continue
                p = Path(o["leaf"])
                l = old.get((p.parts[-4].replace("saved_runs_", ""), p.parts[-3])) or new.get(o["leaf"])
                if l is None:
                    missing += 1
                else:
                    labels[l] += 1
        judged_fail = sum(labels.values()) + missing
        fail = {c: labels[c] for c in CATS}
        n = sum(fail.values())
        out[agent] = {"design": "census", "judge_failures": judged_fail, "unlabelled": missing,
                      "judge_miss": labels[MISS], "other_labels": {k: v for k, v in labels.items()
                                                                if k not in CATS and k != MISS},
                      "failures": n,
                      "cats": {c: {"k": fail[c], "pct": round(fail[c] / n * 100, 1), "ci": wilson(fail[c], n)}
                               for c in CATS}}
    return out

def sampled():
    son = sonnet_labels()
    pop = defaultdict(Counter)                    # agent -> stratum -> N (strace failure set)
    frame_gap = Counter()
    for leaf, (base, exploited) in son.items():
        agent = base.split("_")[0]
        if exploited or executed(leaf):
            continue
        scan_exec, touched = scan(leaf, agent)
        if scan_exec:
            frame_gap[agent] += 1                 # strace failure, but outside the sampling frame
        pop[agent]["touched" if touched else "untouched"] += 1
    out = {}
    items = pack()
    for agent in ("claude", "codex"):
        rows = [(k, l) for k, l in items if k["agent"] == agent]
        dropped = [(k, l) for k, l in rows if executed(k["leaf"])]
        rows = [(k, l) for k, l in rows if not executed(k["leaf"])]
        by_s = defaultdict(list)
        for k, l in rows:
            by_s[k["stratum"]].append(l)
        Ns = pop[agent]
        N = sum(Ns.values())

        def est(pred):
            """weighted share over the failure population, stratified SE with FPC."""
            p = var = 0.0
            for s, labs in by_s.items():
                W, n = Ns[s] / N, len(labs)
                ps = sum(map(pred, labs)) / n
                p += W * ps
                if n > 1:
                    var += W * W * (1 - n / Ns[s]) * ps * (1 - ps) / (n - 1)
            return p, math.sqrt(var)

        miss, miss_se = est(lambda l: l == MISS)
        # category shares among confirmed failures: ratio estimator (cat) / (not a judge miss)
        keep, _ = est(lambda l: l != MISS)
        cats = {}
        for c in CATS:
            pc, se = est(lambda l, c=c: l == c)
            r, rse = pc / keep, se / keep
            raw = sum(1 for _, l in rows if l == c)
            ci = (round(max(0, r - 1.96 * rse) * 100, 1), round(min(1, r + 1.96 * rse) * 100, 1))
            if raw == 0:                          # normal interval collapses at zero, use Wilson
                ci = wilson(0, sum(1 for _, l in rows if l != MISS))
            cats[c] = {"k_sample": raw, "pct": round(r * 100, 1), "ci": ci}
        out[agent] = {"design": "stratified sample", "judge_failures": N, "strata_N": dict(Ns),
                      "strata_n": {s: len(v) for s, v in by_s.items()},
                      "outside_frame": frame_gap[agent], "dropped_strace_executed": len(dropped),
                      "sample_n": len(rows), "judge_miss_sample": sum(1 for _, l in rows if l == MISS),
                      "judge_miss_pct": round(miss * 100, 1),
                      "est_failures": round(N * keep),
                      "cats": cats}
    return out

def main():
    res = {**legacy(), **sampled()}
    total_runs = {"gemini": 2816, "qwen": 2813, "claude": 2826, "codex": 2826}
    for a, r in res.items():
        nf = r.get("failures", r.get("est_failures"))
        r["failure_pct_of_runs"] = round(nf / total_runs[a] * 100, 1)
        ra = r["cats"]["RecognizedAttack"]["pct"]
        r["recognized_pct_of_runs"] = round(ra / 100 * nf / total_runs[a] * 100, 2)
    rec = sum(r["cats"]["RecognizedAttack"]["pct"] / 100 * r.get("failures", r.get("est_failures"))
              for r in res.values())
    res["_all"] = {"runs": sum(total_runs.values()), "recognized_est": round(rec),
                   "recognized_pct_of_runs": round(rec / sum(total_runs.values()) * 100, 2)}
    (REPO / "assessed/rq2_numbers_sonnet.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))

if __name__ == "__main__":
    main()
