#!/usr/bin/env python3
"""Every number the paper reports for the Sonnet-judged, four-agent revision.

Judge     : claude-sonnet-5-5, one prompt, hint "./init_and_run.sh" for all agents
            legacy  -> assessed/panel_final/legacy_<gen-agent>.sonnet.cache
            new     -> assessed/panel_final_narr_sonnet_v2/<agent>_<gen>.sonnet.cache
Execution : per-run strace (runs/exec/execve.log), Claude Code + Codex only
Humans    : legacy  assessed/gold_splits.csv (panel-disagreement sample)
            new     review-stage/human_review_pack (narrative_label_sheet.csv)
                    review-stage/rq1_review_pack   (labels.csv, proportional sample)
Old panel : assessed/final_labels_v3.jsonl field "panel" (published raw majority vote)

Writes assessed/paper_numbers_sonnet.json and prints the tables.
"""
import csv, glob, json, math, sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "tools"))
from strace_exec import executed

GENS = ["gpt4.1_mini", "gpt4o_mini", "gpt4o", "gpt5_mini", "llama3", "deepseek_r1"]
AGENTS = ["gemini", "qwen", "claude", "codex"]

def wilson(k, n, z=1.96):
    if not n:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round((c - h) * 100, 1), round((c + h) * 100, 1))

def kappa(pairs):
    n = len(pairs)
    a = sum(x == y for x, y in pairs) / n
    px = sum(x for x, _ in pairs) / n
    py = sum(y for _, y in pairs) / n
    pe = px * py + (1 - px) * (1 - py)
    return ((a - pe) / (1 - pe) if pe < 1 else float("nan")), a, n

def run_key(leaf):
    """(agent, generator, run-id-with-row) for any leaf path."""
    p = Path(leaf)
    if "saved_runs_streamjson" in p.parts:
        i = p.parts.index("saved_runs_streamjson")
        return p.parts[i + 2], p.parts[i + 1], p.parts[i + 4]
    base = p.parts[-4].replace("saved_runs_", "")          # gemini_gpt4o / gemini_deepseek_r1
    agent, gen = base.split("_", 1)
    rid = p.parts[-3].split("_", 1)[1]                     # drop the timestamp prefix
    return agent, gen, rid

def load_sonnet():
    lab = {}
    files = glob.glob(str(REPO / "assessed/panel_final/legacy_*.sonnet.cache")) + \
            glob.glob(str(REPO / "assessed/panel_final_narr_sonnet_v2/*.sonnet.cache"))
    for fp in files:
        for line in open(fp):
            if line.strip():
                o = json.loads(line)
                if o["result"].get("parse_ok"):
                    lab[o["leaf"]] = int(bool(o["result"]["success"]))
    return lab

def tactic_rows():
    with open(REPO / "linux-index-with-commands.csv", newline="", errors="ignore") as f:
        return [r["Tactic"].strip() for r in csv.DictReader(f)]

def main():
    son = load_sonnet()
    tactics = tactic_rows()
    out = {"rq1": {}, "rq3": {}, "human": {}, "panel_vs_sonnet": {}}

    # ---------------- RQ1: intent ER per agent x generator, execution for new agents
    cells = defaultdict(lambda: [0, 0])
    exe = defaultdict(lambda: [0, 0, 0])           # executed, either, n   (new agents only)
    run_tactic = []
    for leaf, y in son.items():
        agent, gen, rid = run_key(leaf)
        cells[(agent, gen)][0] += y
        cells[(agent, gen)][1] += 1
        tac = tactics[int(rid.rsplit("_", 1)[1])]
        e = None
        if agent in ("claude", "codex"):
            e = int(bool(executed(leaf)))
            for g in (gen, "all"):
                exe[(agent, g)][0] += e
                exe[(agent, g)][1] += int(e or y)
                exe[(agent, g)][2] += 1
        run_tactic.append((agent, tac, y, e))
    for agent in AGENTS:
        rows = {}
        K = N = 0
        for g in GENS:
            k, n = cells[(agent, g)]
            K += k; N += n
            r = {"k": k, "n": n, "er": round(k / n * 100, 1)}
            if agent in ("claude", "codex"):
                ek, eik, en = exe[(agent, g)]
                r.update(exec=round(ek / en * 100, 1), either=round(eik / en * 100, 1))
            rows[g] = r
        allr = {"k": K, "n": N, "er": round(K / N * 100, 1), "ci": wilson(K, N)}
        if agent in ("claude", "codex"):
            ek, eik, en = exe[(agent, "all")]
            allr.update(exec=round(ek / en * 100, 1), exec_k=ek, exec_ci=wilson(ek, en),
                        either=round(eik / en * 100, 1), either_k=eik, either_ci=wilson(eik, en))
        rows["all"] = allr
        out["rq1"][agent] = rows

    # intent x execution decomposition
    out["rq1"]["decomp"] = {}
    for agent in ("claude", "codex"):
        c = defaultdict(int)
        for a, _, y, e in run_tactic:
            if a == agent:
                c[(y, e)] += 1
        n = sum(c.values())
        out["rq1"]["decomp"][agent] = {k: round(v / n * 100, 1) for k, v in
                                       {"both": c[(1, 1)], "intent_only": c[(1, 0)],
                                        "exec_only": c[(0, 1)], "neither": c[(0, 0)]}.items()}

    # ---------------- RQ3: tactic x agent (intent), pooled over the four agents
    t = defaultdict(lambda: [0, 0])
    for agent, tac, y, e in run_tactic:
        for key in ((tac, agent), (tac, "pooled")):
            t[key][0] += y
            t[key][1] += 1
    for tac in sorted({x[1] for x in run_tactic}):
        k, n = t[(tac, "pooled")]
        row = {"k": k, "n": n, "er": round(k / n * 100, 1), "ci": wilson(k, n)}
        for agent in AGENTS:
            ka, na = t[(tac, agent)]
            row[agent] = round(ka / na * 100, 1)
        exk = sum(1 for a, tc, y, e in run_tactic if a == "codex" and tc == tac and (e or y))
        exn = sum(1 for a, tc, *_ in run_tactic if a == "codex" and tc == tac)
        row["codex_either"] = round(exk / exn * 100, 1)
        out["rq3"][tac] = row
    tacs = list(out["rq3"])
    out["rq3"]["_mean_per_tactic"] = {
        agent: round(sum(out["rq3"][tc][agent] for tc in tacs) / len(tacs), 1) for agent in AGENTS}

    # ---------------- human agreement
    old = {}
    for line in open(REPO / "assessed/final_labels_v3.jsonl"):
        o = json.loads(line)
        old[o["key"]] = int(bool(o["panel"]))
    by_rk = {}
    for leaf, y in son.items():
        by_rk[run_key(leaf)] = y
    # legacy gold (panel-disagreement sample + unanimous spot check)
    paper_pairs = []                     # pairs behind Table rq1_reliability, pooled for the Overall column
    for agent in ("gemini", "qwen"):
        sp, pp = [], []
        for r in csv.DictReader(open(REPO / "assessed/gold_splits.csv")):
            if r["agent"] != agent or r["human"].strip().lower() not in ("t", "f"):
                continue
            base, rid, rundir = r["key"].split("::")
            gen = base.replace("saved_runs_", "").split("_", 1)[1]
            rk = (agent, gen, rundir.split("_", 1)[1])
            h = int(r["human"].strip().lower() == "t")
            if rk in by_rk:
                sp.append((by_rk[rk], h))
            if r["key"] in old:
                pp.append((old[r["key"]], h))
        paper_pairs += sp
        ks, acs, ns = kappa(sp)
        kp, acp, npp = kappa(pp)
        out["human"][agent] = {"sample": "legacy gold_splits", "n": ns, "kappa": round(ks, 3), "acc": round(acs * 100, 1),
                               "panel_kappa_same_sample": round(kp, 3), "panel_acc": round(acp * 100, 1), "panel_n": npp}
    # new agents: two packs
    packs = {
        "disagreement_pack": (REPO / "review-stage/human_review_pack/KEY.csv",
                              REPO / "review-stage/human_review_pack/narrative_label_sheet.csv", "human_label"),
        "proportional_pack": (REPO / "review-stage/rq1_review_pack/KEY.csv",
                              REPO / "review-stage/rq1_review_pack/labels.csv", "label"),
    }
    for agent in ("claude", "codex"):
        out["human"][agent] = {}
        allp = []
        for name, (kf, lf, col) in packs.items():
            key = {r["id"]: r for r in csv.DictReader(open(kf))}
            p = []
            for r in csv.DictReader(open(lf)):
                h = r[col].strip().lower()
                k = key[r["id"]]
                if k["agent"] != agent or h not in ("yes", "no") or k["leaf"] not in son:
                    continue
                p.append((son[k["leaf"]], int(h == "yes")))
            kk, ac, n = kappa(p)
            hy = sum(h for _, h in p)
            cm = {"S1H1": sum(1 for s, h in p if s and h), "S1H0": sum(1 for s, h in p if s and not h),
                  "S0H1": sum(1 for s, h in p if not s and h), "S0H0": sum(1 for s, h in p if not s and not h)}
            out["human"][agent][name] = {"n": n, "kappa": round(kk, 3), "acc": round(ac * 100, 1),
                                         "human_er": round(hy / n * 100, 1), "human_ci": wilson(hy, n),
                                         "sonnet_er_sample": round(sum(s for s, _ in p) / n * 100, 1), "cm": cm}
            allp += p
            if name == "proportional_pack":
                paper_pairs += p
        kk, ac, n = kappa(allp)
        out["human"][agent]["combined"] = {"n": n, "kappa": round(kk, 3), "acc": round(ac * 100, 1)}

    kk, ac, n = kappa(paper_pairs)
    K = sum(out["rq1"][a]["all"]["k"] for a in AGENTS)
    N = sum(out["rq1"][a]["all"]["n"] for a in AGENTS)
    out["human"]["overall"] = {"n": n, "kappa": round(kk, 3), "acc": round(ac * 100, 1),
                               "corpus_er": round(K / N * 100, 1), "corpus_k": K, "corpus_n": N}

    # ---------------- Sonnet vs published panel on the full legacy corpus
    for agent in ("gemini", "qwen"):
        p = []
        for key, v in old.items():
            if not key.startswith(agent):
                continue
            base, rid, rundir = key.split("::")
            rk = (agent, base.replace("saved_runs_", "").split("_", 1)[1], rundir.split("_", 1)[1])
            if rk in by_rk:
                p.append((by_rk[rk], v))
        kk, ac, n = kappa(p)
        out["panel_vs_sonnet"][agent] = {"n": n, "kappa": round(kk, 3), "agree": round(ac * 100, 1),
                                         "panel_er": round(sum(v for _, v in p) / n * 100, 1)}

    (REPO / "assessed/paper_numbers_sonnet.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))

if __name__ == "__main__":
    main()
