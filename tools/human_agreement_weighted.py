#!/usr/bin/env python3
"""Judge-vs-human agreement for all four agents on the representative packs, with
stratum weighting and bootstrap CIs (R2.3: validation must cover both verdicts and
account for uncertainty).

Packs   : gemini/qwen  review-stage/rq1_legacy_review_pack  (100 each)
          claude/codex review-stage/rq1_review_pack         (100 each)
Strata  : the judge's CURRENT verdict on each run (post-stratified), weighted to the
          agent's corpus verdict split, so floors/oversampling do not bias the estimates.
Reports : raw kappa/acc; weighted acc, kappa, FPR/FNR, human-corrected ER, each with
          a stratified-bootstrap 95% CI. Writes assessed/human_agreement_weighted.json.
"""
import csv, json, random, sys
from pathlib import Path
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "tools"))
from paper_numbers_sonnet import load_sonnet, run_key, kappa, AGENTS  # noqa

PACKS = {"gemini": "rq1_legacy_review_pack", "qwen": "rq1_legacy_review_pack",
         "claude": "rq1_review_pack", "codex": "rq1_review_pack"}
B, rng = 4000, random.Random(0)

son = load_sonnet()
corpus = {a: [0, 0] for a in AGENTS}            # [judge yes, n]
for leaf, y in son.items():
    a = run_key(leaf)[0]
    corpus[a][0] += y; corpus[a][1] += 1

def pairs(agent):
    d = REPO / "review-stage" / PACKS[agent]
    key = {r["id"]: r for r in csv.DictReader(open(d / "KEY.csv"))}
    out = []
    for r in csv.DictReader(open(d / "labels.csv")):
        h, k = r["label"].strip().lower(), key[r["id"]]
        if k["agent"] == agent and h in ("yes", "no") and k["leaf"] in son:
            out.append((son[k["leaf"]], int(h == "yes")))
    return out

def weighted(p, w1):
    """w1 = corpus share of judge-yes. Returns weighted cell probs and derived stats."""
    s1 = [h for s, h in p if s]; s0 = [h for s, h in p if not s]
    py1 = sum(s1) / len(s1) if s1 else 0.0       # P(human yes | judge yes)
    py0 = sum(s0) / len(s0) if s0 else 0.0       # P(human yes | judge no)
    c11, c10 = w1 * py1, w1 * (1 - py1)          # judge yes & human yes / no
    c01, c00 = (1 - w1) * py0, (1 - w1) * (1 - py0)
    acc = c11 + c00
    hy = c11 + c01
    pe = w1 * hy + (1 - w1) * (1 - hy)
    kap = (acc - pe) / (1 - pe) if pe < 1 else float("nan")
    return dict(acc=acc, kappa=kap, human_er=hy,
                judge_fp=1 - py1,                # share of judge-yes the human calls no
                judge_fn=py0)                    # share of judge-no the human calls yes

def ci(vals):
    v = sorted(x for x in vals if x == x)
    return [round(100 * v[int(0.025 * len(v))], 1), round(100 * v[int(0.975 * len(v)) - 1], 1)]

res, pooled = {}, []
for a in AGENTS:
    p = pairs(a); pooled += p
    w1 = corpus[a][0] / corpus[a][1]
    k, acc, n = kappa(p)
    pt = weighted(p, w1)
    s1 = [x for x in p if x[0]]; s0 = [x for x in p if not x[0]]
    boots = [weighted([rng.choice(s1) for _ in s1] + [rng.choice(s0) for _ in s0], w1) for _ in range(B)]
    res[a] = dict(n=n, n_judge_yes=len(s1), n_judge_no=len(s0),
                  cm=dict(S1H1=sum(h for _, h in s1), S1H0=sum(1 - h for _, h in s1),
                          S0H1=sum(h for _, h in s0), S0H0=sum(1 - h for _, h in s0)),
                  raw_kappa=round(k, 3), raw_acc=round(100 * acc, 1),
                  judge_er=round(100 * w1, 1),
                  **{f"w_{m}": round(100 * pt[m], 1) if m != "kappa" else round(pt[m], 3) for m in pt},
                  **{f"w_{m}_ci": ci([b[m] for b in boots]) if m != "kappa"
                     else [round(x / 100, 3) for x in ci([b[m] for b in boots])] for m in pt})
    print(a, json.dumps(res[a]))
k, acc, n = kappa(pooled)
res["pooled_raw"] = dict(n=n, kappa=round(k, 3), acc=round(100 * acc, 1))
print("pooled", res["pooled_raw"])
json.dump(res, open(REPO / "assessed/human_agreement_weighted.json", "w"), indent=1)
