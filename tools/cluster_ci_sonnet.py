#!/usr/bin/env python3
"""Cluster-bootstrap CIs (cluster = ART entry / payload) and per-payload agreement for RQ1."""
import sys, json, random
from collections import defaultdict
from pathlib import Path
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "tools"))
from paper_numbers_sonnet import load_sonnet, run_key, wilson, AGENTS  # noqa

random.seed(0)
B = 2000
son = load_sonnet()
by = defaultdict(lambda: defaultdict(list))   # agent -> payload(rid) -> [labels]
for leaf, y in son.items():
    agent, gen, rid = run_key(leaf)
    by[agent][rid].append(y)

out = {}
for agent in AGENTS:
    pl = by[agent]
    rids = list(pl)
    runs = [y for r in rids for y in pl[r]]
    K, N = sum(runs), len(runs)
    lo, hi = wilson(K, N); lo /= 100; hi /= 100   # wilson() already returns percent
    # cluster bootstrap over payloads
    bs = []
    for _ in range(B):
        samp = [pl[random.choice(rids)] for _ in rids]
        k = sum(sum(s) for s in samp); n = sum(len(s) for s in samp)
        bs.append(k / n)
    bs.sort()
    blo, bhi = bs[int(0.025 * B)], bs[int(0.975 * B) - 1]
    # per-payload agreement (only payloads with all runs completed, i.e. 6 runs)
    full = [pl[r] for r in rids if len(pl[r]) == 6]
    all6 = sum(sum(s) == 6 for s in full); none = sum(sum(s) == 0 for s in full); mixed = len(full) - all6 - none
    # ICC(1) one-way ANOVA
    groups = [pl[r] for r in rids if len(pl[r]) >= 2]
    gm = sum(sum(g) for g in groups) / sum(len(g) for g in groups)
    k_bar = sum(len(g) for g in groups) / len(groups)
    msb = sum(len(g) * (sum(g) / len(g) - gm) ** 2 for g in groups) / (len(groups) - 1)
    msw = sum(sum((y - sum(g) / len(g)) ** 2 for y in g) for g in groups) / (sum(len(g) for g in groups) - len(groups))
    icc = (msb - msw) / (msb + (k_bar - 1) * msw)
    out[agent] = dict(n_runs=N, n_payloads=len(rids), er=round(100 * K / N, 1),
                      wilson=[round(100 * lo, 1), round(100 * hi, 1)],
                      cluster_boot=[round(100 * blo, 1), round(100 * bhi, 1)],
                      width_wilson=round(100 * (hi - lo), 1), width_boot=round(100 * (bhi - blo), 1),
                      payloads_with_6=len(full), all6=all6, none=none, mixed=mixed,
                      pct_all6=round(100 * all6 / len(full), 1), pct_none=round(100 * none / len(full), 1),
                      pct_mixed=round(100 * mixed / len(full), 1), icc=round(icc, 3))
    print(agent, json.dumps(out[agent]))
json.dump(out, open(REPO / "assessed/cluster_ci_sonnet.json", "w"), indent=1)
