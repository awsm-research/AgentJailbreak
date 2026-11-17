#!/usr/bin/env python3
"""Paired comparison of agents on the same skill files (generator x ART entry).

For every pair of agents: discordant counts, exact McNemar p, and the paired
ER difference with a payload-level cluster-bootstrap 95% CI (resample ART entries).
Writes assessed/paired_agents_sonnet.json.
"""
import sys, json, random
from itertools import combinations
from math import comb
from collections import defaultdict
from pathlib import Path
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "tools"))
from paper_numbers_sonnet import load_sonnet, run_key, AGENTS  # noqa

random.seed(0)
B = 2000
lab = defaultdict(dict)            # agent -> (gen, rid) -> label
for leaf, y in load_sonnet().items():
    agent, gen, rid = run_key(leaf)
    lab[agent][(gen, rid)] = y

def mcnemar(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)

out = {}
for a1, a2 in combinations(AGENTS, 2):
    keys = sorted(set(lab[a1]) & set(lab[a2]))
    by_rid = defaultdict(list)
    for k in keys:
        by_rid[k[1]].append((lab[a1][k], lab[a2][k]))
    b = sum(1 for k in keys if lab[a1][k] and not lab[a2][k])
    c = sum(1 for k in keys if lab[a2][k] and not lab[a1][k])
    n = len(keys)
    diff = (sum(lab[a1][k] for k in keys) - sum(lab[a2][k] for k in keys)) / n
    rids = list(by_rid)
    bs = []
    for _ in range(B):
        s = [by_rid[random.choice(rids)] for _ in rids]
        tot = sum(len(x) for x in s)
        bs.append(sum(p[0] - p[1] for x in s for p in x) / tot)
    bs.sort()
    out[f"{a1}_vs_{a2}"] = dict(n_pairs=n, n_payloads=len(rids), a1_only=b, a2_only=c,
                              mcnemar_p=mcnemar(b, c), diff_pts=round(100 * diff, 1),
                              diff_ci=[round(100 * bs[int(0.025 * B)], 1), round(100 * bs[int(0.975 * B) - 1], 1)])
    print(a1, "vs", a2, json.dumps(out[f"{a1}_vs_{a2}"]))
json.dump(out, open(REPO / "assessed/paired_agents_sonnet.json", "w"), indent=1)
