#!/usr/bin/env python3
"""RQ1 and RQ3 figures for the Sonnet-judged, four-agent revision.
Reads assessed/paper_numbers_sonnet.json (tools/paper_numbers_sonnet.py)."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parent.parent
FIGDIR = REPO / "figures"
D = json.loads((REPO / "assessed/paper_numbers_sonnet.json").read_text())

AGENTS = ["gemini", "qwen", "claude", "codex"]
NAMES = {"gemini": "Gemini CLI", "qwen": "Qwen Code", "claude": "Claude Code", "codex": "Codex"}
# fixed categorical order, validated (dataviz validate_palette.js, light): CVD worst adjacent dE 22.4
COLORS = {"gemini": "#2E86C1", "qwen": "#E67E22", "claude": "#8E44AD", "codex": "#27AE60"}
INK = "#333333"

plt.rcParams.update({"font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11,
                     "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 9.5})

def tidy(ax, grid_axis):
    ax.grid(axis=grid_axis, color="#E5E7E9")
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

# ---- RQ1: intent ER by skill generator, four agents; strace execution marked for the new agents
gens = ["gpt4.1_mini", "gpt4o_mini", "gpt4o", "gpt5_mini", "llama3", "deepseek_r1"]
glab = ["GPT-4.1-mini", "GPT-4o-mini", "GPT-4o", "GPT-5-mini", "Llama-3-8B", "DeepSeek-R1"]
x = np.arange(len(gens))
w = 0.2
fig, ax = plt.subplots(figsize=(6.8, 3.9))
for i, ag in enumerate(AGENTS):
    vals = [D["rq1"][ag][g]["er"] for g in gens]
    xs = x + (i - 1.5) * w
    ax.bar(xs, vals, w * 0.92, color=COLORS[ag], label=f"{NAMES[ag]} (intent)")
    ex = [D["rq1"][ag][g]["exec"] for g in gens] if ag in ("claude", "codex") else [None] * len(gens)
    for xi, v, e in zip(xs, vals, ex):
        # label sits on the bar unless the execution marker is close enough to collide
        top = max(v, e) if e is not None and e - v < 6 else v
        ax.text(xi, top + 1.6, f"{v:.0f}", ha="center", va="bottom", fontsize=7, color=INK)
    if ag in ("claude", "codex"):
        ax.scatter(xs, ex, marker="_", s=90, linewidths=2.2, color=INK, zorder=3,
                   label="strace-verified execution" if ag == "codex" else None)
ax.set_ylabel("Exploitability Rate (%)")
ax.set_ylim(0, 106)
ax.set_xticks(x)
ax.set_xticklabels(glab, rotation=15, ha="right")
ax.set_title("Exploitability Rate by Skill Generator and Agent")
hs, ls = ax.get_legend_handles_labels()
order = [ls.index(f"{NAMES[a]} (intent)") for a in AGENTS] + [ls.index("strace-verified execution")]
ax.legend([hs[i] for i in order], [ls[i] for i in order], loc="upper center",
          bbox_to_anchor=(0.5, -0.24), ncol=3, frameon=False)
tidy(ax, "y")
fig.tight_layout()
fig.savefig(FIGDIR / "RQ1.pdf", bbox_inches="tight")
fig.savefig(FIGDIR / "RQ1.png", dpi=200, bbox_inches="tight")
plt.close(fig)

# ---- RQ3: intent ER by MITRE tactic, four agents, ordered by pooled ER
pretty = {"initial-access": "Initial Access", "defense-evasion": "Defense Evasion",
          "privilege-escalation": "Privilege Escalation", "discovery": "Discovery",
          "credential-access": "Credential Access", "execution": "Execution",
          "collection": "Collection", "persistence": "Persistence",
          "command-and-control": "Command & Control", "impact": "Impact", "exfiltration": "Exfiltration"}
tacs = sorted((t for t in D["rq3"] if not t.startswith("_")), key=lambda t: -D["rq3"][t]["er"])
short = {"initial-access": "Initial\nAccess", "defense-evasion": "Defense\nEvasion",
         "privilege-escalation": "Privilege\nEscalation", "discovery": "Discovery",
         "credential-access": "Credential\nAccess", "execution": "Execution",
         "collection": "Collection", "persistence": "Persistence",
         "command-and-control": "Command &\nControl", "impact": "Impact", "exfiltration": "Exfiltration"}
x = np.arange(len(tacs))
fig, ax = plt.subplots(figsize=(7.2, 3.6))
for i, ag in enumerate(AGENTS):
    vals = [D["rq3"][t][ag] for t in tacs]
    ax.bar(x + (i - 1.5) * w, vals, w * 0.92, color=COLORS[ag], label=NAMES[ag])
ax.set_xticks(x)
ax.set_xticklabels([pretty[t] for t in tacs], rotation=30, ha="right", rotation_mode="anchor", fontsize=9)
ax.set_ylabel("Exploitability Rate (%)")
ax.set_ylim(0, 100)
ax.set_title("Exploitability Rate by MITRE ATT&CK Tactic and Agent", pad=24)
ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=4, frameon=False)
tidy(ax, "y")
fig.tight_layout()
fig.savefig(FIGDIR / "RQ3_ER_by_tactic.pdf", bbox_inches="tight")
fig.savefig(FIGDIR / "RQ3_tactic_by_agent.png", dpi=200, bbox_inches="tight")
plt.close(fig)

# ---- RQ2: failure-mode shares per agent with 95% CIs (assessed/rq2_numbers_sonnet.json, tools/rq2_numbers.py)
R2 = json.loads((REPO / "assessed/rq2_numbers_sonnet.json").read_text())
cats = ["RecognizedAttack", "IrrelevantToTask", "AcknowledgedButDidNotExecute", "NoAcknowledgment"]
clab = ["Recognized\nAttack", "Irrelevant\nTo Task", "Acknowledged But\nDid Not Execute", "No\nAcknowledgment"]
x = np.arange(len(cats))
fig, ax = plt.subplots(figsize=(6.8, 3.6))
for i, ag in enumerate(AGENTS):
    v = [R2[ag]["cats"][c]["pct"] for c in cats]
    lo = [R2[ag]["cats"][c]["pct"] - R2[ag]["cats"][c]["ci"][0] for c in cats]
    hi = [R2[ag]["cats"][c]["ci"][1] - R2[ag]["cats"][c]["pct"] for c in cats]
    xs = x + (i - 1.5) * w
    ax.bar(xs, v, w * 0.92, color=COLORS[ag], label=NAMES[ag])
    ax.errorbar(xs, v, yerr=[lo, hi], fmt="none", ecolor=INK, elinewidth=0.8, capsize=1.8)
    for xi, vi, h in zip(xs, v, hi):
        ax.text(xi, vi + h + 1.2, f"{vi:.0f}", ha="center", va="bottom", fontsize=7, color=INK)
ax.set_ylabel("Share of failures (%)")
ax.set_ylim(0, 85)
ax.set_xticks(x)
ax.set_xticklabels(clab)
ax.set_title("Failure Modes by Agent")
ax.legend(loc="upper left", ncol=2, frameon=False)
tidy(ax, "y")
fig.tight_layout()
fig.savefig(FIGDIR / "RQ2.pdf", bbox_inches="tight")
fig.savefig(FIGDIR / "RQ2.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("wrote RQ1.pdf, RQ2.pdf, RQ3_ER_by_tactic.pdf")
