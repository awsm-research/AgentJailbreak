#!/usr/bin/env python3
"""Sonnet narration-only re-judge of all Claude Code + Codex runs with the LEGACY-STYLE
command hint, so all four agents are judged with an identical prompt.

Why: the first pass put the inner ART payload in the hint ("the mandatory setup command
was: 'dig @...'"). The agents run the wrapper ./init_and_run.sh, which executes that
payload, and Sonnet often read "ran ./init_and_run.sh" as "did not run THAT command".
The legacy Gemini/Qwen runs were judged with the hint './init_and_run.sh', which is what
the skill file actually tells the agent to run. This pass uses that hint for every run.

  submit           : one Anthropic Message Batch over all completed narration transcripts
  ingest           : write valid verdicts to assessed/panel_final_narr_sonnet_v2/<key>.sonnet.cache
  sub [--limit N]  : re-judge runs with no valid verdict via the subscription (claude -p,
                     prompt on stdin), cap-safe and resumable
  status           : coverage + ER per agent
"""
import argparse, glob, json, os, re, subprocess, sys
from pathlib import Path
import requests

sys.path.insert(0, str(Path(__file__).parent))
import assess_preflight_stdout_multi_llm as J
import anthropic_narr_batch as A            # API headers, SYSTEM prompt, verdict(), anchor_for()

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "assessed/panel_final_narr_sonnet_v2"
SP = (REPO / "scratch")
MODEL = "claude-sonnet-5-5"
HINT = ", where the mandatory setup command was: './init_and_run.sh'"   # identical to legacy
GENS = ("gpt4o", "gpt4.1_mini", "llama3", "gpt5_mini", "gpt4o_mini", "deepseek_r1")
CAP_RE = re.compile(r"hit your [\w.\- ]*limit|resets\s+\d|resets .* at ", re.I)

def leaves():
    for ag in ("claude", "codex"):
        for g in GENS:
            base = REPO / f"saved_runs_streamjson/{g}/{ag}/auto_120s"
            for rd in sorted(base.iterdir()):
                runs = rd / "runs"
                if not runs.is_dir():
                    continue
                for leaf in sorted(runs.iterdir()):
                    full = leaf / "stdout.log"
                    if full.exists() and J.classify_transcript(full.read_text(errors="ignore")) == "completed":
                        yield f"{ag}_{g}", leaf

def prompt_for(leaf):
    nar = leaf / "stdout.narration.log"
    text = (nar if nar.exists() else leaf / "stdout.log").read_text(errors="ignore")
    return (J.JUDGE_PROMPT.replace("{CMD_HINT_INLINE}", HINT) +
            f"\n\nSTDOUT BEGIN\n{text}\nSTDOUT END\n").replace("\x00", "")

def load_valid():
    v = {}
    for fp in glob.glob(str(OUT / "*.sonnet.cache")):
        for line in open(fp):
            if line.strip():
                o = json.loads(line)
                if o["result"].get("parse_ok"):
                    v[o["leaf"]] = o
    return v

def append(key, row):
    with open(OUT / f"{key}.sonnet.cache", "a") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

def submit():
    SP.mkdir(parents=True, exist_ok=True)
    reqs, mp = [], {}
    for i, (key, leaf) in enumerate(leaves()):
        cid = f"v{i}"
        mp[cid] = {"key": key, "leaf": str(leaf)}
        reqs.append({"custom_id": cid, "params": {"model": MODEL, "max_tokens": 1024, "system": A.SYSTEM,
                                                  "messages": [{"role": "user", "content": prompt_for(leaf)}]}})
    json.dump(mp, open(SP / "map.json", "w"))
    print(f"requests={len(reqs)}")
    r = requests.post("https://api.anthropic.com/v1/messages/batches", headers=A.H, json={"requests": reqs}, timeout=900)
    if r.status_code not in (200, 201):
        sys.exit(f"ERROR {r.status_code} {r.text[:300]}")
    b = r.json()
    json.dump([b["id"]], open(SP / "ids.json", "w"))
    print("batch:", b["id"], b.get("processing_status"))

def ingest():
    OUT.mkdir(parents=True, exist_ok=True)
    mp = json.load(open(SP / "map.json"))
    have = load_valid()
    stops, wrote = {}, 0
    for bid in json.load(open(SP / "ids.json")):
        b = requests.get(f"https://api.anthropic.com/v1/messages/batches/{bid}", headers=A.H, timeout=60).json()
        print(f"{bid}: {b.get('processing_status')} {b.get('request_counts')}")
        if b.get("processing_status") != "ended":
            sys.exit("still running")
        for line in requests.get(b["results_url"], headers=A.H, timeout=900).text.splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            m = mp.get(r.get("custom_id"))
            if not m or m["leaf"] in have:
                continue
            res, content, stop = r.get("result", {}), "", r.get("result", {}).get("type")
            if res.get("type") == "succeeded":
                stop = res["message"].get("stop_reason")
                content = next((c.get("text", "") for c in res["message"].get("content", []) if c.get("type") == "text"), "")
            stops[stop] = stops.get(stop, 0) + 1
            v = A.verdict(content, A.anchor_for(m["leaf"]))
            if v["parse_ok"]:
                append(m["key"], {"leaf": m["leaf"], "model": "cc:sonnet", "route": "api", "result": v})
                wrote += 1
    print(f"wrote {wrote} valid API verdicts   stop_reasons={stops}")
    status()

def sub(limit):
    OUT.mkdir(parents=True, exist_ok=True)
    have = load_valid()
    todo = [(k, l) for k, l in leaves() if str(l) not in have]
    if limit:
        todo = todo[:limit]
    print(f"subscription: {len(todo)} to judge")
    done = 0
    for key, leaf in todo:
        p = subprocess.run(["claude", "-p", "--model", MODEL, "--output-format", "json",
                            "--disallowedTools", "*", "--permission-mode", "bypassPermissions"],
                           input=prompt_for(leaf), capture_output=True, text=True, timeout=180)
        try:
            content = json.loads(p.stdout or "{}").get("result", "") or ""
        except Exception:
            content = ""
        if CAP_RE.search(content):
            print(f"CAP HIT after {done}: {content[:80]!r}. Stopping cleanly, re-run after reset.")
            break
        v = A.verdict(content, A.anchor_for(str(leaf)))
        if v["parse_ok"]:
            append(key, {"leaf": str(leaf), "model": "cc:sonnet", "route": "subscription", "result": v})
            done += 1
            if done % 25 == 0:
                print(f"  {done}/{len(todo)}", flush=True)
    print(f"subscription recovered {done}")
    status()

def status():
    have = load_valid()
    total = {"claude": 0, "codex": 0}
    for k, _ in leaves():
        total[k.split("_")[0]] += 1
    for ag in ("claude", "codex"):
        rows = [o for o in have.values() if f"/{ag}/" in o["leaf"]]
        er = sum(o["result"]["success"] for o in rows) / len(rows) * 100 if rows else 0
        print(f"  {ag:7s} valid {len(rows)}/{total[ag]}   ER {er:.1f}%")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["submit", "ingest", "sub", "status"])
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    {"submit": submit, "ingest": ingest, "status": status}.get(a.mode, lambda: sub(a.limit))()
