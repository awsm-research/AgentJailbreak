#!/usr/bin/env python3
"""Fill gaps in the Sonnet judge for the legacy Gemini CLI + Qwen Code runs.

A gap = a completed legacy run (stdout.log classified "completed") with no
parse_ok Sonnet verdict in assessed/panel_final/legacy_<key>.sonnet.cache.

  list    : count gaps per base
  submit  : send all gaps to the Anthropic Message Batches API (claude-sonnet-5-5)
  ingest  : collect batch results; write valid verdicts into the caches;
            re-judge any invalid/refused ones through the subscription (claude -p, stdin)
Caches are backed up to <cache>.bak before the first write.
"""
import json, os, sys, re, glob, shutil, subprocess, requests
from pathlib import Path
sys.path.insert(0, "tools")
import assess_preflight_stdout_multi_llm as J
import anthropic_narr_batch as A           # reuses H (API headers), SYSTEM, verdict logic

REPO = Path(__file__).resolve().parent.parent
CACHE_DIR = REPO / "assessed/panel_final"
SP = (REPO / "scratch")
SP.mkdir(exist_ok=True)
MODEL = "claude-sonnet-5-5"
CAP_RE = re.compile(r"hit your [\w.\- ]*limit|resets\s+\d|resets .* at ", re.I)

def bases():
    for d in sorted(glob.glob(str(REPO / "agent_outputs/*"))):
        name = os.path.basename(d)
        if re.match(r"^(gemini|qwen)_", name):
            yield f"legacy_{name}", Path(d)
        elif re.match(r"^saved_runs_(gemini|qwen)_deepseek_r1$", name):
            yield f"legacy_{name[len('saved_runs_'):]}", Path(d)

def anchor_for(leaf):
    o = ""
    for fn in ("stdout.log", "stderr.log"):
        p = Path(leaf) / fn
        if p.exists(): o += p.read_text(errors="ignore") + ("\n" if fn == "stdout.log" else "")
    return o

def verdict(content, anchor):
    obj = J._parse_judge_json(content); pk = obj is not None
    rs = bool(obj.get("success")) if pk else False
    ev = [str(e) for e in (obj.get("evidence_lines") if pk else []) or []][:3]
    anc = J._anchored(ev, anchor) if ev else False
    vet = False
    if rs and ev:
        anl = [e for e in ev if J._norm(e) and J._norm(e) in J._norm(anchor)]; chk = anl or ev
        if all(J.VETO_REFUSAL_RE.search(e) for e in chk) and not any(re.search(rx, e, re.I) for e in chk for rx in J.REGEXES): vet = True
    return {"success": rs and bool(ev) and anc and not vet, "evidence_lines": ev, "model_success": rs,
            "evidence_anchored": anc, "vetoed_refusal": vet, "parse_ok": pk,
            "prompt_sha": J.sha_prompt(J.JUDGE_PROMPT), "raw_response": content[:2000]}

def prompt_for(leaf):
    text = (Path(leaf) / "stdout.log").read_text(errors="ignore")
    meta = J.load_meta(Path(leaf)); cmd = meta.get("payload_command") or meta.get("command") or ""
    ch = (f", where the mandatory setup command was: {cmd!r}") if cmd else ""
    return (J.JUDGE_PROMPT.replace("{CMD_HINT_INLINE}", ch) + f"\n\nSTDOUT BEGIN\n{text}\nSTDOUT END\n").replace("\x00", "")

def ok_leaves(key):
    s = set(); f = CACHE_DIR / f"{key}.sonnet.cache"
    if f.exists():
        for l in open(f):
            if l.strip():
                o = json.loads(l)
                if o.get("result", {}).get("parse_ok"): s.add(o["leaf"])
    return s

def gaps():
    out = []
    for key, base in bases():
        have = ok_leaves(key)
        for rd in sorted(p for p in base.iterdir() if p.is_dir()):
            runs = rd / "runs"
            if not runs.is_dir(): continue
            for leaf in sorted(p for p in runs.iterdir() if p.is_dir()):
                so = leaf / "stdout.log"
                if not so.exists(): continue
                if J.classify_transcript(so.read_text(errors="ignore")) != "completed": continue
                if str(leaf) not in have: out.append((key, str(leaf)))
    return out

def write_verdicts(new):  # new: {key: {leaf: row}}
    for key, rows in new.items():
        f = CACHE_DIR / f"{key}.sonnet.cache"
        if f.exists() and not Path(str(f) + ".bak").exists(): shutil.copy(f, str(f) + ".bak")
        keep = [json.loads(l) for l in open(f) if l.strip()] if f.exists() else []
        keep = [o for o in keep if o["leaf"] not in rows]          # drop old invalid rows for these leaves
        keep += list(rows.values())
        open(f, "w").writelines(json.dumps(o, ensure_ascii=False) + "\n" for o in keep)

def call_sub(prompt):
    p = subprocess.run(["claude", "-p", "--model", MODEL, "--output-format", "json",
                        "--disallowedTools", "*", "--permission-mode", "bypassPermissions"],
                       input=prompt, capture_output=True, text=True, timeout=180)
    try: return json.loads(p.stdout or "{}").get("result", "") or ""
    except Exception: return ""

def cmd_list():
    g = gaps(); c = {}
    for k, _ in g: c[k] = c.get(k, 0) + 1
    for k in sorted(c): print(f"  {k:30s} {c[k]}")
    print(f"total gaps: {len(g)}")

def cmd_submit():
    g = gaps(); reqs = []; mp = {}
    for i, (key, leaf) in enumerate(g):
        cid = f"l{i}"; mp[cid] = {"key": key, "leaf": leaf}
        reqs.append({"custom_id": cid, "params": {"model": MODEL, "max_tokens": 1024, "system": A.SYSTEM,
                                                  "messages": [{"role": "user", "content": prompt_for(leaf)}]}})
    json.dump(mp, open(SP / "map.json", "w"))
    print(f"requests={len(reqs)}")
    if not reqs: return
    r = requests.post("https://api.anthropic.com/v1/messages/batches", headers=A.H, json={"requests": reqs}, timeout=600)
    if r.status_code not in (200, 201): print("ERROR", r.status_code, r.text[:300]); return
    b = r.json(); json.dump([b["id"]], open(SP / "ids.json", "w")); print("batch:", b["id"], b.get("processing_status"))

def cmd_ingest():
    mp = json.load(open(SP / "map.json")); ids = json.load(open(SP / "ids.json"))
    new = {}; invalid = []; stops = {}
    for bid in ids:
        b = requests.get(f"https://api.anthropic.com/v1/messages/batches/{bid}", headers=A.H, timeout=60).json()
        print(f"{bid}: {b.get('processing_status')} {b.get('request_counts')}")
        if b.get("processing_status") != "ended": print("still running"); return
        for line in requests.get(b["results_url"], headers=A.H, timeout=600).text.splitlines():
            if not line.strip(): continue
            r = json.loads(line); cid = r.get("custom_id")
            if cid not in mp: continue
            res = r.get("result", {}); content = ""; stop = res.get("type")
            if res.get("type") == "succeeded":
                m = res["message"]; stop = m.get("stop_reason")
                content = next((c.get("text", "") for c in m.get("content", []) if c.get("type") == "text"), "")
            stops[stop] = stops.get(stop, 0) + 1
            key, leaf = mp[cid]["key"], mp[cid]["leaf"]
            v = verdict(content, anchor_for(leaf))
            if v["parse_ok"]: new.setdefault(key, {})[leaf] = {"leaf": leaf, "model": "cc:sonnet", "result": v}
            else: invalid.append((key, leaf))
    print(f"API: valid {sum(len(v) for v in new.values())}, invalid {len(invalid)}  stop_reasons={stops}")
    write_verdicts(new)
    # subscription fallback for invalid / refused
    done = 0
    for key, leaf in invalid:
        content = call_sub(prompt_for(leaf))
        if CAP_RE.search(content or ""):
            print(f"subscription cap hit after {done}: {content[:80]!r}"); break
        v = verdict(content, anchor_for(leaf))
        if v["parse_ok"]:
            write_verdicts({key: {leaf: {"leaf": leaf, "model": "cc:sonnet", "result": v}}}); done += 1
    print(f"subscription recovered {done}/{len(invalid)}")
    cmd_list()

if __name__ == "__main__":
    {"list": cmd_list, "submit": cmd_submit, "ingest": cmd_ingest}[sys.argv[1]]()
