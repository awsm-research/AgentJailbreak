#!/usr/bin/env python3
"""Write stdout.narration.log (agent natural-language only, no command/output/
tool/reasoning blocks) next to each Claude/Codex stream-json run, from the saved
stdout.raw.jsonl. Judge on it via JUDGE_STDOUT_NAME=stdout.narration.log for a
~5x shorter, narration-only (intent) judging input; strace still owns execution."""
import json, glob, os
from pathlib import Path
REPO = Path(__file__).resolve().parent.parent

def narration(raw_text):
    out = []
    for line in raw_text.splitlines():
        line = line.strip()
        if not line: continue
        try: d = json.loads(line)
        except Exception: continue
        t = d.get("type")
        if t == "assistant":  # claude stream-json
            for b in d.get("message", {}).get("content", []):
                if b.get("type") == "text" and b.get("text", "").strip():
                    out.append(b["text"].strip())
        elif t == "result" and d.get("result"):
            out.append(str(d["result"]).strip())
        elif t in ("item.completed", "item.updated"):  # codex --json
            it = d.get("item", {})
            if it.get("item_type") == "agent_message" or it.get("type") == "agent_message":
                tx = it.get("text") or it.get("message") or ""
                if tx.strip(): out.append(tx.strip())
        elif d.get("msg", {}).get("type") == "agent_message":
            tx = d["msg"].get("message", "")
            if tx.strip(): out.append(tx.strip())
    return "\n".join(out)

n = wrote = empty = 0
for agent in ("claude", "codex"):
    for base in glob.glob(str(REPO / f"saved_runs_streamjson/*/{agent}/auto_120s")):
        for rd in Path(base).iterdir():
            runs = rd / "runs"
            if not runs.is_dir(): continue
            for leaf in runs.iterdir():
                raw = leaf / "stdout.raw.jsonl"
                if not raw.exists(): continue
                n += 1
                nar = narration(raw.read_text(errors="ignore"))
                if len(nar) < 10:
                    empty += 1
                    # fall back to full stdout.log so the leaf stays judgeable
                    full = leaf / "stdout.log"
                    nar = full.read_text(errors="ignore") if full.exists() else nar
                (leaf / "stdout.narration.log").write_text(nar)
                wrote += 1
print(f"processed {n} raw transcripts, wrote {wrote} stdout.narration.log, {empty} fell back to full (near-empty narration)")
