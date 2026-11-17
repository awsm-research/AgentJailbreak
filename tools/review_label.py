#!/usr/bin/env python3
"""Terminal labeller for a review pack under review-stage/ (RQ1 or RQ2).

Each pack holds transcripts/<id>.txt, items.csv (id, planted_command) and
label_config.json (keys, label names, definitions). The labeller shows each
narration-only transcript with preflight-related words highlighted, asks for a
label, saves every answer immediately to labels.csv, and resumes at the first
unlabelled item next time.

  python3 tools/review_label.py rq1            # label the RQ1 pack
  python3 tools/review_label.py rq2            # label the RQ2 pack
  python3 tools/review_label.py rq1 --report   # counts only

At the prompt:
  <key>          set the label (keys listed on screen, ? for definitions)
  <key> <text>   label plus a note, e.g. "n said it would not run untrusted scripts"
  v              open the transcript in `less` (scroll, search with /)
  b              back to the previous item      g 42   go to item 42
  s              skip for now                   q      save and quit
"""
import argparse, csv, json, os, re, shutil, subprocess, sys, textwrap, time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent / "review-stage"
PACKS = {"rq1": ROOT / "rq1_review_pack", "rq2": ROOT / "rq2_review_pack",
         "rq1legacy": ROOT / "rq1_legacy_review_pack"}

TTY = sys.stdout.isatty()
def c(code, s): return f"\033[{code}m{s}\033[0m" if TTY else s
HL = re.compile(r"(preflight|init_and_run(?:\.sh)?|setup_env(?:\.sh)?|repo_dev_skills(?:\.yaml)?|skills?\b|"
                r"initiali[sz]\w*|mandatory|malicious|suspicious|unsafe|untrusted|security|harmful|"
                r"won't|will not|should not|shouldn't|did not|didn't|not run|skip\w*|refus\w*|"
                r"\brun(?:ning)?\b|execut\w*)", re.I)

class Pack:
    def __init__(self, path):
        self.path = path
        cfg = json.loads((path / "label_config.json").read_text())
        self.labels = cfg["labels"]            # {key: name}
        self.defs = cfg["definitions"]         # multi-line help text
        self.menu = cfg["menu"]                # one-line key reminder
        with open(path / "items.csv") as f:
            self.items = list(csv.DictReader(f))
        self.order = [it["id"] for it in self.items]

    def load(self):
        p = self.path / "labels.csv"
        if not p.exists():
            return {}
        with open(p) as f:
            return {r["id"]: r for r in csv.DictReader(f)}

    def save(self, labels):
        tmp = self.path / "labels.csv.tmp"
        with open(tmp, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["id", "label", "note", "labelled_at"])
            w.writeheader()
            for iid in self.order:
                if iid in labels:
                    w.writerow(labels[iid])
        tmp.replace(self.path / "labels.csv")

def render(pack, item, idx, labels):
    iid, total = item["id"], len(pack.items)
    text = (pack.path / "transcripts" / f"{iid}.txt").read_text(errors="ignore").strip()
    width = min(shutil.get_terminal_size((100, 40)).columns, 110)
    if TTY:
        os.system("clear")
    done = sum(1 for r in labels.values() if r["label"])
    print(c("1;36", f"{iid}   item {idx + 1}/{total}   labelled {done}/{total}"))
    cur = labels.get(iid)
    if cur and cur["label"]:
        print(c("33", f"current label: {cur['label']}" + (f"  note: {cur['note']}" if cur["note"] else "")))
    print(c("2", "planted preflight command:"))
    for ln in textwrap.wrap(item.get("planted_command") or "(none)", width - 2):
        print(c("2", "  " + ln))
    print(c("2", "─" * width))
    body = "\n".join(textwrap.fill(p, width) if p.strip() else "" for p in text.splitlines())
    print(HL.sub(lambda m: c("1;33", m.group(0)), body) if TTY else body)
    print(c("2", "─" * width))
    print(c("2", f"{len(text)} chars  "), c("1", pack.menu),
          c("2", "  v=view b=back g N=goto s=skip ?=help q=quit"))

def report(pack):
    labels = pack.load()
    cnt = Counter(r["label"] for r in labels.values() if r["label"])
    print(f"labelled {sum(cnt.values())}/{len(pack.items)}")
    for k, v in cnt.most_common():
        print(f"  {k:30s} {v}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pack", choices=sorted(PACKS))
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    path = PACKS[a.pack]
    if not (path / "items.csv").exists():
        sys.exit(f"No pack at {path}. Build it first (tools/build_{a.pack}_review_pack.py).")
    pack = Pack(path)
    if a.report:
        return report(pack)

    items, labels = pack.items, pack.load()
    idx = next((i for i, it in enumerate(items) if not labels.get(it["id"], {}).get("label")), len(items))
    msg = ""
    while True:
        if idx >= len(items):
            pending = [i for i, it in enumerate(items) if not labels.get(it["id"], {}).get("label")]
            if not pending:
                print(c("1;32", f"All {len(items)} items labelled. Saved to {path / 'labels.csv'}"))
                return
            idx, msg = pending[0], f"reached the end, back to the first skipped item ({len(pending)} left)"
        item = items[idx]
        render(pack, item, idx, labels)
        if msg:
            print(c("35", msg)); msg = ""
        try:
            ans = input(c("1", "label> ")).strip()
        except (EOFError, KeyboardInterrupt):
            print(); pack.save(labels); return
        if not ans:
            continue
        key, _, rest = ans.partition(" ")
        key = key.lower()
        if key in pack.labels:
            labels[item["id"]] = {"id": item["id"], "label": pack.labels[key], "note": rest.strip(),
                                  "labelled_at": time.strftime("%Y-%m-%d %H:%M:%S")}
            pack.save(labels)
            idx += 1
        elif key == "v":
            subprocess.run(["less", "-R", str(path / "transcripts" / f"{item['id']}.txt")])
        elif key == "b":
            idx = max(0, idx - 1)
        elif key == "s":
            idx += 1
        elif key == "g":
            try:
                n = int(rest)
                idx = next(i for i, iid in enumerate(pack.order) if int(re.sub(r"\D", "", iid)) == n)
            except (ValueError, StopIteration):
                msg = "usage: g 42"
        elif key == "?":
            msg = pack.defs
        elif key == "q":
            pack.save(labels)
            print(f"saved. {sum(1 for r in labels.values() if r['label'])}/{len(items)} labelled.")
            return
        else:
            msg = "unknown key, type ? for the label definitions"

if __name__ == "__main__":
    main()
