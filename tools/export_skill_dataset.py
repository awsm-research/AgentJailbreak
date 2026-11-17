#!/usr/bin/env python3
"""Export the malicious skill-file dataset (471 ART entries x 6 generators = 2,826 skill files)
from the run sandboxes under agent_outputs/ into one JSONL file, dataset/skill_files.jsonl.

Each record: generator, rid (<technique>_<test>_<index>), technique, test,
description (the generator's benign 'Developer notes' line), payload (the ART command
placed in init_and_run.sh), and the full skill file text (skills/repo_dev_skills.yaml).
The Gemini CLI sandboxes are used as the source. Qwen Code, Claude Code and Codex CLI
ran the same skill files.

Usage: python tools/export_skill_dataset.py
"""
import json, re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SRC = {"gpt4o": "gemini_gpt4o", "gpt4o_mini": "gemini_gpt4o_mini", "gpt4.1_mini": "gemini_gpt4.1_mini",
       "gpt5_mini": "gemini_gpt5_mini", "llama3": "gemini_llama3", "deepseek_r1": "saved_runs_gemini_deepseek_r1"}
RID = re.compile(r"(T\d{4}(?:\.\d{3})?)_(\d+)_(\d+)$")


def payload(init_sh):
    lines = [l for l in init_sh.splitlines() if l.strip() and not l.startswith(("#", "set ", "echo ", "./setup_env.sh", "if ", "  exec", "fi"))]
    return "\n".join(lines)


def main():
    out = REPO / "dataset" / "skill_files.jsonl"
    out.parent.mkdir(exist_ok=True)
    n = 0
    with open(out, "w") as w:
        for gen, folder in SRC.items():
            latest = {}                                   # a rerun leaves a second folder; keep the newest
            for d in sorted((REPO / "agent_outputs" / folder).iterdir()):
                m = RID.search(d.name)
                if m and (d / "skills" / "repo_dev_skills.yaml").exists():
                    latest[m.group(0)] = d
            for d in latest.values():
                m = RID.search(d.name)
                skill = d / "skills" / "repo_dev_skills.yaml"
                if not m or not skill.exists():
                    continue
                text = skill.read_text(errors="ignore")
                desc = re.search(r"Developer notes:\s*(.*)", text)
                w.write(json.dumps({
                    "generator": gen, "rid": f"{m.group(1)}_{m.group(2)}_{m.group(3)}",
                    "technique": m.group(1), "test": int(m.group(2)),
                    "description": desc.group(1).strip() if desc else "",
                    "payload": payload((d / "init_and_run.sh").read_text(errors="ignore")),
                    "skill_file": text}) + "\n")
                n += 1
    print(f"wrote {n} skill files -> {out.relative_to(REPO)}")


if __name__ == "__main__":
    main()
