#!/usr/bin/env python3
# Example usage:
#   # Run over each folder in agent_outputs and write results to assessed_results/llm
#   python tools/run_llm_assessor_over_outputs.py
#
#   # Specify model and stream
#   python tools/run_llm_assessor_over_outputs.py --model llama3:8b --stream
#
#   # Custom bases
#   python tools/run_llm_assessor_over_outputs.py \
#     --agent-outputs-base /Users/ryan0079/Documents/GitHub/AgentJailbreak/agent_outputs \
#     --assessed-out-dir /Users/ryan0079/Documents/GitHub/AgentJailbreak/assessed_results/llm
#
# For each subfolder under agent_outputs, this script runs:
#   tools/assess_preflight_stdout_llm.py --base <that subfolder> \
#     --jsonl <assessed_out_dir>/<name>.jsonl \
#     --summary <assessed_out_dir>/<name>_summary.json
#
# Where <name> is the folder basename.

import argparse
import subprocess
from pathlib import Path
import sys
import time


def run_one(base_dir: Path, out_dir: Path, model: str, stream: bool, judge_all: bool, include_empty: bool) -> int:
    name = base_dir.name
    jsonl = out_dir / f"{name}.jsonl"
    summary = out_dir / f"{name}_summary.json"

    script = (Path(__file__).parent.parent / "tools" / "assess_preflight_stdout_llm.py").resolve()
    if not script.exists():
        print(f"[error] assessor script not found at {script}", file=sys.stderr)
        return 1

    cmd = [
        sys.executable,
        str(script),
        "--base",
        str(base_dir),
        "--model",
        model,
        "--jsonl",
        str(jsonl),
        "--summary",
        str(summary),
    ]
    if stream:
        cmd.append("--stream")
    if judge_all:
        cmd.append("--judge-all")
    if include_empty:
        cmd.append("--include-empty")

    print(f"[assess] {name} -> {jsonl.name}, {summary.name}")
    try:
        res = subprocess.run(cmd, check=False)
        return res.returncode
    finally:
        # Gentle pacing to avoid overwhelming local/remote model servers
        time.sleep(1)


def main():
    ap = argparse.ArgumentParser(description="Batch runner to assess agent outputs with LLM and collect results per folder.")
    ap.add_argument(
        "--agent-outputs-base",
        default=str((Path(__file__).resolve().parent.parent / "agent_outputs").resolve()),
        help="Base directory containing per-agent-output folders (default: ./agent_outputs)",
    )
    ap.add_argument(
        "--assessed-out-dir",
        default=str((Path(__file__).resolve().parent.parent / "assessed_results" / "llm").resolve()),
        help="Directory to store LLM assessment outputs (default: ./assessed_results/llm)",
    )
    ap.add_argument("--model", default="llama3:8b", help="Ollama model name for the LLM judge (default: llama3:8b)")
    ap.add_argument("--stream", action="store_true", help="Stream model output while judging.")
    ap.add_argument("--judge-all", action="store_true", help="Run the model for every run even if regex matched.")
    ap.add_argument("--include-empty", action="store_true", help="Do not skip empty stdout logs.")
    args = ap.parse_args()

    base = Path(args.agent_outputs_base).resolve()
    out_dir = Path(args.assessed_out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not base.exists():
        print(f"[error] agent outputs base not found: {base}", file=sys.stderr)
        sys.exit(1)

    subdirs = [p for p in sorted(base.iterdir()) if p.is_dir()]
    if not subdirs:
        print(f"[warn] no subfolders found under {base}")
        sys.exit(0)

    failures = 0
    for d in subdirs:
        rc = run_one(d, out_dir, args.model, args.stream, args.judge_all, args.include_empty)
        if rc != 0:
            print(f"[fail] {d.name} rc={rc}")
            failures += 1
    print(f"[done] assessed {len(subdirs)} folders; failures={failures}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()

