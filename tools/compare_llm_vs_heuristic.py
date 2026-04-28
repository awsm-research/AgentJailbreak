#!/usr/bin/env python3
# Example usage:
#   # Compare all available pairs under default dirs and print mismatches
#   python tools/compare_llm_vs_heuristic.py
#
#   # Restrict to one model family (by filename stem)
#   python tools/compare_llm_vs_heuristic.py --only qwen_gpt4o_mini
#
#   # Write mismatches to a JSONL file
#   python tools/compare_llm_vs_heuristic.py --out results/mismatches_qwen_gpt4o_mini.jsonl --only qwen_gpt4o_mini
#
#   # Use custom directories
#   python tools/compare_llm_vs_heuristic.py --llm-dir results/llm --heuristic-dir results/heuristic
#
# This script compares per-run predictions produced by:
# - tools/assess_preflight_stdout_llm.py (stored as JSONL files in results/llm)
#   records look like: {"run_dir": ".../runs/<leaf>", "success": true/false, ...}
# - tools/assess_preflight_stdout.py (stored as *_stdout.jsonl files in results/heuristic)
#   records look like: {"run_dir": ".../runs/<leaf>", "preflight_executed": true/false, ...}
# It matches runs by exact "run_dir" path and reports inconsistencies where
#   LLM.success != Heuristic.preflight_executed
#
# Pairing heuristic and llm files:
#   For each LLM JSONL "<stem>.jsonl" in --llm-dir, we look for a corresponding
#   heuristic JSONL named "<stem>_stdout.jsonl" in --heuristic-dir.
#   Use --only <stem> to restrict the comparison to a specific pair.

'''
python tools/compare_llm_vs_heuristic.py --out results/mismatches_qwen_gpt4o_mini.jsonl --only gemini_gpt4o_mini
'''

import argparse
import json
from pathlib import Path
from typing import Dict, Any, Iterable, Tuple, Optional, List
import sys
import time
import json as _json
import tty
import termios


def read_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                # Best-effort: skip malformed lines
                continue


def load_llm_map(jsonl_path: Path) -> Dict[str, Dict[str, Any]]:
    """
    Map: key(run_leaf) -> { success, method?, evidence_lines?, src, run_dir }
    Key uses the leaf directory name of run_dir to normalize differing roots (agent_outputs vs saved_runs).
    """
    result: Dict[str, Dict[str, Any]] = {}
    for rec in read_jsonl(jsonl_path):
        run_dir = rec.get("run_dir")
        if not run_dir:
            continue
        leaf_key = Path(run_dir).name  # e.g., 1773879855_T1556.003_1_0
        result[leaf_key] = {
            "success": bool(rec.get("success", False)),
            "method": rec.get("method"),
            "evidence_lines": rec.get("evidence_lines", []),
            "src": str(jsonl_path),
            "run_dir": run_dir,
        }
    return result


def load_heuristic_map(jsonl_path: Path) -> Dict[str, Dict[str, Any]]:
    """
    Map: key(run_leaf) -> { preflight_executed, matched_patterns?, samples?, src, run_dir }
    """
    result: Dict[str, Dict[str, Any]] = {}
    for rec in read_jsonl(jsonl_path):
        run_dir = rec.get("run_dir")
        if not run_dir:
            continue
        leaf_key = Path(run_dir).name
        result[leaf_key] = {
            "preflight_executed": bool(rec.get("preflight_executed", False)),
            "matched_patterns": rec.get("matched_patterns", []),
            "samples": rec.get("samples", []),
            "src": str(jsonl_path),
            "run_dir": run_dir,
        }
    return result


def find_pairs(llm_dir: Path, heuristic_dir: Path, only: Optional[List[str]] = None) -> List[Tuple[str, Path, Path]]:
    """
    Return list of (stem, llm_jsonl, heuristic_jsonl).
    LLM files are "<stem>.jsonl"; heuristic are "<stem>_stdout.jsonl".
    """
    pairs: List[Tuple[str, Path, Path]] = []
    llm_files = {p.stem: p for p in llm_dir.glob("*.jsonl")}
    for stem, llm_path in sorted(llm_files.items()):
        if only and stem not in only:
            continue
        heuristic_path = heuristic_dir / f"{stem}_stdout.jsonl"
        if heuristic_path.exists():
            pairs.append((stem, llm_path, heuristic_path))
    return pairs


def compare_pair(stem: str, llm_path: Path, heuristic_path: Path, out: Optional[Path], verbose: bool) -> dict:
    # Build LLM map and preserve order as in the JSONL file
    llm_map: Dict[str, Dict[str, Any]] = {}
    llm_order: List[str] = []
    for rec in read_jsonl(llm_path):
        run_dir = rec.get("run_dir")
        if not run_dir:
            continue
        key = Path(run_dir).name
        if key not in llm_order:
            llm_order.append(key)
        llm_map[key] = {
            "success": bool(rec.get("success", False)),
            "method": rec.get("method"),
            "evidence_lines": rec.get("evidence_lines", []),
            "src": str(llm_path),
            "run_dir": run_dir,
        }

    heu_map = load_heuristic_map(heuristic_path)

    overlap = set(llm_map.keys()) & set(heu_map.keys())
    only_llm = set(llm_map.keys()) - set(heu_map.keys())
    only_heu = set(heu_map.keys()) - set(llm_map.keys())

    mismatches: List[dict] = []
    # Keep mismatch listing in the LLM file order for consistency
    for key in [k for k in llm_order if k in overlap]:
        l_ok = bool(llm_map[key]["success"])
        h_ok = bool(heu_map[key]["preflight_executed"])
        if l_ok != h_ok:
            rec = {
                "stem": stem,
                "run_dir_llm": llm_map[key].get("run_dir"),
                "run_dir_heuristic": heu_map[key].get("run_dir"),
                "run_leaf_key": key,
                "llm_success": l_ok,
                "heuristic_preflight_executed": h_ok,
                "llm_method": llm_map[key].get("method"),
                "llm_evidence_head": (llm_map[key].get("evidence_lines") or [])[:2],
                "heuristic_patterns": (heu_map[key].get("matched_patterns") or [])[:2],
                "heuristic_samples_head": (heu_map[key].get("samples") or [])[:1],
                "llm_src": llm_map[key].get("src"),
                "heuristic_src": heu_map[key].get("src"),
            }
            mismatches.append(rec)

    return {
        "stem": stem,
        "counts": {
            "llm_total": len(llm_map),
            "heuristic_total": len(heu_map),
            "overlap": len(overlap),
            "only_llm": len(only_llm),
            "only_heuristic": len(only_heu),
            "mismatches": len(mismatches),
        },
        "mismatches": mismatches,
        "llm_map": llm_map,
        "heu_map": heu_map,
        "llm_order": llm_order,
        "only_llm": sorted(list(only_llm)) if verbose else [],
        "only_heuristic": sorted(list(only_heu)) if verbose else [],
    }


def _final_file_for(stem: str, final_dir: Path) -> Path:
    return final_dir / f"{stem}_final.jsonl"


def _load_finalized_keys(stem: str, final_dir: Path) -> set:
    path = _final_file_for(stem, final_dir)
    decided: set = set()
    if not path.exists():
        return decided
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = _json.loads(line)
                except Exception:
                    continue
                k = obj.get("run_leaf_key")
                if k:
                    decided.add(k)
    except Exception:
        pass
    return decided


def _append_final(stem: str, final_dir: Path, record: dict) -> None:
    path = _final_file_for(stem, final_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(_json.dumps(record) + "\n")


def _get_keypress() -> str:
    """
    Read a single keypress from stdin (raw mode). Return one of:
    LEFT, RIGHT, UP, DOWN, ENTER, SPACE, Q, S, OTHER
    """
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch1 = sys.stdin.read(1)
        if ch1 == "\x1b":  # ESC sequence
            ch2 = sys.stdin.read(1)
            if ch2 == "[":
                ch3 = sys.stdin.read(1)
                if ch3 == "D":
                    return "LEFT"
                if ch3 == "C":
                    return "RIGHT"
                if ch3 == "A":
                    return "UP"
                if ch3 == "B":
                    return "DOWN"
            return "OTHER"
        if ch1 in ("\r", "\n"):
            return "ENTER"
        if ch1 == " ":
            return "SPACE"
        if ch1 in ("q", "Q"):
            return "Q"
        if ch1 in ("s", "S"):
            return "S"
        return "OTHER"
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def interactive_review(stem: str, mismatches: List[dict], final_dir: Path) -> None:
    """
    Present mismatches one by one.
    Choose final attack success label:
      - LEFT  = False (no attack success)
      - RIGHT = True  (attack success)
    Keys: Q quit, S skip, ENTER next (no save).
    Decisions are appended to assessed_results/final/<stem>_final.jsonl.
    Existing decisions (by run_leaf_key) are skipped automatically.
    """
    decided = _load_finalized_keys(stem, final_dir)
    todo = [m for m in mismatches if m.get("run_leaf_key") not in decided]
    if not todo:
        print(f"[interactive] no pending mismatches for {stem}")
        return
    print(f"[interactive] reviewing {len(todo)} mismatches for {stem}")
    for idx, m in enumerate(todo, 1):
        print("\n" + "=" * 80)
        print(f"[{stem}] case {idx}/{len(todo)}  key={m.get('run_leaf_key')}")
        print(f"LLM: success={m.get('llm_success')}  method={m.get('llm_method')}")
        if m.get("llm_evidence_head"):
            print(f"  evidence: {m['llm_evidence_head']}")
        print(f"Heuristic: preflight_executed={m.get('heuristic_preflight_executed')}")
        if m.get("heuristic_patterns"):
            print(f"  patterns: {m['heuristic_patterns']}")
        if m.get("heuristic_samples_head"):
            print(f"  sample: {m['heuristic_samples_head']}")
        print("\nChoose final attack success: ← False | → True | S skip | Q quit")
        while True:
            key = _get_keypress()
            if key == "LEFT":
                choice = "false"
                break
            if key == "RIGHT":
                choice = "true"
                break
            if key == "S":
                choice = "skip"
                break
            if key == "Q":
                print("[interactive] quitting early; progress saved.")
                return
            # ignore others
        if choice == "skip":
            print("  -> skipped")
            continue
        final_success = True if choice == "true" else False
        final = {
            "stem": stem,
            "run_leaf_key": m.get("run_leaf_key"),
            "choice": choice,                 # "true" or "false"
            "final_success": final_success,   # boolean
            "llm_success": m.get("llm_success"),
            "heuristic_preflight_executed": m.get("heuristic_preflight_executed"),
            "llm_method": m.get("llm_method"),
            "llm_evidence_head": m.get("llm_evidence_head"),
            "heuristic_patterns": m.get("heuristic_patterns"),
            "heuristic_samples_head": m.get("heuristic_samples_head"),
            "run_dir_llm": m.get("run_dir_llm"),
            "run_dir_heuristic": m.get("run_dir_heuristic"),
            "timestamp": int(time.time()),
        }
        _append_final(stem, final_dir, final)
        print(f"  -> saved: final_success={final_success}")


def main():
    ap = argparse.ArgumentParser(description="Compare LLM vs. heuristic preflight predictions and report inconsistencies.")
    ap.add_argument("--llm-dir", default="assessed_results/llm", help="Directory containing LLM JSONL files (default: assessed_results/llm)")
    ap.add_argument("--heuristic-dir", default="assessed_results/heuristic", help="Directory containing heuristic *_stdout.jsonl files (default: assessed_results/heuristic)")
    ap.add_argument("--only", action="append", help="Limit comparison to specific filename stem(s), e.g., --only qwen_gpt4o_mini (can repeat)")
    ap.add_argument("--out", help="Optional JSONL path to write mismatches")
    ap.add_argument("--verbose", action="store_true", help="Show non-overlapping runs and extra details")
    ap.add_argument("--interactive", action="store_true", help="Interactively review mismatches and choose final attack success (False/True) with arrows.")
    ap.add_argument("--final-dir", default="assessed_results/final", help="Directory to store finalized decisions per stem (default: assessed_results/final)")
    args = ap.parse_args()

    llm_dir = Path(args.llm_dir).resolve()
    heu_dir = Path(args.heuristic_dir).resolve()
    if not llm_dir.exists() or not heu_dir.exists():
        print(f"[error] directories not found: {llm_dir} or {heu_dir}")
        return

    pairs = find_pairs(llm_dir, heu_dir, args.only)
    if not pairs:
        print("[warn] no pairs found to compare")
        return

    out_path = Path(args.out).resolve() if args.out else None
    out_f = None
    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_f = out_path.open("w", encoding="utf-8")

    final_dir = Path(args.final_dir).resolve()

    grand_mismatches = 0
    for stem, llm_jsonl, heu_jsonl in pairs:
        res = compare_pair(stem, llm_jsonl, heu_jsonl, out_path, args.verbose)
        llm_map = res.get("llm_map", {})
        heu_map = res.get("heu_map", {})
        llm_order = res.get("llm_order", [])
        c = res["counts"]
        print(f"\n=== {stem} ===")
        print(f"- LLM total:        {c['llm_total']}")
        print(f"- Heuristic total:  {c['heuristic_total']}")
        print(f"- Overlap:          {c['overlap']}")
        print(f"- Only in LLM:      {c['only_llm']}")
        print(f"- Only in Heuristic:{c['only_heuristic']}")
        print(f"- MISMATCHES:       {c['mismatches']}")

        for m in res["mismatches"]:
            print(f"\n  [mismatch] key={m['run_leaf_key']}")
            print(f"    run_dir_llm: {m['run_dir_llm']}")
            print(f"    run_dir_heuristic: {m['run_dir_heuristic']}")
            print(f"    llm_success: {m['llm_success']} ({m.get('llm_method')})  heuristic_preflight_executed: {m['heuristic_preflight_executed']}")
            if m["llm_evidence_head"]:
                print(f"    llm_evidence: {m['llm_evidence_head']}")
            if m["heuristic_patterns"]:
                print(f"    heuristic_patterns: {m['heuristic_patterns']}")
            if m["heuristic_samples_head"]:
                print(f"    heuristic_samples: {m['heuristic_samples_head']}")
            if out_f:
                out_f.write(json.dumps(m) + "\n")
        if args.verbose:
            if res["only_llm"]:
                print(f"\n  [only_llm] {len(res['only_llm'])} entries")
            if res["only_heuristic"]:
                print(f"  [only_heuristic] {len(res['only_heuristic'])} entries")
        grand_mismatches += c["mismatches"]

        if args.interactive and res["mismatches"]:
            interactive_review(stem, res["mismatches"], final_dir)
            # After interactive review, also append all agreements in LLM order
            decided_keys = _load_finalized_keys(stem, final_dir)
            for key in llm_order:
                if key in decided_keys:
                    continue
                if key not in llm_map or key not in heu_map:
                    continue
                l_ok = bool(llm_map[key]["success"])
                h_ok = bool(heu_map[key]["preflight_executed"])
                if l_ok == h_ok:
                    final = {
                        "stem": stem,
                        "run_leaf_key": key,
                        "choice": "auto_agree",
                        "final_success": bool(l_ok),
                        "llm_success": l_ok,
                        "heuristic_preflight_executed": h_ok,
                        "llm_method": llm_map[key].get("method"),
                        "llm_evidence_head": (llm_map[key].get("evidence_lines") or [])[:2],
                        "heuristic_patterns": (heu_map[key].get("matched_patterns") or [])[:2],
                        "heuristic_samples_head": (heu_map[key].get("samples") or [])[:1],
                        "run_dir_llm": llm_map[key].get("run_dir"),
                        "run_dir_heuristic": heu_map[key].get("run_dir"),
                        "timestamp": int(time.time()),
                    }
                    _append_final(stem, final_dir, final)

    if out_f:
        out_f.close()
        print(f"\n[wrote mismatches] {out_path}")
    print(f"\n[done] total mismatches across pairs: {grand_mismatches}")


if __name__ == "__main__":
    main()

