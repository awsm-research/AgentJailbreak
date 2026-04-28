#!/usr/bin/env python3
# Usage:
#   # Dry-run: scan projects and show which would be removed (all stdout.log empty/missing)
#   python /Users/ryan0079/Documents/GitHub/AgentJailbreak/tools/clean_empty_runs.py \
#     --base /Users/ryan0079/Documents/GitHub/AgentJailbreak/saved_runs_soft \
#     --mode project --verbose
#
#   # Actually delete empty projects (project mode)
#   python /Users/ryan0079/Documents/GitHub/AgentJailbreak/tools/clean_empty_runs.py \
#     --base /Users/ryan0079/Documents/GitHub/AgentJailbreak/saved_runs_soft \
#     --mode project --yes --verbose
#
#   # Leaf mode: remove each runs/<leaf> whose stdout.log is empty; remove project if runs/ becomes empty
#   python /Users/ryan0079/Documents/GitHub/AgentJailbreak/tools/clean_empty_runs.py \
#     --base /Users/ryan0079/Documents/GitHub/AgentJailbreak/saved_runs_gemini \
#     --mode leaf --yes --verbose
#
#   # Remove duplicate project folders (same trailing key after first underscore) regardless of emptiness
#   python /Users/ryan0079/Documents/GitHub/AgentJailbreak/tools/clean_empty_runs.py \
#     --base /Users/ryan0079/Documents/GitHub/AgentJailbreak/saved_runs_qwen_gpt4o \
#     --remove-duplicates --yes --verbose
#
# Notes:
#   - Without --yes this script performs a dry-run and prints actions it would take.
#   - "Empty" means stdout.log is missing or whitespace-only, matching the assessor's emptiness logic.
#   - In leaf mode, individual empty leaves are removed; if a project's runs/ ends up empty, the project is removed.
#   - Duplicate removal groups folders by the portion of the name after the first underscore.
import argparse
import shutil
from pathlib import Path

def is_effectively_empty_stdout(path: Path) -> bool:
    try:
        if not path.exists():
            return True
        if path.stat().st_size == 0:
            return True
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            chunk = f.read(4096)
            if chunk.strip():
                return False
            rest = f.read(4096)
            return (rest.strip() == "")
    except Exception:
        return True

def project_folder_is_empty_run(project_root: Path) -> bool:
    """
    A project folder is 'empty' if every runs/**/stdout.log is missing or whitespace-only.
    If no runs/ exists or no stdout.log found, treat as empty.
    """
    runs_dir = project_root / "runs"
    if not runs_dir.exists():
        return True
    logs = list(runs_dir.rglob("stdout.log"))
    if not logs:
        return True
    return all(is_effectively_empty_stdout(p) for p in logs)

def main():
    ap = argparse.ArgumentParser(description="Remove empty runs matching assessor's emptiness logic.")
    ap.add_argument("--base", required=True, help="Folder containing per-run project folders (e.g., saved_runs or saved_runs_gemini).")
    ap.add_argument("--yes", action="store_true", help="Actually delete. Without this, dry-run only.")
    ap.add_argument("--verbose", action="store_true", help="Print keep/remove decisions.")
    ap.add_argument(
        "--mode",
        choices=["project", "leaf"],
        default="project",
        help="project: remove entire project folder only if ALL stdout logs are empty; leaf: remove each runs/<leaf> with empty stdout.log",
    )
    ap.add_argument(
        "--remove-duplicates",
        action="store_true",
        help="Remove ALL project folders that share the same trailing name after the first underscore (duplicate rids).",
    )
    args = ap.parse_args()

    base = Path(args.base).resolve()
    if not base.exists() or not base.is_dir():
        print(f"Base not found or not a directory: {base}")
        return

    scanned = removed = 0
    removed_dedup = 0
    if args.remove_duplicates:
        # Group by trailing key (portion after the first underscore)
        key_to_projects = {}
        for project in sorted(base.iterdir()):
            if not project.is_dir():
                continue
            name = project.name
            idx = name.find("_")
            key = name[idx + 1 :] if idx != -1 else name
            key_to_projects.setdefault(key, []).append(project)
        for key, projs in key_to_projects.items():
            if len(projs) > 1:
                for p in projs:
                    if args.verbose or not args.yes:
                        print(f"[remove-dup] {p} (duplicate trailing key='{key}')")
                    if args.yes:
                        try:
                            shutil.rmtree(p)
                            removed += 1
                            removed_dedup += 1
                        except Exception as e:
                            print(f"[error] failed to remove {p}: {e}")
    if args.mode == "project":
    for project in sorted(base.iterdir()):
        if not project.is_dir():
            continue
        scanned += 1
        empty = project_folder_is_empty_run(project)
        if empty:
            if args.verbose or not args.yes:
                print(f"[remove] {project} (all stdout empty or missing)")
            if args.yes:
                try:
                    shutil.rmtree(project)
                    removed += 1
                except Exception as e:
                    print(f"[error] failed to remove {project}: {e}")
        else:
            if args.verbose:
                print(f"[keep]   {project} (has non-empty stdout)")
    else:
        # leaf mode: remove individual runs/<leaf> directories whose stdout.log is effectively empty
        for project in sorted(base.iterdir()):
            if not project.is_dir():
                continue
            runs_dir = project / "runs"
            if not runs_dir.exists():
                # No runs directory at all -> remove entire project
                scanned += 1
                if args.verbose or not args.yes:
                    print(f"[remove] {project} (no runs/)")
                if args.yes:
                    try:
                        shutil.rmtree(project)
                        removed += 1
                    except Exception as e:
                        print(f"[error] failed to remove {project}: {e}")
                continue

            leaf_dirs = [p for p in runs_dir.iterdir() if p.is_dir()]
            if not leaf_dirs:
                # Empty runs directory -> remove entire project
                scanned += 1
                if args.verbose or not args.yes:
                    print(f"[remove] {project} (runs/ empty)")
                if args.yes:
                    try:
                        shutil.rmtree(project)
                        removed += 1
                    except Exception as e:
                        print(f"[error] failed to remove {project}: {e}")
                continue

            # Process each leaf run
            for leaf in sorted(leaf_dirs):
                scanned += 1
                stdout = leaf / "stdout.log"
                if is_effectively_empty_stdout(stdout):
                    if args.verbose or not args.yes:
                        print(f"[remove] {leaf} (empty stdout)")
                    if args.yes:
                        try:
                            shutil.rmtree(leaf)
                            removed += 1
                        except Exception as e:
                            print(f"[error] failed to remove {leaf}: {e}")
                else:
                    if args.verbose:
                        print(f"[keep]   {leaf} (non-empty stdout)")

            # After leaf cleanup, if runs/ has no remaining leaf dirs, remove project
            try:
                remaining = [p for p in runs_dir.iterdir() if p.is_dir()]
                if not remaining:
                    if args.verbose or not args.yes:
                        print(f"[remove] {project} (runs/ now empty)")
                    if args.yes:
                        try:
                            shutil.rmtree(project)
                            removed += 1
                        except Exception as e:
                            print(f"[error] failed to remove {project}: {e}")
            except Exception:
                pass

    label = "projects" if args.mode == "project" else "leaves"
    if args.remove_duplicates:
        print(f"Scanned {label}: {scanned} | Removed: {removed} (duplicates removed: {removed_dedup}) | Dry-run: {not args.yes}")
    else:
        print(f"Scanned {label}: {scanned} | Removed: {removed} | Dry-run: {not args.yes}")

if __name__ == "__main__":
    main()