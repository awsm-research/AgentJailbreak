#!/usr/bin/env python3
import csv
import re
from pathlib import Path
import shutil
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = PROJECT_ROOT / "linux-index-with-descriptions.csv"
BACKUP_PATH = PROJECT_ROOT / "linux-index-with-descriptions.csv.bak"
PROJECT_REPO = PROJECT_ROOT / "project_repo"

PLACEHOLDER_PATTERN = re.compile(r"#\{([^}]+)\}")

# Files available in project_repo to map to
FILE_CANDIDATES = {
    "api_keys": PROJECT_REPO / "api_keys",
    "private_key": PROJECT_REPO / "private_key.sample",
    "readme": PROJECT_REPO / "README.md",
    "config": PROJECT_REPO / "global_config.yaml",
    "global_config": PROJECT_REPO / "global_config.yaml",
    "tmp": PROJECT_REPO / "tmp_test.txt",
}

PATH_KEYWORDS = {
    "path", "file", "folder", "dir", "directory", "output", "input", "log",
    "history", "passwd", "password", "key", "keys", "ssh", "netrc",
    "config", "yaml", "yml", "cron", "service", "module", "bin", "script",
    "image", "db", "journal", "artifact"
}

def is_path_like(name: str) -> bool:
    lower = name.lower()
    return any(kw in lower for kw in PATH_KEYWORDS)

def choose_repo_path(name: str) -> Path:
    n = name.lower()
    if "private" in n or "id_rsa" in n or ("key" in n and "public" not in n):
        return FILE_CANDIDATES["private_key"]
    if "api" in n or "credential" in n or ("key" in n and "private" not in n):
        return FILE_CANDIDATES["api_keys"]
    if "readme" in n:
        return FILE_CANDIDATES["readme"]
    if "global" in n:
        return FILE_CANDIDATES["global_config"]
    if "config" in n or "yaml" in n or "yml" in n:
        # Map any config/yaml-like placeholder to global_config.yaml
        return FILE_CANDIDATES["global_config"]
    if "folder" in n or "dir" in n or "directory" in n:
        return PROJECT_REPO
    # defaults
    return FILE_CANDIDATES["tmp"]

def substitute_placeholders(cmd: str) -> str:
    def repl(match: re.Match) -> str:
        name = match.group(1)
        if not is_path_like(name):
            # leave non-path placeholders untouched
            return match.group(0)
        path = choose_repo_path(name)
        # use POSIX-style relative path with project_repo prefix
        rel = path.relative_to(PROJECT_ROOT)
        return str(rel)
    updated = PLACEHOLDER_PATTERN.sub(repl, cmd)
    # Also rewrite any previous literal paths to config.yaml to global_config.yaml
    updated = updated.replace("project_repo/config.yaml", "project_repo/global_config.yaml")
    return updated

def main() -> int:
    if not CSV_PATH.exists():
        print(f"CSV not found: {CSV_PATH}", file=sys.stderr)
        return 1
    if not PROJECT_REPO.exists():
        print(f"project_repo not found: {PROJECT_REPO}", file=sys.stderr)
        return 1

    # backup
    if not BACKUP_PATH.exists():
        shutil.copy2(CSV_PATH, BACKUP_PATH)

    rows = []
    changed = 0
    with CSV_PATH.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        if not fieldnames or "Command" not in fieldnames:
            print("CSV missing 'Command' column", file=sys.stderr)
            return 1
        for row in reader:
            cmd = row.get("Command", "")
            new_cmd = substitute_placeholders(cmd)
            if new_cmd != cmd:
                row["Command"] = new_cmd
                changed += 1
            rows.append(row)

    with CSV_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Updated {changed} command(s). Backup at {BACKUP_PATH.name}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

