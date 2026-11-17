#!/usr/bin/env bash
# Sequential 5-generator sweep for one agent (claude|codex), exec-aware funnel.
# Loops the five remaining skill-file generators one at a time (max 1 docker
# container per agent process, same proven load as the gpt4o run). Each
# generator writes its OWN results file and save-root so --skip-done never
# collides across generators (rid_short is shared across generators). Fully
# resumable: re-run this script and completed (rid,cell) rows are skipped.
#
# Usage: AGENT=claude bash exec_aware/run_all_generators.sh
#        AGENT=codex  bash exec_aware/run_all_generators.sh
set -u
cd "$(dirname "$0")/.."
AGENT="${AGENT:?set AGENT=claude|codex}"

# generator label -> skill-file source directory (bash 3.2 compatible: no assoc arrays)
# GENS can be overridden via env to run a subset (for parallel per-generator jobs).
GENS="${GENS:-gpt4o_mini gpt4.1_mini gpt5_mini llama3 deepseek_r1}"
# Output naming is env-overridable so a re-capture (e.g. stream-json) can write
# to separate files without clobbering the original --print run.
RES_PREFIX="${RES_PREFIX:-results_mainfull}"
SAVE_BASE="${SAVE_BASE:-saved_runs_mainfull}"
src_for() {
  case "$1" in
    gpt4o)       echo agent_outputs/gemini_gpt4o ;;
    gpt4o_mini)  echo agent_outputs/gemini_gpt4o_mini ;;
    gpt4.1_mini) echo agent_outputs/gemini_gpt4.1_mini ;;
    gpt5_mini)   echo agent_outputs/gemini_gpt5_mini ;;
    llama3)      echo agent_outputs/gemini_llama3 ;;
    deepseek_r1) echo agent_outputs/saved_runs_gemini_deepseek_r1 ;;
    *) echo "UNKNOWN_GEN_$1" ;;
  esac
}

for gen in $GENS; do
  subset="review-stage/exec_aware_full_${gen}.json"
  src="$(src_for "$gen")"
  res="${RES_PREFIX}_${AGENT}_${gen}.jsonl"
  echo "=== [$(date '+%F %T')] $AGENT / $gen -> $res ==="
  nice python3 exec_aware/run_exec_aware.py \
    --agent "$AGENT" --subset "$subset" --src "$src" --generator "$gen" \
    --permissions auto --timeouts 120 --max-attempts 3 --backoff 15,45 --skip-done \
    --results "$res" --save-root "${SAVE_BASE}/${gen}"
  echo "=== [$(date '+%F %T')] DONE $AGENT / $gen ==="
done
echo "=== [$(date '+%F %T')] ALL GENERATORS DONE for $AGENT ==="
