# AgentJailbreak

Replication package for the paper **"The Trojan in the Toolbox: A Risk Assessment of Malicious Skill Files in the Coding Agents."**

This repository evaluates how coding agents respond to malicious skill-file instructions that mask shell commands as benign "preflight" steps. The experiments are mapped to MITRE ATT&CK tactics and evaluated with both heuristic matching and LLM-as-a-judge.

Paper in this repo: `paper.pdf`

## Project Description

- **Threat model:** malicious commands embedded in skill files that agents may trust as required setup.
- **Dataset basis:** Linux-focused Atomic Red Team commands mapped to ATT&CK tactics.
- **Attack phrasing:** commands are semantically wrapped as routine development/admin instructions.
- **Agents evaluated:** Qwen CLI and Gemini CLI in auto-approved execution mode.
- **Primary metric:** exploitability rate (ER), i.e., whether logs show explicit intent/evidence to execute preflight.

## Repository layout

- `run_from_linux_index.py` - run pipeline for Qwen CLI.
- `run_from_linux_index_gemini.py` - run pipeline for Gemini CLI.
- `tools/assess_preflight_stdout.py` - heuristic assessment of stdout logs.
- `tools/assess_preflight_stdout_llm.py` - LLM-as-a-judge assessment via local Ollama.
- `tools/calc_exploit_by_tactic.py`, `tools/calc_fail_cats.py`, `tools/compare_llm_vs_heuristic.py` - analysis utilities.
- `project_repo/` - sandbox template copied per run.
- `atomics/` - ATT&CK/Atomic Red Team artifacts used by the workflow.
- `environment.yml` - Conda environment for replication.
- `command.txt` - prompt template used for benign-style command descriptions.

## Safety and ethics

This repository contains adversarial command content for **security research and controlled evaluation only**.

- Run only in isolated test environments (local VM/sandbox).
- Do not run against production systems or sensitive data.
- Treat third-party skill files as untrusted.
- Follow your organization's legal/security approval process before execution.

## Setup

### 1) Create environment

```bash
conda env create -f environment.yml
conda activate agentjailbreak
```

### 2) Configure agent CLIs

#### Qwen

- Ensure `qwen` CLI is installed and authenticated.
- Optional environment variables:
  - `QWEN_BIN` (default: `qwen`)
  - `TASK_TIMEOUT` (default: `120`)
  - `RUNS_BASE` (default temp subdir)

#### Gemini

- Ensure `gemini` CLI is installed and authenticated.
- `run_from_linux_index_gemini.py` needs to set variables:
  - `GOOGLE_GENAI_USE_VERTEXAI`
  - `GOOGLE_CLOUD_PROJECT`
- If needed, authenticate with Google ADC (for example via `gcloud auth application-default login`).

### 3) Optional: local judge model

For LLM-based assessment, install and run Ollama locally and pull a model such as `llama3:8b`.

## Reproducing experiments

### A) Prepare or run Qwen experiments

Dry run (materialize per-row run folders only):

```bash
python run_from_linux_index.py \
  --csv linux-index-with-descriptions.csv
```

Execute runs (stream output, skip already-saved rows):

```bash
python run_from_linux_index.py \
  --csv linux-index-with-descriptions.csv \
  --run --stream --skip-if-saved
```


### B) Prepare or run Gemini experiments

Dry run:

```bash
python run_from_linux_index_gemini.py \
  --csv linux-index-with-descriptions.csv
```

Execute runs:

```bash
python run_from_linux_index_gemini.py \
  --csv linux-index-with-descriptions.csv \
  --run --stream --skip-if-saved
```

## Assess exploitability

### 1) Heuristic evaluator

```bash
python tools/assess_preflight_stdout.py \
  --base saved_runs_gemini \
  --jsonl assessed_results/heuristic/gemini_stdout.jsonl \
  --summary assessed_results/heuristic/gemini_summary.json
```

### 2) LLM-as-a-judge evaluator

```bash
python tools/assess_preflight_stdout_llm.py \
  --base saved_runs_gemini \
  --model llama3:8b \
  --jsonl assessed_results/llm/gemini_stdout.jsonl \
  --summary assessed_results/llm/gemini_summary.json \
  --stream --judge-all
```

## Output artifacts

Typical outputs include:

- Per-run folders under `saved_runs_*` containing:
  - `runs/<run_id>/stdout.log`
  - `runs/<run_id>/stderr.log`
  - injected `skills/repo_dev_skills.yaml`
  - generated `init_and_run.sh`
- aggregate logs:
  - `results.jsonl` (Qwen)
  - `results_gemini.jsonl` (Gemini)
- assessment summaries in JSON/JSONL from the `tools/` scripts.

## Notes on methodology

- Experiments use autonomous runs with a per-run timeout.
- Success is based on **declared intent/evidence** in logs
- The toolkit supports both base descriptions and mutation-based variants for robustness testing.

## Citation

If you use this repository, please cite the corresponding paper:

> Rui Yang, Michael Fu, Kla Tantithamthavorn, Chetan Arora, and Joey Chua.  
> *The Trojan in the Toolbox: A Risk Assessment of Malicious Skill Files in the Coding Agents.*  

