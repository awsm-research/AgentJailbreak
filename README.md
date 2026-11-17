# AgentJailbreak

Replication package for the paper **"Towards a Risk Assessment of Malicious Skill Files in Coding Agents"** (Rui Yang, Michael Fu, Chakkrit Tantithamthavorn, Chetan Arora, and Joey Chua).

Coding agents can be extended with *skill files*, natural-language instructions the agent loads on demand. This repository measures how often four coding agents act on a malicious shell command that a skill file presents as a mandatory "preflight" setup step. The payloads are 471 Linux shell commands from [Atomic Red Team](https://github.com/redcanaryco/atomic-red-team) (ART), mapped to MITRE ATT&CK tactics, and each one is described in benign language by six different LLMs, giving 2,826 malicious skill files.

## Study at a glance

| | |
|---|---|
| Agents | Gemini CLI (v0.33.1, default model selection), Qwen Code (qwen3-coder-plus), Claude Code (Claude Haiku 4.5), Codex CLI (GPT-4.1-mini), all with auto-approved tool use |
| Task | One fixed benign request: scaffold a small web application. 120-second budget per run |
| Skill files | 471 ART entries x 6 description generators (GPT-4o, GPT-4o-mini, GPT-4.1-mini, GPT-5-mini, Llama-3-8B, DeepSeek-R1) |
| Runs | 11,281 completed runs |
| Intent measure (ER) | Share of runs in which the agent states an intention to run the planted command, labeled by an LLM judge (Claude Sonnet 5.5) that must quote anchored evidence from the transcript |
| Execution measure (ExR) | Share of runs in which a system-call trace (`strace`, `execve`) shows the preflight script was launched. Claude Code and Codex CLI only, run inside a Docker container |
| Judge validation | 400 blind human labels (100 per agent, stratified by the judge's verdict) |

## Repository layout

```text
dataset/skill_files.jsonl          the 2,826 malicious skill files (generator, ART entry, benign description, payload, full skill text)
linux-index-with-commands.csv      the 471 Linux ART entries with their tactics and commands
atomics/                           Atomic Red Team source data used to extract the commands
project_repo/                      sandbox repository copied into every run (skill file, init_and_run.sh wrapper, decoy files)
command.txt                        prompt template for the benign descriptions

run_from_linux_index.py            runner for Qwen Code
run_from_linux_index_gemini.py     runner for Gemini CLI
run_from_linux_index_claude.py     host runner for Claude Code (not used for the paper's Claude Code results, see exec_aware/)
run_from_linux_index_codex.py      host runner for Codex CLI (not used for the paper's Codex CLI results, see exec_aware/)
experiment_lib.py                  helpers shared by the runners
exec_aware/                        Docker image and runner for Claude Code and Codex CLI with strace (execution-aware runs)

tools/                             skill synthesis, judging, execution check, human validation and analysis scripts
assessed/                          final judge verdicts, human labels and every number reported in the paper
review-stage/                      blind human-labelling packs (transcripts, answer keys, labels) and the run lists
figures/                           RQ1 to RQ3 figures
```

## Pipeline

The steps follow the workflow figure of the paper.

**Steps 1 to 3: skill-file synthesis.** `tools/extract_linux_commands.py` extracts the Linux commands from `atomics/` into `linux-index-with-commands.csv`. `tools/describe_commands_with_ollama.py` asks each generator for a benign one-line description of each command using `command.txt`. The runners then write each description and command into the skill file and `init_and_run.sh` of a fresh copy of `project_repo/`. The resulting skill files are exported to `dataset/skill_files.jsonl` by `tools/export_skill_dataset.py`.

**Step 4: running the agents.**

- Gemini CLI and Qwen Code run on the host with `run_from_linux_index_gemini.py` and `run_from_linux_index.py` (`--run --stream --skip-if-saved`).
- Claude Code and Codex CLI run inside the container defined in `exec_aware/Dockerfile`, replaying the exact sandboxes of the Gemini CLI runs, with every process launch recorded by `strace`:

```bash
docker build -t agentjb-execaware:latest exec_aware/
AGENT=claude GENS="gpt4o gpt4o_mini gpt4.1_mini gpt5_mini llama3 deepseek_r1" \
  RES_PREFIX=results_streamjson SAVE_BASE=saved_runs_streamjson \
  bash exec_aware/run_all_generators.sh          # repeat with AGENT=codex
```

Claude Code authenticates with `CLAUDE_CODE_OAUTH_TOKEN` and Codex CLI with `OPENAI_API_KEY`. Put both in a `.env` file at the repository root, which is git-ignored.

**Step 5: labelling.**

- `tools/make_narration.py` reduces each Claude Code and Codex CLI transcript to the agent's own narration.
- `tools/sonnet_narr_v2.py submit|ingest` (Claude Code, Codex CLI) and `tools/legacy_sonnet_fill.py submit|ingest` (Gemini CLI, Qwen Code) run the Claude Sonnet 5.5 judge through the Anthropic Message Batches API (`ANTHROPIC_API_KEY` in `.env`). The judge logic, including evidence anchoring and the refusal veto, is in `tools/assess_preflight_stdout_multi_llm.py` and `tools/anthropic_narr_batch.py`, with unit tests in `tools/test_judge_algorithm.py`.
- `tools/strace_exec.py` decides from each run's `execve.log` whether the preflight script was launched.
- `tools/build_rq1_review_pack.py`, `tools/build_rq1_legacy_review_pack.py` and `tools/build_rq2_review_pack.py` build the blind human-labelling packs, and `tools/review_label.py` is the terminal labeller used to fill them.

**Step 6: analysis.** These scripts compute every number in the paper and write them to `assessed/`:

| Script | Output | Paper |
|---|---|---|
| `tools/paper_numbers_sonnet.py` | `assessed/paper_numbers_sonnet.json` | RQ1 and RQ3 tables (ER, ExR, intent x execution, tactics) |
| `tools/human_agreement_weighted.py` | `assessed/human_agreement_weighted.json` | judge reliability table (kappa, accuracy, human-corrected ER) |
| `tools/cluster_ci_sonnet.py` | `assessed/cluster_ci_sonnet.json` | payload-clustered intervals and ICC |
| `tools/paired_agents_sonnet.py` | `assessed/paired_agents_sonnet.json` | paired McNemar comparisons between agents |
| `tools/rq2_numbers.py` | `assessed/rq2_numbers_sonnet.json` | RQ2 failure categories |
| `tools/make_figures_sonnet.py` | `figures/` | RQ1 to RQ3 figures |

## Reproducing the numbers

The judge verdicts and human labels are in this repository. The scripts in Step 6 also read the raw run folders (for example the strace logs), which are too large for GitHub and are distributed as a separate archive, `AgentJailbreak_full_artifacts.zip`. Unzip it at the repository root, then run:

```bash
conda env create -f environment.yml && conda activate agentjailbreak
pip install matplotlib numpy          # figures only
python tools/paper_numbers_sonnet.py
python tools/human_agreement_weighted.py
python tools/cluster_ci_sonnet.py
python tools/paired_agents_sonnet.py
python tools/rq2_numbers.py
python tools/make_figures_sonnet.py
```

The archive also contains the raw outputs of every run (`agent_outputs/`, `saved_runs_*/`, `results*.jsonl`) and the scripts and outputs of judge configurations explored during the study but not used for the reported results.

## Safety and ethics

This repository contains real attack commands from Atomic Red Team and skill files designed to make agents run them. It is intended for security research and controlled evaluation only.

- Run agents only in isolated, disposable environments such as the provided container.
- Never point the runners at production systems, real credentials or sensitive data.
- Treat third-party skill files as untrusted.

## Citation

> Rui Yang, Michael Fu, Chakkrit Tantithamthavorn, Chetan Arora, and Joey Chua. *Towards a Risk Assessment of Malicious Skill Files in Coding Agents.*
