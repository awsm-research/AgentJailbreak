# Execution-aware experiment image: runs Claude Code and Codex CLIs *inside*
# Linux so real Atomic Red Team preflight commands execute contained and
# observable. Agents authenticate at runtime (CLAUDE_CODE_OAUTH_TOKEN for
# Claude Code; OPENAI_API_KEY for Codex). Disposable per run.
FROM node:20-bookworm

# Tools commonly referenced by Linux ART commands + general dev task tooling.
RUN apt-get update && apt-get install -y --no-install-recommends \
      bash coreutils findutils grep sed gawk procps iproute2 iputils-ping \
      curl wget ca-certificates git python3 python3-pip python3-venv \
      cron at sudo util-linux lsof netcat-openbsd openssh-client \
      xz-utils gzip tar zip unzip jq vim-tiny less strace \
 && rm -rf /var/lib/apt/lists/*

# Agent CLIs (Linux builds).
RUN npm install -g @anthropic-ai/claude-code @openai/codex \
 && claude --version && codex --version

# Non-root user so agents behave like a normal developer workstation account,
# while still able to sudo (auto-approved threat model).
RUN useradd -m -s /bin/bash dev && echo "dev ALL=(ALL) NOPASSWD:ALL" >/etc/sudoers.d/dev
WORKDIR /work
ENV HOME=/home/dev
ENV CODEX_HOME=/home/dev/.codex
ENV PATH="/work/bin:${PATH}"
RUN mkdir -p /home/dev/.codex /home/dev/.claude && chown -R dev:dev /home/dev /work
USER dev
CMD ["/bin/bash"]
