#!/usr/bin/env bash
set -euo pipefail

# Run setup
echo "Initializing..."
./setup_env.sh

# Run the actual tool (replace current shell)
exec "$@"

