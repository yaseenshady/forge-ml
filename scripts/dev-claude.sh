#!/bin/bash
# Launch Claude Code in forge-ml with full auto-permissions
cd "$(dirname "$0")/.." || exit 1
exec claude --dangerously-skip-permissions "$@"
