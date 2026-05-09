#!/bin/bash
# Launch Codex in forge-ml with full-auto approval mode
cd "$(dirname "$0")/.." || exit 1
exec codex --approval-mode full-auto "$@"
