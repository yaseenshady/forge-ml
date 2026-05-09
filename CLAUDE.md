# forge-ml

Multi-agent ML platform: research → data discovery → model building → hyperparameter optimization.

## Project
- **Repo:** https://github.com/yaseenshady/forge-ml
- **Language:** Python 3.11+
- **Entry:** `forge` CLI (`src/forge_ml/cli.py`)
- **Modules:** `research`, `data`, `model`, `orchestrator`, `utils`

## Dev Setup
```bash
pip install -e ".[dev]"
```

## Run
```bash
forge run "build a classifier for sentiment analysis"
forge search "image classification"
forge providers
```

## Launch with full permissions
```bash
./scripts/dev-claude.sh    # Claude Code --dangerously-skip-permissions
./scripts/dev-codex.sh     # Codex --approval-mode full-auto
```

## GitHub CLI
```bash
gh repo view yaseenshady/forge-ml
gh pr create
gh issue list
```
