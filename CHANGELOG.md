# Changelog

## [Unreleased]

### Added
- CLI-based LLM providers: `claude` (Claude Code), `codex`, and `gh copilot` run in full-auto mode with no API keys required
- Smart stage routing: codex handles build/optimize, claude-code handles research/summarize, copilot as fallback
- CLI providers auto-detected at startup and take priority over API key providers

## [0.2.0] - 2026-05-09

### Added
- `executor.py` — subprocess sandbox that runs LLM-generated training scripts and captures JSON metrics from stdout
- Real model execution wired into the pipeline: generated scripts are actually run, not just printed
- `auto_train()` baseline always runs alongside the generated script for a reliable fallback metric
- Optuna hyperparameter optimization now runs on real loaded data, not hypothetical inputs
- Actual eval metrics passed into the summarize stage so the final report has real numbers
- `TaskContext` fields: `generated_script`, `script_success`, `execution_stdout`, `execution_stderr`

### Changed
- Orchestrator stage routing now prefers the best provider per stage instead of always using the default

## [0.1.0] - 2026-05-04

### Added
- Initial multi-agent pipeline: research → data → build → optimize → summarize
- Claude and OpenAI provider adapters with prompt caching
- `auto_train()` — one-call sklearn baseline for classification and regression
- Optuna optimizer with cross-validated objective
- HuggingFace and Kaggle dataset search
- `forge` CLI with `run`, `search`, and `providers` commands
- `README.md` with pipeline diagram, usage examples, and architecture overview
