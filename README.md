<div align="center">

# 🔥 forge-ml

**A multi-agent ML platform that turns a sentence into a trained, optimized model.**

[![Python](https://img.shields.io/badge/python-3.11%2B-blue?logo=python)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![PyPI version](https://img.shields.io/badge/pypi-0.1.0-orange)](https://pypi.org/project/forge-ml/)

</div>

---

You describe an ML task in plain English. Forge does the rest — researching approaches, finding a dataset, writing and *running* training code, optimizing hyperparameters, and handing you a summary with real metrics.

```bash
forge run "classify sentiment of movie reviews into positive or negative"
```

```
✓ Research complete
✓ Dataset: stanfordnlp/imdb (HuggingFace)
✓ Baseline: accuracy=0.9231
✓ Optuna best score: 0.9418  {n_estimators: 312, max_depth: 14}
✓ Total cost: $0.0041 | Providers: claude, openai

• Task: binary sentiment classification on movie reviews
• Dataset: stanfordnlp/imdb — 50 000 samples, label=sentiment
• Baseline accuracy 92.3 % → Optuna tuned to 94.2 %
• Best params: n_estimators=312, max_depth=14
• Pipeline cost: $0.004
```

---

## How it works

Forge runs six coordinated agents in sequence. Each stage gets the outputs of the previous one, so every decision is grounded in real data.

```
┌──────────┐    ┌──────────┐    ┌──────────┐    ┌───────────┐    ┌──────────┐    ┌──────────┐
│ Research │ → │   Data   │ → │  Build   │ → │  Execute  │ → │ Optimize │ → │Summarize │
│ (Claude) │    │ (Claude) │    │  (GPT-4) │    │(subprocess│    │ (Optuna) │    │ (Claude) │
└──────────┘    └──────────┘    └──────────┘    └───────────┘    └──────────┘    └──────────┘
     │                │               │                │               │
  SOTA approaches  dataset +      runnable         real metrics    best params
  architectures    target col     Python script    from stdout     + score
```

| Stage | Agent | What it does |
|-------|-------|-------------|
| **Research** | Claude | Identifies SOTA approaches, architectures, relevant datasets |
| **Data** | Claude | Picks the best public dataset (HuggingFace / Kaggle / UCI), returns JSON spec |
| **Build** | GPT-4o | Writes a clean, runnable training script |
| **Execute** | Subprocess | Runs the script in an isolated process, captures JSON metrics from stdout |
| **Optimize** | Optuna | Runs 30-trial hyperparameter search on the real dataset |
| **Summarize** | Claude | Produces a plain-English report with real numbers |

Providers are routed by stage — Claude for reasoning-heavy work, GPT-4o for code generation, Optuna for search. If one provider is unavailable, Forge falls back gracefully.

---

## Install

```bash
pip install forge-ml
```

Or from source:

```bash
git clone https://github.com/yaseensh/forge-ml
cd forge-ml
pip install -e .
```

---

## Configuration

Create a `.env` file in your project root (or export these variables):

```env
# Required: at least one provider key
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...

# Optional
GEMINI_API_KEY=...
KAGGLE_USERNAME=...
KAGGLE_KEY=...

# Override defaults
FORGE_DEFAULT_PROVIDER=claude   # claude | openai | gemini
FORGE_MAX_AGENTS=4
FORGE_WORKSPACE=~/.forge-ml
```

---

## Usage

### CLI

```bash
# Run the full pipeline
forge run "predict house prices from tabular features"

# Force a specific provider
forge run "detect fraud in credit card transactions" --provider openai

# Save the full run context as JSON
forge run "classify dog breeds from images" --output run.json

# Search HuggingFace datasets
forge search "medical image classification" --limit 8

# List available providers
forge providers
```

### Python API

```python
import asyncio
from forge_ml.orchestrator import Orchestrator

async def main():
    orc = Orchestrator()
    ctx = await orc.run("classify sentiment of movie reviews into positive or negative")

    print(f"Dataset:  {ctx.dataset_name} ({ctx.dataset_source})")
    print(f"Metrics:  {ctx.eval_metrics}")
    print(f"Best:     {ctx.best_score:.4f}  {ctx.best_params}")
    print(f"Cost:     ${ctx.total_cost_usd:.4f}")

asyncio.run(main())
```

### Auto-train on your own data

```python
import pandas as pd
from forge_ml.model.builder import auto_train
from forge_ml.model.optimizer import optimize

df = pd.read_csv("my_dataset.csv")

# Baseline in one call
result = auto_train(df, target_column="label", task_type="classification")
print(result.score)  # e.g. 0.9231

# Hyperparameter search
best = optimize(df, target_column="label", task_type="classification", n_trials=50)
print(best)  # {"best_params": {...}, "best_score": 0.9418, "n_trials": 50}
```

---

## Architecture

```
forge-ml/
├── src/forge_ml/
│   ├── orchestrator/
│   │   ├── router.py       # Main pipeline — routes stages, runs execution
│   │   ├── context.py      # Shared mutable state across all agents
│   │   └── providers/      # Claude, OpenAI (base + concrete implementations)
│   ├── research/
│   │   ├── researcher.py   # Queries an LLM for SOTA approaches
│   │   └── summarizer.py   # Distills research into structured output
│   ├── data/
│   │   ├── finder.py       # Searches HuggingFace + Kaggle concurrently
│   │   ├── loader.py       # Loads any source into a pandas DataFrame
│   │   └── sources/        # HuggingFace, Kaggle, UCI adapters
│   ├── model/
│   │   ├── builder.py      # auto_train() — sklearn baseline in one call
│   │   ├── evaluator.py    # Classification + regression metric helpers
│   │   ├── optimizer.py    # Optuna study with cross-validated objective
│   │   └── executor.py     # Subprocess sandbox — runs generated scripts
│   ├── utils/
│   │   ├── config.py       # Env-var config (ForgeConfig)
│   │   └── logging.py      # Rich-powered colored logs
│   └── cli.py              # Typer CLI (forge run / search / providers)
├── examples/
│   └── sentiment_analysis.py
└── tests/
```

---

## Supported providers

| Provider | Models | Stages |
|----------|--------|--------|
| **Anthropic Claude** | claude-3-5-sonnet, claude-3-haiku | research, data, summarize |
| **OpenAI** | gpt-4o, gpt-4o-mini | build, optimize |
| **Gemini** | gemini-1.5-pro *(coming soon)* | all |

---

## Roadmap

- [ ] Streaming CLI output (token-by-token from each agent)
- [ ] Run history & comparison (SQLite persistence)
- [ ] Gemini and Mistral provider adapters
- [ ] Full Kaggle dataset loading
- [ ] Fine-tuning pipeline (LoRA / QLoRA)
- [ ] Web UI (Streamlit dashboard)
- [ ] Agent memory — learn from past runs

---

## License

MIT © [Yaseen Shady](https://github.com/yaseensh)
