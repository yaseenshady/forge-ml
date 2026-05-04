"""Subprocess-based sandboxed executor for LLM-generated training scripts.

The script is expected to print a JSON dict of metrics as its last output line, e.g.:
    {"accuracy": 0.93, "f1_macro": 0.91}

If no JSON line is found the result still carries stdout/stderr for debugging.
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ExecutionResult:
    success: bool
    metrics: dict[str, float] = field(default_factory=dict)
    stdout: str = ""
    stderr: str = ""
    exit_code: int = 0
    timed_out: bool = False

    def summary(self) -> str:
        if self.timed_out:
            return "timed out"
        if not self.success:
            last_err = self.stderr.strip().splitlines()[-1] if self.stderr else "unknown error"
            return f"failed: {last_err}"
        if self.metrics:
            parts = ", ".join(f"{k}={v:.4f}" for k, v in self.metrics.items())
            return f"ok — {parts}"
        return "ok (no metrics parsed)"


def run_script(code: str, timeout: int = 120) -> ExecutionResult:
    """Write *code* to a temp file and execute it in a fresh subprocess.

    Args:
        code:    Python source to run.
        timeout: Wall-clock seconds before forceful kill.

    Returns:
        ExecutionResult with .success, .metrics, .stdout, .stderr.
    """
    with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False, encoding="utf-8") as f:
        f.write(code)
        script_path = Path(f.name)

    try:
        proc = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        stdout = proc.stdout.strip()
        stderr = proc.stderr.strip()
        exit_code = proc.returncode

        metrics = _parse_metrics(stdout)

        return ExecutionResult(
            success=exit_code == 0,
            metrics=metrics,
            stdout=stdout,
            stderr=stderr,
            exit_code=exit_code,
        )

    except subprocess.TimeoutExpired:
        return ExecutionResult(
            success=False,
            stderr=f"Timed out after {timeout}s",
            exit_code=-1,
            timed_out=True,
        )
    finally:
        script_path.unlink(missing_ok=True)


def _parse_metrics(stdout: str) -> dict[str, float]:
    """Scan stdout lines in reverse for the first valid JSON metrics dict."""
    if not stdout:
        return {}
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            data = json.loads(line)
            if isinstance(data, dict):
                return {k: float(v) for k, v in data.items() if isinstance(v, (int, float))}
        except (json.JSONDecodeError, ValueError):
            continue
    return {}
