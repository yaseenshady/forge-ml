#!/usr/bin/env python3
"""
forge-ml AI driver — programmatically prompts Claude Code, Codex, or Copilot.
Stolen directly from LEXOIRE backend/src/services/{claude,codex,copilot}-service.ts.

Usage:
  python scripts/ai.py --provider claude  --prompt "build a classifier"
  python scripts/ai.py --provider codex   --prompt "refactor the orchestrator"
  python scripts/ai.py --prompt "your task here"   # defaults to claude
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from shutil import which

HOME = Path.home()

# ── Binary resolution (mirrors LEXOIRE resolveCommandBinary) ─────────────────

def _find(*candidates: str | None) -> str | None:
    for c in candidates:
        if not c:
            continue
        p = Path(c).expanduser()
        if p.exists() and os.access(p, os.X_OK):
            return str(p)
        found = which(c)
        if found:
            return found
    return None

CLAUDE_BIN = _find(
    os.environ.get("CLAUDE_COMMAND"),
    str(HOME / ".local/bin/claude"),
    str(HOME / ".npm-global/bin/claude"),
    "claude",
)

CODEX_BIN = _find(
    os.environ.get("CODEX_COMMAND"),
    str(HOME / ".npm-global/bin/codex"),
    str(HOME / ".local/bin/codex"),
    "codex",
)

COPILOT_BIN = _find(
    os.environ.get("COPILOT_COMMAND"),
    "/opt/homebrew/bin/copilot",
    "/usr/local/bin/copilot",
    "copilot",
)

SYSTEM_PROMPT = (
    "You are an expert ML engineer working inside the forge-ml multi-agent ML platform. "
    "Be concise, direct, and action-oriented. When given a task, do it. "
    "Prefer short plain-English status and results."
)

REPO_ROOT = str(Path(__file__).parent.parent)


# ── Claude ────────────────────────────────────────────────────────────────────

def run_claude(prompt: str, session_id: str | None = None) -> str:
    if not CLAUDE_BIN:
        sys.exit("[error] claude binary not found. Install: npm i -g @anthropic-ai/claude-code")

    args = [
        CLAUDE_BIN,
        "--dangerously-skip-permissions",
        "--verbose",
        "--output-format", "stream-json",
        "--include-partial-messages",
        "--system-prompt", SYSTEM_PROMPT,
    ]
    if session_id:
        args += ["--resume", session_id]
    args += ["--print", prompt]

    proc = subprocess.Popen(
        args,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=REPO_ROOT,
    )

    full = ""
    buf = ""
    streaming_used = False
    new_session_id = None

    def handle_line(line: str) -> None:
        nonlocal full, streaming_used, new_session_id
        line = line.strip()
        if not line:
            return
        try:
            ev = json.loads(line)
            inner = ev.get("event") if ev.get("type") == "stream_event" else None

            if inner and inner.get("type") == "content_block_delta":
                delta = inner.get("delta", {})
                if delta.get("type") == "text_delta":
                    streaming_used = True
                    chunk = delta["text"]
                    print(chunk, end="", flush=True)
                    full += chunk

            if not streaming_used and ev.get("type") == "content_block_delta":
                delta = ev.get("delta", {})
                if delta.get("type") == "text_delta":
                    streaming_used = True
                    chunk = delta["text"]
                    print(chunk, end="", flush=True)
                    full += chunk

            if not streaming_used and ev.get("type") == "assistant":
                for block in ev.get("message", {}).get("content", []):
                    if block.get("type") == "text" and block.get("text"):
                        print(block["text"], end="", flush=True)
                        full += block["text"]

            if ev.get("type") == "result" and ev.get("session_id"):
                new_session_id = ev["session_id"]

            if ev.get("type") == "result" and ev.get("result") and not full:
                print(ev["result"], end="", flush=True)
                full = ev["result"]
        except json.JSONDecodeError:
            pass

    assert proc.stdout
    for raw_line in proc.stdout:
        buf += raw_line
        while "\n" in buf:
            line, buf = buf.split("\n", 1)
            handle_line(line)

    proc.wait()
    if buf.strip():
        handle_line(buf)

    print()
    if new_session_id:
        print(f"[claude session: {new_session_id}]", file=sys.stderr)
    return full.strip()


# ── Codex ─────────────────────────────────────────────────────────────────────

def run_codex(prompt: str, session_id: str | None = None) -> str:
    if not CODEX_BIN:
        sys.exit("[error] codex binary not found. Install: npm i -g @openai/codex")

    if session_id:
        args = [
            CODEX_BIN, "exec", "resume", session_id,
            "--dangerously-bypass-approvals-and-sandbox",
            "--skip-git-repo-check",
            "--json",
            prompt,
        ]
    else:
        args = [
            CODEX_BIN, "exec",
            "--dangerously-bypass-approvals-and-sandbox",
            "--skip-git-repo-check",
            "--json",
            prompt,
        ]

    proc = subprocess.Popen(
        args,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=REPO_ROOT,
    )

    full = ""
    buf = ""
    new_session_id = None

    def handle_line(line: str) -> None:
        nonlocal full, new_session_id
        line = line.strip()
        if not line:
            return
        try:
            ev = json.loads(line)

            if ev.get("type") == "item.completed":
                item = ev.get("item", {})
                if item.get("type") == "agent_message":
                    text = item.get("text", "")
                    if text:
                        for i in range(0, len(text), 32):
                            print(text[i:i+32], end="", flush=True)
                        full += text

            if ev.get("type") == "item.updated":
                delta = ev.get("item", {}).get("text_delta", "")
                if delta:
                    print(delta, end="", flush=True)
                    full += delta

            if ev.get("session_id"):
                new_session_id = ev["session_id"]
            if ev.get("type") == "session_started" and ev.get("id"):
                new_session_id = ev["id"]

            if ev.get("type") == "result" and ev.get("result") and not full:
                print(ev["result"], end="", flush=True)
                full = ev["result"]

        except json.JSONDecodeError:
            pass

    assert proc.stdout
    for raw_line in proc.stdout:
        buf += raw_line
        while "\n" in buf:
            line, buf = buf.split("\n", 1)
            handle_line(line)

    proc.wait()
    if buf.strip():
        handle_line(buf)

    print()
    if new_session_id:
        print(f"[codex session: {new_session_id}]", file=sys.stderr)
    return full.strip()


# ── Copilot ───────────────────────────────────────────────────────────────────

def run_copilot(prompt: str, session_id: str | None = None) -> str:
    if not COPILOT_BIN:
        sys.exit("[error] copilot binary not found.")

    args = [COPILOT_BIN, "api", "completions", "--prompt", prompt]
    if session_id:
        args += ["--session", session_id]

    proc = subprocess.Popen(
        args, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, cwd=REPO_ROOT,
    )
    out, err = proc.communicate()
    if out.strip():
        print(out.strip())
        return out.strip()
    if err.strip():
        print(f"[copilot stderr] {err.strip()}", file=sys.stderr)
    return ""


# ── CLI entry ─────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="forge-ml AI driver")
    parser.add_argument("--provider", choices=["claude", "codex", "copilot"], default="claude")
    parser.add_argument("--prompt", required=True, help="The prompt / task to send")
    parser.add_argument("--session", default=None, help="Resume a previous CLI session ID")
    args = parser.parse_args()

    if args.provider == "claude":
        run_claude(args.prompt, args.session)
    elif args.provider == "codex":
        run_codex(args.prompt, args.session)
    elif args.provider == "copilot":
        run_copilot(args.prompt, args.session)


if __name__ == "__main__":
    main()
