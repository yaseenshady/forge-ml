"""Shared rich console logger."""
from rich.console import Console
from rich.theme import Theme

theme = Theme({
    "info": "cyan",
    "success": "green bold",
    "warn": "yellow",
    "error": "red bold",
    "agent": "magenta",
})

console = Console(theme=theme)

def log(msg: str, style: str = "info") -> None:
    console.print(f"[{style}]{msg}[/{style}]")

def agent_log(provider: str, msg: str) -> None:
    console.print(f"[agent][{provider}][/agent] {msg}")
