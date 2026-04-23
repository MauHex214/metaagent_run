"""Prompt 模板加载工具。"""

from functools import lru_cache
from pathlib import Path

PROMPT_DIR = Path(__file__).resolve().parent / "prompts"


@lru_cache(maxsize=8)
def _load_system_prompt(prompt_version: str) -> str:
    """Load a prompt template by name from the prompts/ directory."""
    path = PROMPT_DIR / f"{prompt_version}.txt"
    if not path.exists():
        available = sorted(p.stem for p in PROMPT_DIR.glob("*.txt"))
        raise FileNotFoundError(
            f"Prompt not found: {prompt_version}. Available: {available}"
        )
    return path.read_text(encoding="utf-8").strip()
