from functools import lru_cache
import json
from pathlib import Path
from typing import Any, Dict, List, Union

from .config import PROMPT_VERSION


PROMPT_DIR = Path(__file__).resolve().parent / "prompts"


@lru_cache(maxsize=8)
def load_prompt_template(prompt_version: str) -> str:
    prompt_file = PROMPT_DIR / f"{prompt_version}.txt"
    if not prompt_file.exists():
        available = sorted(path.stem for path in PROMPT_DIR.glob("*.txt"))
        raise FileNotFoundError(
            f"Prompt template not found: {prompt_version}. Available: {available}"
        )
    with open(prompt_file, "r", encoding="utf-8") as file:
        return file.read().strip()


def build_prompt(
    items: Union[Dict[str, Any], List[Dict[str, Any]]],
    prompt_version: str = PROMPT_VERSION,
) -> List[Dict[str, str]]:
    if isinstance(items, dict):
        data_batch = [items]
    else:
        data_batch = items
    user_payload = (
        "Please extract the required fields from the following samples in JSON array format:\n\n"
        + json.dumps(data_batch, ensure_ascii=False, indent=2)
    )
    system_prompt = load_prompt_template(prompt_version)
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_payload},
    ]
