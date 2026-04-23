from functools import lru_cache
import json
from pathlib import Path
from typing import Dict, List

from .config import RELATION_PROMPT_VERSION


PROMPT_DIR = Path(__file__).resolve().parent / "prompts"


@lru_cache(maxsize=8)
def load_prompt_template(prompt_version: str) -> str:
    prompt_file = PROMPT_DIR / f"{prompt_version}.txt"
    if not prompt_file.exists():
        available = sorted(path.stem for path in PROMPT_DIR.glob("*.txt"))
        raise FileNotFoundError(
            f"Prompt template not found: {prompt_version}. Available: {available}"
        )
    with prompt_file.open("r", encoding="utf-8") as file:
        return file.read().strip()


def build_prompt(
    input_text: str,
    prompt_version: str = RELATION_PROMPT_VERSION,
) -> List[Dict[str, str]]:
    payload = json.dumps({"text": input_text}, ensure_ascii=False, indent=2)
    system_prompt = load_prompt_template(prompt_version)
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": payload},
    ]


def build_batch_prompt(
    input_texts: List[str],
    prompt_version: str = RELATION_PROMPT_VERSION,
) -> List[Dict[str, str]]:
    batch = [{"text": text} for text in input_texts]
    input_json_str = json.dumps(batch, ensure_ascii=False, indent=2)
    system_prompt = load_prompt_template(prompt_version)
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": input_json_str},
    ]
