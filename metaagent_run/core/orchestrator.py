import json
from typing import Any, Dict, List


def load_json_items(file_path: str) -> List[Dict[str, Any]]:
    with open(file_path, "r", encoding="utf-8") as file:
        raw_data = json.load(file)
    if isinstance(raw_data, dict):
        return [raw_data]
    if isinstance(raw_data, list):
        return [entry for entry in raw_data if isinstance(entry, dict)]
    raise ValueError("Input format error: expected JSON object or array.")
