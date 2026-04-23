"""viz_final 共享数据加载。"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

DATA_DIR: Path = Path(__file__).resolve().parents[4] / "env_field_pipeline_output"


def load_json(name: str) -> dict:
    with open(DATA_DIR / name, "r", encoding="utf-8") as f:
        return json.load(f)


def load_csv(name: str, **kwargs: Any) -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / name, **kwargs)
