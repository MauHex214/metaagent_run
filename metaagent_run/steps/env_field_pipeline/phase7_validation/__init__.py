"""Phase 7: Schema validation via stratified extraction sampling.

Iteratively run step5 on stratified paper samples (env × section × era),
accumulate per-target tries/successes, and prune unextractable targets
from schema v1 to produce schema v2.
"""
from __future__ import annotations

from pathlib import Path
from typing import Final

from metaagent_run.steps.env_field_pipeline import config as ep_config

# ── Output directory ────────────────────────────────────────────────────
PHASE7_DIR: Final[Path] = ep_config.OUTPUT_DIR / "phase7_validation"

# ── Upstream paths ──────────────────────────────────────────────────────
ACCESSION_LIST_PATH: Final[Path] = ep_config.PROJECT_ROOT_DIR / "pmid_run-accession0418.list"

# ── Stratification dimensions ───────────────────────────────────────────
TARGET_ENVS: Final[tuple] = ("Open_ocean", "Coastal_waters", "Lake", "Wetlands")

# step2 section_type → normalized tier (the 4 tiers we sample on)
SECTION_TIER_MAP: Final[dict] = {
    "METHODS": "methods",
    "RESULTS": "results",
    "supplementary": "supplementary",
    "SUPPL": "supplementary",
    "TABLE": "table",
}
TIERS: Final[tuple] = ("methods", "results", "supplementary", "table")

# 3-era buckets: pre-MIxS-or-MIxS-early merged into "<2016" so the formal-run
# 48-cell (env × section × era) all have ≥3 papers after the accession filter
# kills 86% of the universe (mostly because pre-2011 papers rarely deposited
# SRA accessions).
ERAS: Final[tuple] = ("<2016", "2016-2020", "2021+")


def to_era(year: int) -> str:
    if year <= 2015:
        return "<2016"
    if year <= 2020:
        return "2016-2020"
    return "2021+"


def ensure_phase7_dir() -> None:
    PHASE7_DIR.mkdir(parents=True, exist_ok=True)
