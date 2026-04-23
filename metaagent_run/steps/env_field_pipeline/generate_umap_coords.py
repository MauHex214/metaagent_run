"""Regenerate env3_viz_coords.csv after phase3 re-run (post raw-key expansion).

Workflow:
    1. Read env3_final_annotations.csv (4132 rows post-expansion).
    2. Build one descriptive sentence per raw_key combining subtype,
       quantity_kind, and modifier_bag so the embedding carries
       semantic context beyond the bare field name.
    3. Embed with sentence-transformers (all-MiniLM-L6-v2, local,
       no API calls).
    4. UMAP-reduce to 2D.
    5. Write env_field_pipeline_output/viz/env3_viz_coords.csv with
       columns (raw_key, family, subtype, quantity_kind,
       modifier_bag, description, x, y) — schema matches the previous
       coords file so viz_final/06/07/09/10 can be re-run unchanged.

Usage:
    python3 -m metaagent_run.steps.env_field_pipeline.generate_umap_coords
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd

from . import config

logger = logging.getLogger(__name__)

VIZ_DIR: Path = config.OUTPUT_DIR / "viz"
COORDS_CSV: Path = VIZ_DIR / "env3_viz_coords.csv"
EMBEDDINGS_NPY: Path = VIZ_DIR / "env3_field_embeddings.npy"
DESCRIPTIONS_JSONL: Path = VIZ_DIR / "env3_field_descriptions.jsonl"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
UMAP_N_NEIGHBORS = 30
UMAP_MIN_DIST = 0.1
UMAP_SEED = 42


def _build_description(row: pd.Series) -> str:
    rk = str(row["raw_key"])
    sub = str(row.get("subtype", ""))
    qk = str(row.get("quantity_kind", ""))
    bag_raw = row.get("modifier_bag", "")
    bag = str(bag_raw) if pd.notna(bag_raw) and str(bag_raw).strip() else ""

    parts = [rk]
    if sub:
        parts.append(f"a {sub} field")
    if qk and qk != rk and qk != "other":
        parts.append(f"measuring {qk}")
    if bag:
        parts.append(f"with modifiers {bag.replace('|', ', ')}")
    return ". ".join(parts) + "."


def run() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    VIZ_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Loading env3_final_annotations.csv …")
    df = pd.read_csv(config.PHASE3_OUTPUT)
    logger.info("Loaded %d rows", len(df))

    # If raw_key duplicates exist (shouldn't after dedup fix in phase3), keep first
    df = df.drop_duplicates(subset=["raw_key"], keep="first").reset_index(drop=True)
    logger.info("After dedup: %d rows", len(df))

    # Build descriptions
    descriptions = df.apply(_build_description, axis=1).tolist()
    logger.info("Sample description: %r", descriptions[0])

    # Save descriptions for provenance
    with open(DESCRIPTIONS_JSONL, "w", encoding="utf-8") as f:
        import json as _json
        for rk, desc in zip(df["raw_key"], descriptions):
            f.write(_json.dumps({"raw_key": rk, "description": desc},
                                ensure_ascii=False) + "\n")

    # Embed
    logger.info("Loading embedding model %s …", EMBEDDING_MODEL)
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(EMBEDDING_MODEL)
    logger.info("Encoding %d descriptions (may take a few minutes on CPU) …",
                len(descriptions))
    embeddings = model.encode(
        descriptions,
        batch_size=128,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    logger.info("Embeddings shape: %s", embeddings.shape)
    np.save(EMBEDDINGS_NPY, embeddings)
    logger.info("Saved embeddings → %s", EMBEDDINGS_NPY)

    # UMAP
    logger.info("UMAP reducing %dD → 2D (n_neighbors=%d, min_dist=%.2f) …",
                embeddings.shape[1], UMAP_N_NEIGHBORS, UMAP_MIN_DIST)
    import umap
    reducer = umap.UMAP(
        n_neighbors=UMAP_N_NEIGHBORS,
        min_dist=UMAP_MIN_DIST,
        n_components=2,
        metric="cosine",
        random_state=UMAP_SEED,
    )
    coords = reducer.fit_transform(embeddings)
    logger.info("UMAP done. coords shape: %s, x range [%.2f, %.2f], y range [%.2f, %.2f]",
                coords.shape, coords[:, 0].min(), coords[:, 0].max(),
                coords[:, 1].min(), coords[:, 1].max())

    # Assemble output CSV
    out = df[["raw_key", "family", "subtype", "quantity_kind", "modifier_bag"]].copy()
    out["description"] = descriptions
    out["x"] = coords[:, 0]
    out["y"] = coords[:, 1]
    # NaN bag → "" so downstream viz doesn't choke
    out["modifier_bag"] = out["modifier_bag"].fillna("")
    out.to_csv(COORDS_CSV, index=False)
    logger.info("Wrote coords → %s (%d rows)", COORDS_CSV, len(out))

    # Family distribution sanity check
    logger.info("Family counts: %s", out["family"].value_counts().to_dict())
    # Subtype top 10
    logger.info("Top 10 subtypes: %s", out["subtype"].value_counts().head(10).to_dict())


if __name__ == "__main__":
    run()
