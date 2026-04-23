"""Post-processing layer: apply expert review decisions to Step 3 synonym_groups.

Triggered by ``--review [path]`` on ``metaagent_run.steps.step3.new``. Reads the
decision JSON produced by manual review, applies it to the freshly-written
``output_file`` (the envelope containing ``synonym_groups``), and writes:

  - ``<output_file>``                                 — post-review canonical map
  - ``<output_stem>.pre_review<output_suffix>``       — backup of pre-review snapshot
  - ``step3_canonical_review_audit.json`` (sibling)   — per-decision audit trail

Only the canonical→aliases mapping is touched. The alias-keyed PMID indices
(``field_pmid_index``, ``field_pmid_env_index``) are unaffected by canonical
regrouping and are not rewritten. ``side_tags`` is canonical-keyed but audit-only
(no downstream consumer); it is left as produced by the main pipeline.

Supported actions (validated against the JSON file at load time):
  REMOVE_ALIAS       detach aliases; each becomes a singleton canonical
  SPLIT_CANONICAL    replace one canonical with N new ones; every original
                     member must be explicitly covered by some split_group
  MERGE_CANONICALS   union members of ``merge_with`` canonicals into primary
  RENAME_CANONICAL   rename a canonical key; membership unchanged

Decisions are applied in array order; later decisions see the effect of earlier
ones. Malformed decisions or references to nonexistent canonicals raise
``ReviewError`` before any file is mutated (aside from the backup).
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

LOGGER = logging.getLogger("step3.canonical_review")

VALID_ACTIONS = {
    "REMOVE_ALIAS",
    "SPLIT_CANONICAL",
    "MERGE_CANONICALS",
    "RENAME_CANONICAL",
}


class ReviewError(ValueError):
    """Raised when a review decision is malformed or references a
    canonical/alias that does not exist in the current state."""


# ─────────────────────────────────────────────────────────────────────
#  Load
# ─────────────────────────────────────────────────────────────────────

def load_decisions(path: Path) -> List[Dict[str, Any]]:
    """Load the decisions array. Accepts three shapes:
      (1) top-level JSON list of decision objects
      (2) object with key ``"decisions"`` mapping to such a list
      (3) object with ``"decisions"`` itself an object that contains an
          inner ``"decisions"`` list (one level of nesting — observed
          when AI tooling wraps the payload with its own metadata block)
    """
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        raw = data.get("decisions")
        if isinstance(raw, list):
            return raw
        if isinstance(raw, dict):
            inner = raw.get("decisions")
            if isinstance(inner, list):
                LOGGER.warning(
                    "Review file %s nests the decisions list one level "
                    "deeper than the canonical schema; unwrapping for "
                    "compatibility. Consider flattening at the source.",
                    path,
                )
                return inner
    raise ReviewError(
        f"Review file {path} must be a JSON list or an object with a "
        f"'decisions' list at the top level."
    )


# ─────────────────────────────────────────────────────────────────────
#  Core apply
# ─────────────────────────────────────────────────────────────────────

def apply_decisions(
    canonical_map: Dict[str, List[str]],
    decisions: List[Dict[str, Any]],
) -> Tuple[Dict[str, List[str]], List[Dict[str, Any]]]:
    """Apply decisions in order to a copy of ``canonical_map``.

    Returns (new_canonical_map, audit_log). Raises ``ReviewError`` on the
    first malformed decision or invalid reference; on failure, no partial
    state is returned.
    """
    groups: Dict[str, List[str]] = {c: list(a) for c, a in canonical_map.items()}
    audit: List[Dict[str, Any]] = []

    for idx, decision in enumerate(decisions):
        if not isinstance(decision, dict):
            raise ReviewError(f"decision[{idx}] is not a JSON object")
        canonical = decision.get("canonical")
        action = decision.get("action")
        if action not in VALID_ACTIONS:
            raise ReviewError(
                f"decision[{idx}] has invalid action {action!r}; "
                f"must be one of {sorted(VALID_ACTIONS)}"
            )
        if not isinstance(canonical, str) or not canonical:
            raise ReviewError(f"decision[{idx}] missing required string 'canonical'")

        if action == "REMOVE_ALIAS":
            entry = _apply_remove_alias(groups, decision, idx)
        elif action == "SPLIT_CANONICAL":
            entry = _apply_split(groups, decision, idx)
        elif action == "MERGE_CANONICALS":
            entry = _apply_merge(groups, decision, idx)
        else:  # RENAME_CANONICAL
            entry = _apply_rename(groups, decision, idx)

        entry.update(
            {
                "decision_index": idx,
                "canonical": canonical,
                "action": action,
                "reason": decision.get("reason"),
                "reviewer": decision.get("reviewer"),
                "decided_at": decision.get("decided_at"),
            }
        )
        audit.append(entry)
        LOGGER.info(
            "[Review %d/%d] %s on '%s' → %s",
            idx + 1, len(decisions), action, canonical,
            entry.get("summary", "applied"),
        )

    return groups, audit


# ─────────────────────────────────────────────────────────────────────
#  Per-action handlers
# ─────────────────────────────────────────────────────────────────────

def _apply_remove_alias(
    groups: Dict[str, List[str]],
    decision: Dict[str, Any],
    idx: int,
) -> Dict[str, Any]:
    canonical = decision["canonical"]
    aliases = decision.get("aliases_to_remove")
    if not isinstance(aliases, list) or not aliases:
        raise ReviewError(
            f"decision[{idx}] REMOVE_ALIAS needs non-empty list 'aliases_to_remove'"
        )
    if canonical not in groups:
        raise ReviewError(
            f"decision[{idx}] REMOVE_ALIAS: canonical '{canonical}' not found"
        )
    if canonical in aliases:
        raise ReviewError(
            f"decision[{idx}] REMOVE_ALIAS: cannot remove the canonical "
            f"itself ('{canonical}') — use SPLIT_CANONICAL or RENAME_CANONICAL"
        )
    members_set = set(groups[canonical])
    missing = [a for a in aliases if a not in members_set]
    if missing:
        raise ReviewError(
            f"decision[{idx}] REMOVE_ALIAS('{canonical}'): aliases not "
            f"members of this canonical: {missing}"
        )
    collision = [a for a in aliases if a in groups and a != canonical]
    if collision:
        raise ReviewError(
            f"decision[{idx}] REMOVE_ALIAS('{canonical}'): new singleton "
            f"would collide with existing canonical(s): {collision}"
        )

    remaining = [m for m in groups[canonical] if m not in set(aliases)]
    groups[canonical] = sorted(remaining)
    for alias in aliases:
        groups[alias] = [alias]
    return {
        "summary": f"detached {len(aliases)} alias(es) as singletons",
        "removed_aliases": list(aliases),
        "new_singletons": list(aliases),
        "parent_remaining_size": len(remaining),
    }


def _apply_split(
    groups: Dict[str, List[str]],
    decision: Dict[str, Any],
    idx: int,
) -> Dict[str, Any]:
    canonical = decision["canonical"]
    split_groups = decision.get("split_groups")
    if not isinstance(split_groups, list) or not split_groups:
        raise ReviewError(
            f"decision[{idx}] SPLIT_CANONICAL needs non-empty list 'split_groups'"
        )
    if canonical not in groups:
        raise ReviewError(
            f"decision[{idx}] SPLIT_CANONICAL: canonical '{canonical}' not found"
        )

    original_members = set(groups[canonical])
    covered: set[str] = set()
    prepared: List[Tuple[str, List[str]]] = []

    for si, sg in enumerate(split_groups):
        if not isinstance(sg, dict):
            raise ReviewError(
                f"decision[{idx}] SPLIT_CANONICAL.split_groups[{si}] is not an object"
            )
        new_canon = sg.get("new_canonical")
        members = sg.get("members")
        if not isinstance(new_canon, str) or not new_canon:
            raise ReviewError(
                f"decision[{idx}] split_groups[{si}] missing string 'new_canonical'"
            )
        if not isinstance(members, list) or not members:
            raise ReviewError(
                f"decision[{idx}] split_groups[{si}]='{new_canon}' needs non-empty 'members'"
            )
        not_in_original = [m for m in members if m not in original_members]
        if not_in_original:
            raise ReviewError(
                f"decision[{idx}] split_groups[{si}]='{new_canon}': members "
                f"not in original canonical '{canonical}': {not_in_original}"
            )
        already = [m for m in members if m in covered]
        if already:
            raise ReviewError(
                f"decision[{idx}] split_groups[{si}]='{new_canon}': members "
                f"already covered by an earlier split_group: {already}"
            )
        covered.update(members)
        prepared.append((new_canon, sorted(set(members))))

    uncovered = original_members - covered
    if uncovered:
        raise ReviewError(
            f"decision[{idx}] SPLIT_CANONICAL('{canonical}'): "
            f"{len(uncovered)} member(s) not covered by any split_group: "
            f"{sorted(uncovered)}"
        )

    for new_canon, _ in prepared:
        if new_canon != canonical and new_canon in groups:
            raise ReviewError(
                f"decision[{idx}] SPLIT_CANONICAL: new_canonical "
                f"'{new_canon}' collides with an existing canonical"
            )

    del groups[canonical]
    for new_canon, members in prepared:
        groups[new_canon] = members

    return {
        "summary": f"split into {len(prepared)} group(s)",
        "new_canonicals": [nc for nc, _ in prepared],
        "member_counts": {nc: len(m) for nc, m in prepared},
    }


def _apply_merge(
    groups: Dict[str, List[str]],
    decision: Dict[str, Any],
    idx: int,
) -> Dict[str, Any]:
    canonical = decision["canonical"]
    merge_with = decision.get("merge_with")
    if not isinstance(merge_with, list) or not merge_with:
        raise ReviewError(
            f"decision[{idx}] MERGE_CANONICALS needs non-empty list 'merge_with'"
        )
    if canonical not in groups:
        raise ReviewError(
            f"decision[{idx}] MERGE_CANONICALS: canonical '{canonical}' not found"
        )
    if canonical in merge_with:
        raise ReviewError(
            f"decision[{idx}] MERGE_CANONICALS: cannot merge '{canonical}' with itself"
        )
    missing = [m for m in merge_with if m not in groups]
    if missing:
        raise ReviewError(
            f"decision[{idx}] MERGE_CANONICALS('{canonical}'): merge_with "
            f"canonical(s) not found: {missing}"
        )

    merged: set[str] = set(groups[canonical])
    absorbed_counts: Dict[str, int] = {}
    for other in merge_with:
        absorbed_counts[other] = len(groups[other])
        merged.update(groups[other])
        del groups[other]
    groups[canonical] = sorted(merged)
    return {
        "summary": f"merged {len(merge_with)} canonical(s) into '{canonical}'",
        "absorbed": absorbed_counts,
        "new_size": len(merged),
    }


def _apply_rename(
    groups: Dict[str, List[str]],
    decision: Dict[str, Any],
    idx: int,
) -> Dict[str, Any]:
    canonical = decision["canonical"]
    new_name = decision.get("new_name")
    if not isinstance(new_name, str) or not new_name:
        raise ReviewError(
            f"decision[{idx}] RENAME_CANONICAL needs string 'new_name'"
        )
    if canonical not in groups:
        raise ReviewError(
            f"decision[{idx}] RENAME_CANONICAL: canonical '{canonical}' not found"
        )
    if new_name == canonical:
        return {"summary": "noop (new_name == canonical)", "new_name": new_name}
    if new_name in groups:
        raise ReviewError(
            f"decision[{idx}] RENAME_CANONICAL: new_name '{new_name}' "
            f"collides with an existing canonical"
        )
    groups[new_name] = groups.pop(canonical)
    return {"summary": f"renamed → '{new_name}'", "new_name": new_name}


# ─────────────────────────────────────────────────────────────────────
#  File-level orchestration (called from the step3 orchestrator)
# ─────────────────────────────────────────────────────────────────────

def run_review_postprocess(
    output_file: Path,
    review_decisions_file: Path,
    audit_file: Path,
) -> Dict[str, Any]:
    """Apply ``review_decisions_file`` to ``output_file`` in place.

    Writes a pre-review backup next to ``output_file`` and an audit log
    to ``audit_file``. Returns a small summary dict for logging.
    """
    if not review_decisions_file.exists():
        raise FileNotFoundError(
            f"Review decisions file not found: {review_decisions_file}"
        )
    if not output_file.exists():
        raise FileNotFoundError(
            f"Step 3 output file not found; run the pipeline first: {output_file}"
        )

    decisions = load_decisions(review_decisions_file)
    with output_file.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    pre_groups = payload.get("synonym_groups")
    if not isinstance(pre_groups, dict):
        raise ReviewError(
            f"{output_file} has no 'synonym_groups' dict to apply review to"
        )

    backup_file = output_file.with_name(
        f"{output_file.stem}.pre_review{output_file.suffix}"
    )
    with backup_file.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
    LOGGER.info("Backed up pre-review snapshot → %s", backup_file)

    new_groups, audit = apply_decisions(pre_groups, decisions)

    new_payload = dict(payload)
    new_payload["synonym_groups"] = new_groups
    new_payload["review_applied"] = {
        "source": str(review_decisions_file),
        "total_decisions": len(decisions),
        "pre_canonical_count": len(pre_groups),
        "post_canonical_count": len(new_groups),
    }
    with output_file.open("w", encoding="utf-8") as file:
        json.dump(new_payload, file, ensure_ascii=False, indent=2)

    audit_file.parent.mkdir(parents=True, exist_ok=True)
    with audit_file.open("w", encoding="utf-8") as file:
        json.dump(
            {
                "source": str(review_decisions_file),
                "pre_canonical_count": len(pre_groups),
                "post_canonical_count": len(new_groups),
                "decisions": audit,
            },
            file,
            ensure_ascii=False,
            indent=2,
        )
    LOGGER.info("Wrote post-review canonical map → %s", output_file)
    LOGGER.info("Wrote review audit log → %s", audit_file)

    return {
        "pre_canonical_count": len(pre_groups),
        "post_canonical_count": len(new_groups),
        "decisions_applied": len(decisions),
    }
