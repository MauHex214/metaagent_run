import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import numpy as np

LOGGER = logging.getLogger(__name__)


def compute_env_paper_counts(
    paper_env_map: Dict[str, List[str]],
) -> Dict[str, int]:
    from .config import VALID_SUB_ENVS

    env_counts = {env: 0 for env in VALID_SUB_ENVS}
    for envs in paper_env_map.values():
        valid = [env for env in envs if env in VALID_SUB_ENVS]
        deduped = list(dict.fromkeys(valid))
        if len(deduped) == 1:
            env_counts[deduped[0]] += 1
    return env_counts


def compute_env_coverage_counts(
    pmid_env_index: Dict[str, Dict[str, List[str]]],
) -> Dict[str, int]:
    from .config import VALID_SUB_ENVS

    env_unions = {env: set() for env in VALID_SUB_ENVS}
    for env_map in pmid_env_index.values():
        if not isinstance(env_map, dict):
            continue
        for env in VALID_SUB_ENVS:
            pmids = env_map.get(env, [])
            if isinstance(pmids, list):
                env_unions[env].update(str(pmid) for pmid in pmids if pmid)
    return {env: len(pmids) for env, pmids in env_unions.items()}


def _expand_fields_for_profile(
    seed_fields: Set[str],
    synonym_groups: Dict[str, List[str]],
    field_to_slot: Dict[str, str],
    allowed_slot: str,
    available_fields: Set[str],
) -> Set[str]:
    expanded = set(seed_fields)
    for field in list(seed_fields):
        norm = field.lower()
        for canonical, members in synonym_groups.items():
            members_lower = {member.lower() for member in members}
            if norm not in members_lower and canonical.lower() != norm:
                continue
            for member in members:
                if member not in available_fields:
                    continue
                mapped_slot = field_to_slot.get(member.lower())
                if mapped_slot is not None and mapped_slot != allowed_slot:
                    continue
                expanded.add(member)
            break
    return expanded


def compute_env_profile(
    pmid_env_index: Dict[str, Dict[str, List[str]]],
    mapped_results: List[Dict[str, Any]],
    synonym_groups: Dict[str, List[str]],
    env_denominator_counts: Dict[str, int],
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    from .config import VALID_SUB_ENVS

    field_to_slot: Dict[str, str] = {}
    slot_to_raw: Dict[str, Set[str]] = defaultdict(set)
    for record in mapped_results:
        field = str(record["field"])
        slot = str(record["mixs_slot"])
        field_to_slot[field.lower()] = slot
        if slot != "UNMAPPED":
            slot_to_raw[slot].add(field)

    available_fields = set(pmid_env_index.keys())
    profiles: Dict[str, Dict[str, Dict[str, Any]]] = {}

    for slot, raw_fields in slot_to_raw.items():
        expanded = _expand_fields_for_profile(raw_fields, synonym_groups, field_to_slot, slot, available_fields)
        slot_profile: Dict[str, Dict[str, Any]] = {}
        for env in VALID_SUB_ENVS:
            pmid_union: Set[str] = set()
            for field in expanded:
                pmid_union.update(pmid_env_index.get(field, {}).get(env, []))
            denominator = env_denominator_counts.get(env, 0)
            slot_profile[env] = {
                "pmid_count": len(pmid_union),
                "pmid_pct": round(len(pmid_union) / denominator * 100, 2) if denominator else 0.0,
            }
        profiles[slot] = slot_profile

    for record in mapped_results:
        if str(record["mixs_slot"]) != "UNMAPPED":
            continue
        field = str(record["field"])
        expanded = _expand_fields_for_profile({field}, synonym_groups, field_to_slot, "UNMAPPED", available_fields)
        field_profile: Dict[str, Dict[str, Any]] = {}
        for env in VALID_SUB_ENVS:
            pmid_union: Set[str] = set()
            for expanded_field in expanded:
                pmid_union.update(pmid_env_index.get(expanded_field, {}).get(env, []))
            denominator = env_denominator_counts.get(env, 0)
            field_profile[env] = {
                "pmid_count": len(pmid_union),
                "pmid_pct": round(len(pmid_union) / denominator * 100, 2) if denominator else 0.0,
            }
        profiles[field] = field_profile

    return profiles


def compute_frequency(
    mapped_results: List[Dict[str, Any]],
    pmid_index: Dict[str, List[str]],
    synonym_groups: Dict[str, List[str]],
) -> Dict[str, Dict[str, Any]]:
    slot_to_raw: Dict[str, Set[str]] = defaultdict(set)
    slot_to_info: Dict[str, Dict[str, str]] = {}
    field_to_slot: Dict[str, str] = {}

    for record in mapped_results:
        field_to_slot[str(record["field"]).lower()] = str(record["mixs_slot"])

    for record in mapped_results:
        field = str(record["field"])
        slot = str(record["mixs_slot"])
        if slot == "UNMAPPED":
            continue
        slot_to_raw[slot].add(field)
        if slot not in slot_to_info:
            slot_to_info[slot] = {
                "mixs_title": str(record["mixs_title"]),
                "mixs_slot": slot,
            }

    pmid_index_keys = set(pmid_index.keys())
    for slot, raw_fields in slot_to_raw.items():
        expanded = set(raw_fields)
        for field in list(raw_fields):
            norm = field.lower()
            for canonical, members in synonym_groups.items():
                members_lower = {member.lower() for member in members}
                if norm in members_lower or canonical.lower() == norm:
                    for member in members:
                        if member not in pmid_index_keys:
                            continue
                        mapped_slot = field_to_slot.get(member.lower())
                        if mapped_slot is not None and mapped_slot != slot:
                            continue
                        expanded.add(member)
                    break
        slot_to_raw[slot] = expanded

    result: Dict[str, Dict[str, Any]] = {}
    for slot, raw_fields in slot_to_raw.items():
        pmid_union: Set[str] = set()
        contributing: List[str] = []
        for field in raw_fields:
            pmids = pmid_index.get(field, [])
            if pmids:
                pmid_union.update(pmids)
                contributing.append(field)
        result[slot] = {
            **slot_to_info[slot],
            "pmid_count": len(pmid_union),
            "raw_fields": sorted(raw_fields),
            "contributing_fields": sorted(contributing),
            "raw_field_count": len(raw_fields),
        }
    return result


def compute_unmapped_frequency(
    mapped_results: List[Dict[str, Any]],
    pmid_index: Dict[str, List[str]],
    synonym_groups: Dict[str, List[str]],
) -> Dict[str, Dict[str, Any]]:
    pmid_index_keys = set(pmid_index.keys())
    field_to_slot: Dict[str, str] = {}
    for record in mapped_results:
        field_to_slot[str(record["field"]).lower()] = str(record["mixs_slot"])

    result: Dict[str, Dict[str, Any]] = {}
    for record in mapped_results:
        if str(record["mixs_slot"]) != "UNMAPPED":
            continue

        field = str(record["field"])
        expanded: Set[str] = {field}
        norm = field.lower()
        for canonical, members in synonym_groups.items():
            members_lower = {member.lower() for member in members}
            if norm in members_lower or canonical.lower() == norm:
                for member in members:
                    if member not in pmid_index_keys:
                        continue
                    mapped_slot = field_to_slot.get(member.lower())
                    if mapped_slot is not None and mapped_slot != "UNMAPPED":
                        continue
                    expanded.add(member)
                break

        pmid_union: Set[str] = set()
        contributing: List[str] = []
        for expanded_field in expanded:
            pmids = pmid_index.get(expanded_field, [])
            if pmids:
                pmid_union.update(pmids)
                contributing.append(expanded_field)

        result[field] = {
            "field": field,
            "pmid_count": len(pmid_union),
            "raw_fields": sorted(expanded),
            "contributing_fields": sorted(contributing),
            "raw_field_count": len(expanded),
            "reason": record.get("reason", ""),
        }
    return result


def compute_total_sampled_pmids(pmid_index: Dict[str, List[str]]) -> int:
    all_pmids: Set[str] = set()
    for pmids in pmid_index.values():
        all_pmids.update(pmids)
    return len(all_pmids)


def compute_tier2_threshold(tier2_min_pmid: int = 3) -> int:
    """返回 Tier 2 绝对 PMID 阈值。"""
    LOGGER.info("Tier 2 阈值: %d PMIDs (绝对阈值)", tier2_min_pmid)
    return tier2_min_pmid


def compute_tiered_output(
    freq_data: Dict[str, Dict[str, Any]],
    unmapped_freq: Dict[str, Dict[str, Any]],
    mapped_results: List[Dict[str, Any]],
    total_sampled_pmids: int,
    tier2_min_pmid: int = 3,
    exclusion_set: Optional[Set[str]] = None,
    exclusion_entries: Optional[List[Dict[str, str]]] = None,
    env_profiles: Optional[Dict[str, Dict[str, Dict[str, Any]]]] = None,
    env_paper_counts: Optional[Dict[str, int]] = None,
    env_coverage_counts: Optional[Dict[str, int]] = None,
) -> Dict[str, Any]:
    excluded_fields = exclusion_set or set()
    exclusion_reason = {
        entry["field"]: entry.get("reason", "manually excluded")
        for entry in (exclusion_entries or [])
        if "field" in entry
    }

    tier1: List[Dict[str, Any]] = []
    for slot, data in sorted(freq_data.items(), key=lambda item: item[1]["pmid_count"], reverse=True):
        slot_fields = [
            {"field": record["field"], "confidence": record["confidence"]}
            for record in mapped_results
            if record["mixs_slot"] == slot
        ]
        pct = data["pmid_count"] / total_sampled_pmids * 100 if total_sampled_pmids else 0
        entry = {
            "mixs_slot": slot,
            "mixs_title": data["mixs_title"],
            "pmid_count": data["pmid_count"],
            "pmid_pct": round(pct, 2),
            "justification": f"MIxS standard alignment ({slot} - {data['mixs_title']})",
            "mapped_fields": slot_fields,
            "raw_field_count": data["raw_field_count"],
            "contributing_fields": data.get("contributing_fields", []),
        }
        if env_profiles is not None and slot in env_profiles:
            entry["env_profile"] = env_profiles[slot]
        tier1.append(entry)

    tier2: List[Dict[str, Any]] = []
    tier3: List[Dict[str, Any]] = []
    excluded: List[Dict[str, Any]] = []
    for field, data in sorted(unmapped_freq.items(), key=lambda item: item[1]["pmid_count"], reverse=True):
        pct = data["pmid_count"] / total_sampled_pmids * 100 if total_sampled_pmids else 0
        entry = {
            "field": field,
            "pmid_count": data["pmid_count"],
            "pmid_pct": round(pct, 2),
            "raw_field_count": data["raw_field_count"],
            "contributing_fields": data.get("contributing_fields", []),
            "unmapped_reason": data.get("reason", ""),
        }
        if env_profiles is not None and field in env_profiles:
            entry["env_profile"] = env_profiles[field]
        if field in excluded_fields:
            entry["justification"] = f"Excluded: {exclusion_reason.get(field, 'manually excluded')}"
            excluded.append(entry)
            continue
        if data["pmid_count"] >= tier2_min_pmid:
            entry["justification"] = (
                "High literature frequency ({} PMIDs, {:.1f}%) >= absolute threshold ({} PMIDs); no MIxS equivalent".format(
                    data["pmid_count"], pct, tier2_min_pmid
                )
            )
            tier2.append(entry)
        else:
            entry["justification"] = (
                "Low literature frequency ({} PMIDs, {:.1f}%), below Tier 2 threshold ({} PMIDs)".format(
                    data["pmid_count"], pct, tier2_min_pmid
                )
            )
            tier3.append(entry)

    metadata = {
        "pipeline": "step3 -> synonym_fix -> step4_metadata_extend (refactored module) -> tier output",
        "tier_logic": {
            "tier1": "MIxS-aligned: field mapped to MIxS standard term by LLM",
            "tier2": "High-frequency domain-specific: unmapped but PMID count >= {} (absolute threshold)".format(
                tier2_min_pmid
            ),
            "tier3": "Low-frequency / noise: unmapped and PMID count < threshold",
            "excluded": "Manually excluded: not environmental sample metadata (see tier_exclusion_list.json)",
        },
        "tier2_threshold_pmids": tier2_min_pmid,
        "tier2_threshold_method": "absolute cutoff ({} PMIDs)".format(tier2_min_pmid),
        "total_sampled_pmids": total_sampled_pmids,
        "counts": {
            "tier1_mixs_terms": len(tier1),
            "tier2_high_freq": len(tier2),
            "tier3_low_freq_or_noise": len(tier3),
            "excluded": len(excluded),
            "total_fields_processed": len(mapped_results),
        },
        "next_step": "Human review of Tier 1 mapping accuracy + Tier 2 inclusion decisions -> step5 extraction target list",
    }
    if env_paper_counts is not None:
        metadata["env_paper_counts"] = env_paper_counts
        metadata["env_paper_counts_note"] = "Only single-sub-environment papers are counted from paper_env_map; this is a reference total, not the denominator used for env_profile percentages."
    if env_coverage_counts is not None:
        metadata["env_coverage_counts"] = env_coverage_counts
        metadata["env_coverage_counts_note"] = (
            "Per-environment percentages use the number of PMIDs covered by field_pmid_env_index in each environment as the denominator."
        )

    return {
        "metadata": metadata,
        "tier1": tier1,
        "tier2": tier2,
        "tier3": tier3,
        "excluded": excluded,
    }


def save_tiered_output(tiered: Dict[str, Any], output_dir: Any, filename: str = "step4b_tiered_extraction_targets.json") -> None:
    path = output_dir / filename
    with path.open("w", encoding="utf-8") as file:
        json.dump(tiered, file, ensure_ascii=False, indent=2)
    LOGGER.info("Tier 分级结果: %s", path)


# ═══════════════════════════════════════════════════════════════
# Environment field classification & extraction target output
# ═══════════════════════════════════════════════════════════════

def classify_env_fields(
    tiered: Dict[str, Any],
    exclusion_set: Optional[Set[str]] = None,
    min_pmid: int = 5,
    min_env_pct: float = 0.5,
    universal_min_envs: int = 3,
    fisher_alpha: float = 0.05,
    enrichment_threshold: float = 2.0,
) -> tuple:
    """
    Build a unified field list from tiered output, apply exclusion
    filtering, inclusion criterion, enrichment + Fisher test, and
    classify into Universal / Shared / Signature.

    A field is considered "present" in env E iff it is reported by
    ≥ min_pmid papers AND ≥ min_env_pct % of env-E papers. The dual
    constraint (absolute floor + relative coverage) prevents small
    envs (e.g. Wetlands, n≈3150) from being over-represented by
    borderline-frequency fields.

    Classification is enrichment-first: a field strongly enriched in
    one env (Fisher p < alpha AND enrichment ≥ enrichment_threshold)
    is labelled Signature regardless of how many envs it is "present"
    in. Fields with no significantly-enriched env fall back to a
    presence-count rule (≥ universal_min_envs → Universal, 2 → Shared,
    1 → Signature). This captures env-specific fields whose absolute
    pct is modest but whose cross-env contrast is scientifically
    decisive (e.g. primary_production is Open_ocean-distinctive at
    only ~0.9% coverage — classification by count alone would bury it
    in Universal).

    Returns (all_entries, included, excluded_low_freq, env_n, total_papers).
    """
    from .config import VALID_SUB_ENVS
    from scipy.stats import fisher_exact

    envs = sorted(VALID_SUB_ENVS)
    excluded_fields = exclusion_set or set()
    env_n = {e: int(tiered["metadata"]["env_coverage_counts"][e]) for e in envs}
    total_papers = int(tiered["metadata"]["total_sampled_pmids"])

    def _is_present(entry: Dict[str, Any], env: str) -> bool:
        """Dual-threshold presence: absolute floor AND relative coverage."""
        n = entry.get(env + "_n", 0)
        pct = entry.get(env + "_pct", 0.0)
        return n >= min_pmid and pct >= min_env_pct

    # 1. Build unified field list (Tier 1 + 2 + 3, minus exclusions)
    all_entries: List[Dict[str, Any]] = []
    for tier_key, tier_num in [("tier1", 1), ("tier2", 2), ("tier3", 3)]:
        for t in tiered.get(tier_key, []):
            ep = t.get("env_profile", {})
            label = t["mixs_title"] if tier_num == 1 else t["field"]
            raw_field = t.get("field", label)

            # Tier 2/3: exclude if raw field is in exclusion list
            if tier_num != 1:
                if raw_field in excluded_fields:
                    continue
            # All tiers: exclude if ALL contributing fields are excluded
            contribs = set(t.get("contributing_fields", []))
            if contribs and contribs.issubset(excluded_fields):
                continue

            entry: Dict[str, Any] = dict(
                label=label,
                slot=t.get("mixs_slot", raw_field),
                tier=tier_num,
                mapped=(tier_num == 1),
                total_pmid=t["pmid_count"],
                total_pct=t["pmid_pct"],
                contributing_fields=t.get("contributing_fields", []),
                raw_field_count=t.get("raw_field_count", 0),
            )
            for env in envs:
                if env in ep:
                    entry[env + "_n"]   = ep[env]["pmid_count"]
                    entry[env + "_pct"] = ep[env]["pmid_pct"]
                else:
                    entry[env + "_n"]   = 0
                    entry[env + "_pct"] = 0.0
            all_entries.append(entry)

    # 2. Inclusion criterion: field is a step5 target if it is "present"
    #    (dual-threshold) in at least one env.
    included: List[Dict[str, Any]] = []
    excluded_low: List[Dict[str, Any]] = []
    for e in all_entries:
        if any(_is_present(e, env) for env in envs):
            included.append(e)
        else:
            excluded_low.append(e)

    # 3. Enrichment + Fisher exact test
    for e in included:
        pcts = [e[env + "_pct"] for env in envs]
        mean_pct = float(np.mean(pcts)) if float(np.mean(pcts)) > 0 else 1e-9
        e["enrichment"] = {}
        e["fisher_p"] = {}
        for env in envs:
            e["enrichment"][env] = e[env + "_pct"] / mean_pct
            a = e[env + "_n"]
            b = e["total_pmid"] - a
            c = env_n[env] - a
            d = (total_papers - env_n[env]) - b
            # 防护：剔除多环境论文后理论上不会出现负值，
            # 但仍保留下界保护以应对数据质量异常
            if c < 0 or d < 0:
                LOGGER.warning(
                    "Fisher 列联表出现负值: field=%s env=%s a=%d b=%d c=%d d=%d，跳过检验",
                    e["label"], env, a, b, c, d,
                )
                e["fisher_p"][env] = 1.0
            elif a <= 0:
                e["fisher_p"][env] = 1.0
            else:
                _, p = fisher_exact([[a, b], [c, d]], alternative="greater")
                e["fisher_p"][env] = p

    # 4. Classify: enrichment-first, fall back to presence-count.
    classified_by_enrichment = 0
    classified_by_count = 0
    for e in included:
        envs_with_min = [env for env in envs if _is_present(e, env)]
        e["n_envs_present"] = len(envs_with_min)
        e["envs_present"] = envs_with_min
        # Env(s) where this field is meaningfully over-represented
        # relative to the corpus mean. Used for classification (strict
        # threshold) AND reported for downstream annotation.
        e["sig_envs"] = [
            env for env in envs
            if e["fisher_p"][env] < fisher_alpha
               and e["enrichment"][env] >= enrichment_threshold
        ]
        n_sig = len(e["sig_envs"])
        if n_sig >= 1:
            classified_by_enrichment += 1
            e["classification_basis"] = "enrichment"
            if n_sig == 1:
                e["category"] = "Signature"
            elif n_sig == 2:
                e["category"] = "Shared"
            else:  # n_sig ≥ 3 → strongly present across the majority of envs
                e["category"] = "Universal"
        else:
            # No single env shows strong enrichment → fall back to count
            classified_by_count += 1
            e["classification_basis"] = "count"
            n_present = len(envs_with_min)
            if n_present >= universal_min_envs:
                e["category"] = "Universal"
            elif n_present == 2:
                e["category"] = "Shared"
            else:
                e["category"] = "Signature"

    cat_order = {"Universal": 0, "Shared": 1, "Signature": 2}
    included.sort(key=lambda x: (cat_order.get(x["category"], 9), -x["total_pmid"]))

    LOGGER.info(
        "classify_env_fields: %d entries -> %d included, %d below threshold",
        len(all_entries), len(included), len(excluded_low),
    )
    LOGGER.info(
        "  classification basis: %d by enrichment (Fisher p<%.3g, enrichment≥%.1f),"
        " %d by presence count",
        classified_by_enrichment, fisher_alpha, enrichment_threshold,
        classified_by_count,
    )
    for cat in ["Universal", "Shared", "Signature"]:
        n = sum(1 for e in included if e["category"] == cat)
        LOGGER.info("  %s: %d fields", cat, n)

    return all_entries, included, excluded_low, env_n, total_papers


def save_env_extraction_targets(
    included: List[Dict[str, Any]],
    all_entries: List[Dict[str, Any]],
    excluded_low_freq: List[Dict[str, Any]],
    env_n: Dict[str, int],
    total_papers: int,
    output_dir: Any,
    min_pmid: int = 5,
    min_env_pct: float = 0.5,
    universal_min_envs: int = 3,
    filename: str = "step4b_env_extraction_targets.json",
) -> None:
    """Write env_extraction_targets.json for step 5."""
    from .config import VALID_SUB_ENVS

    envs = sorted(VALID_SUB_ENVS)
    env_label = {"Open_ocean": "Open Ocean", "Coastal_waters": "Coastal Waters",
                 "Lake": "Lake", "Wetlands": "Wetlands"}

    def _is_present(e: Dict[str, Any], env: str) -> bool:
        return e.get(env + "_n", 0) >= min_pmid and e.get(env + "_pct", 0.0) >= min_env_pct

    def field_record(e: Dict[str, Any], env: Optional[str] = None) -> Dict[str, Any]:
        rec = dict(
            field=e["label"], slot=e["slot"], tier=e["tier"],
            mapped=e["mapped"], total_pmid=e["total_pmid"],
            total_pct=e["total_pct"], category=e["category"],
        )
        if env:
            rec["env_pmid"] = e[env + "_n"]
            rec["env_pct"]  = e[env + "_pct"]
            rec["enrichment"] = round(e["enrichment"][env], 2)
            rec["fisher_p"] = round(e["fisher_p"][env], 4)
        return rec

    presence_rule = (
        "\u2265 {} PMIDs AND \u2265 {}% of env papers".format(min_pmid, min_env_pct)
    )
    output: Dict[str, Any] = {
        "metadata": {
            "description": "Per-environment extraction targets for step 5",
            "inclusion_criterion": "present in at least one env ({})".format(presence_rule),
            "presence_rule": presence_rule,
            "classification": {
                "Universal": "present in \u2265 {} envs ({})".format(universal_min_envs, presence_rule),
                "Shared":    "present in exactly 2 envs ({})".format(presence_rule),
                "Signature": "present in only 1 env ({}); fingerprint field".format(presence_rule),
            },
            "fisher_test_role": "Annotation only (* marker) \u2014 not used for classification",
            "total_sampled_pmids": total_papers,
            "env_paper_counts": env_n,
            "field_counts": {
                "total_discovered": len(all_entries),
                "included": len(included),
                "excluded_low_freq": len(excluded_low_freq),
            },
            "category_counts": {
                cat: sum(1 for e in included if e["category"] == cat)
                for cat in ["Universal", "Shared", "Signature"]
            },
        },
        "global_fields": {
            cat: [field_record(e) for e in included if e["category"] == cat]
            for cat in ["Universal", "Shared", "Signature"]
        },
        "per_environment": {},
    }
    for env in envs:
        env_fields = sorted(
            [e for e in included if _is_present(e, env)],
            key=lambda x: -x[env + "_pct"],
        )
        output["per_environment"][env] = {
            "label": env_label.get(env, env),
            "n_papers": env_n[env],
            "n_fields": len(env_fields),
            "n_universal": sum(1 for e in env_fields if e["category"] == "Universal"),
            "n_shared":    sum(1 for e in env_fields if e["category"] == "Shared"),
            "n_signature": sum(1 for e in env_fields if e["category"] == "Signature"),
            "n_unmapped":  sum(1 for e in env_fields if not e["mapped"]),
            "fields": [field_record(e, env) for e in env_fields],
        }

    path = Path(output_dir) / filename
    with path.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    LOGGER.info("Saved env_extraction_targets.json: %s", path)


def save_supplementary_fields(
    excluded_low_freq: List[Dict[str, Any]],
    output_dir: Any,
    min_pmid: int = 3,
    filename: str = "step4b_supplementary_low_freq_fields.json",
) -> None:
    """Write supplementary_low_freq_fields.json (appendix)."""
    from .config import VALID_SUB_ENVS

    envs = sorted(VALID_SUB_ENVS)
    records = []
    for e in excluded_low_freq:
        rec = dict(
            field=e["label"], slot=e["slot"], tier=e["tier"],
            mapped=e["mapped"], total_pmid=e["total_pmid"], total_pct=e["total_pct"],
        )
        for env in envs:
            rec[env + "_n"] = e[env + "_n"]
        records.append(rec)
    records.sort(key=lambda x: -x["total_pmid"])

    output = {
        "metadata": {
            "description": "Fields below inclusion threshold (< {} PMIDs in all envs)".format(min_pmid),
            "note": "Not included in extraction targets; listed for completeness",
            "count": len(records),
        },
        "fields": records,
    }
    path = Path(output_dir) / filename
    with path.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    LOGGER.info("Saved supplementary_low_freq_fields.json: %s", path)
