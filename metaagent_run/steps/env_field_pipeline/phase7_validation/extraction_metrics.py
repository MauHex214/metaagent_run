"""Per-target tries/success aggregation across phase7 rounds.

For each canonical target field in env6_extraction_targets.json, count:
  - n_tries:    # papers where ANY section's step2 metadata_keys_found
                contains the target name or one of its aliases
                (= the paper had an opportunity to express this field)
  - n_success:  # papers where step5 extracted a metadata item whose
                raw_field matches the target name or aliases
                (= step5 actually produced something for this field)

Reads:
  - env_field_pipeline_output/env6_extraction_targets.json
  - env_field_pipeline_output/phase7_validation/paper_pool_index.json
  - env_field_pipeline_output/phase7_validation/sampled_papers_round_NN.csv (rounds 1..N)
  - env_field_pipeline_output/phase7_validation/round_NN_step5_output.json (rounds 1..N)
  - relation_v1_step2_relation_output.json  (for metadata_keys_found per section)

Writes:
  - extraction_success_round_NN.csv     (this round only)
  - extraction_success_cumulative.csv   (rounds 1..N, growing as more rounds land)
  - coverage_per_round.csv              (one row per round: papers, targets_tried, targets_succeeded)

NOTE on limitation: success_rate reflects whether step5 produced a non-null
value, NOT whether the value is correct. Value-level precision requires
sampled human review (see briefing §7.4).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable

from metaagent_run.steps.env_field_pipeline import config as ep_config
from metaagent_run.steps.env_field_pipeline.phase7_validation import (
    PHASE7_DIR,
    ensure_phase7_dir,
)

import os
# Default to v1b (raw_key_expansion injected) if it exists, else v1.
_V1B = ep_config.OUTPUT_DIR / "env6_extraction_targets_v1b.json"
_V1 = ep_config.OUTPUT_DIR / "env6_extraction_targets.json"
ENV6_PATH = Path(os.environ.get("PHASE7_ENV6_PATH",
                                str(_V1B if _V1B.exists() else _V1)))
SAMPLE_FAILURE_LIMIT = 10  # cap failure pmids in CSV cell


# ─── Alias indexing ─────────────────────────────────────────────────────

def _normalize_key(s: str) -> str:
    """Match the rule the section-filter uses (cf. metadata_extractor)."""
    return str(s).lower().strip()


def load_target_catalog() -> tuple[dict, dict]:
    """Returns (target_meta, alias_to_target):
      target_meta: target_name → {tier, subtype, envs (set of envs that list it)}
      alias_to_target: normalized alias → target_name
                       (collisions: prefer the target whose own name == alias;
                        otherwise first-write-wins; warn on real collisions.)
    """
    with open(ENV6_PATH, "r", encoding="utf-8") as f:
        env6 = json.load(f)

    target_meta: dict[str, dict] = {}
    alias_to_target: dict[str, str] = {}

    for env_name, env_block in env6.get("per_environment", {}).items():
        for fdef in env_block.get("fields", []):
            name = fdef["field"]
            tm = target_meta.setdefault(name, {
                "tier": fdef.get("tier"),
                "subtype": fdef.get("subtype"),
                "envs": set(),
                "aliases": set(),
            })
            tm["envs"].add(env_name)
            tm["aliases"].add(name)
            for a in fdef.get("aliases") or []:
                tm["aliases"].add(a)

    collisions = 0
    for tname, tm in target_meta.items():
        for a in tm["aliases"]:
            key = _normalize_key(a)
            if not key:
                continue
            if key in alias_to_target and alias_to_target[key] != tname:
                # Prefer when the alias literally equals the target's own name
                if _normalize_key(tname) == key:
                    alias_to_target[key] = tname
                else:
                    collisions += 1
            else:
                alias_to_target[key] = tname
    if collisions:
        print(f"  warning: {collisions} alias→target collisions "
              f"(first-write-wins for non-canonical aliases)", file=sys.stderr)

    return target_meta, alias_to_target


# ─── Step2 lookup: pmid → set of normalized step2 keys ──────────────────

def load_pmid_step2_keys(pool_index: dict, pmids_filter: set[str]) -> dict[str, set[str]]:
    """For each pmid in pmids_filter, the union of metadata_keys_found across
    all its metadata-bearing sections (normalized)."""
    print(f"  scanning step2 for {len(pmids_filter)} pmids ...", flush=True)
    out: dict[str, set[str]] = defaultdict(set)
    with open(ep_config.STEP2_INPUT, "r", encoding="utf-8") as f:
        records = json.load(f)
    for r in records:
        pmid = r.get("pmid")
        if pmid not in pmids_filter:
            continue
        keys = r.get("metadata_keys_found")
        if not isinstance(keys, list):
            continue
        for k in keys:
            nk = _normalize_key(k)
            if nk:
                out[pmid].add(nk)
    return out


# ─── Round IO ───────────────────────────────────────────────────────────

def round_paths(round_num: int) -> tuple[Path, Path]:
    sampled = PHASE7_DIR / f"sampled_papers_round_{round_num:02d}.csv"
    output = PHASE7_DIR / f"round_{round_num:02d}_step5_output.json"
    return sampled, output


def load_round_pmids(round_num: int) -> set[str]:
    sampled, _ = round_paths(round_num)
    with open(sampled, "r", encoding="utf-8") as f:
        return {row["pmid"] for row in csv.DictReader(f)}


def load_round_step5_output(round_num: int) -> list[dict]:
    _, out_path = round_paths(round_num)
    with open(out_path, "r", encoding="utf-8") as f:
        return json.load(f)


def discover_rounds() -> list[int]:
    rounds = []
    for p in PHASE7_DIR.glob("sampled_papers_round_*.csv"):
        try:
            r = int(p.stem.rsplit("_", 1)[-1])
            out = PHASE7_DIR / f"round_{r:02d}_step5_output.json"
            if out.exists():
                rounds.append(r)
        except ValueError:
            continue
    return sorted(rounds)


# ─── Aggregation ────────────────────────────────────────────────────────

def aggregate(
    rounds: Iterable[int],
    target_meta: dict,
    alias_to_target: dict[str, str],
) -> tuple[dict, dict, set[str]]:
    """Returns (tries, success, all_pmids):
        tries[target] = set of pmids where target was offered (key in step2)
        success[target] = set of pmids where target was extracted (raw_field hit)
        all_pmids = union of pmids across rounds
    """
    rounds = list(rounds)
    all_pmids: set[str] = set()
    for r in rounds:
        all_pmids |= load_round_pmids(r)

    # Step2 lookup once for the union pmid set
    pool_index = json.load(open(PHASE7_DIR / "paper_pool_index.json", "r", encoding="utf-8"))
    pmid_step2_keys = load_pmid_step2_keys(pool_index, all_pmids)

    tries: dict[str, set[str]] = defaultdict(set)
    success: dict[str, set[str]] = defaultdict(set)

    # Tries: a target is "tried" for a paper if ANY of its step2 keys
    # normalizes to one of the target's aliases.
    for pmid, keys in pmid_step2_keys.items():
        for k in keys:
            t = alias_to_target.get(k)
            if t:
                tries[t].add(pmid)

    # Success: a target is "successful" for a paper if step5 output for that
    # paper contains any metadata item whose raw_field maps to this target.
    for r in rounds:
        for paper in load_round_step5_output(r):
            pmid = paper.get("pmid")
            for sample in paper.get("samples", []):
                for m in sample.get("metadata", []) or []:
                    raw = _normalize_key(m.get("raw_field", ""))
                    if not raw:
                        continue
                    t = alias_to_target.get(raw)
                    if t:
                        success[t].add(pmid)

    return tries, success, all_pmids


# ─── Output ─────────────────────────────────────────────────────────────

def write_success_csv(
    path: Path,
    tries: dict[str, set[str]],
    success: dict[str, set[str]],
    target_meta: dict,
) -> None:
    rows = []
    for t, meta in target_meta.items():
        ntries = len(tries.get(t, set()))
        nsucc = len(success.get(t, set()))
        rate = round(nsucc / ntries, 4) if ntries > 0 else 0.0
        failures = sorted(tries.get(t, set()) - success.get(t, set()))
        rows.append({
            "target": t,
            "tier": meta.get("tier"),
            "subtype": meta.get("subtype"),
            "envs": ";".join(sorted(meta.get("envs", []))),
            "n_tries": ntries,
            "n_success": nsucc,
            "success_rate": rate,
            "n_failure_pmids": len(failures),
            "sample_failure_pmids": ";".join(failures[:SAMPLE_FAILURE_LIMIT]),
        })
    rows.sort(key=lambda r: (-r["n_tries"], r["target"]))
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [
            "target", "tier", "subtype", "envs", "n_tries", "n_success",
            "success_rate", "n_failure_pmids", "sample_failure_pmids"])
        w.writeheader()
        w.writerows(rows)


def write_coverage_per_round(
    path: Path,
    target_meta: dict,
    alias_to_target: dict[str, str],
) -> None:
    """One row per round, cumulative through that round."""
    rounds = discover_rounds()
    if not rounds:
        return
    fieldnames = ["round", "cumulative_papers", "papers_with_samples",
                  "targets_tried", "targets_succeeded",
                  "delta_targets_tried", "delta_targets_succeeded"]
    rows = []
    prev_tried = prev_succ = 0
    for upto in rounds:
        tries, success, all_pmids = aggregate(
            range(1, upto + 1), target_meta, alias_to_target,
        )
        # papers_with_samples for the *cumulative* set
        ws_pmids: set[str] = set()
        for r in range(1, upto + 1):
            for paper in load_round_step5_output(r):
                if paper.get("samples"):
                    ws_pmids.add(paper["pmid"])
        nt = len(tries)
        ns = len(success)
        rows.append({
            "round": upto,
            "cumulative_papers": len(all_pmids),
            "papers_with_samples": len(ws_pmids),
            "targets_tried": nt,
            "targets_succeeded": ns,
            "delta_targets_tried": nt - prev_tried,
            "delta_targets_succeeded": ns - prev_succ,
        })
        prev_tried, prev_succ = nt, ns
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


# ─── CLI ────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(prog="extraction_metrics")
    p.add_argument("--round", type=int, default=None,
                   help="Round to write extraction_success_round_NN.csv for "
                        "(default: latest round detected)")
    p.add_argument("--cumulative-only", action="store_true",
                   help="Skip per-round CSV; only write cumulative + coverage")
    args = p.parse_args()

    ensure_phase7_dir()
    rounds = discover_rounds()
    if not rounds:
        print("No rounds found.", file=sys.stderr)
        sys.exit(1)

    print(f"Loading target catalog from {ENV6_PATH.name} ...", flush=True)
    target_meta, alias_to_target = load_target_catalog()
    print(f"  {len(target_meta)} unique targets, "
          f"{len(alias_to_target)} alias entries", flush=True)

    # Per-round CSV
    if not args.cumulative_only:
        target_round = args.round if args.round is not None else max(rounds)
        if target_round not in rounds:
            print(f"Round {target_round} not found. Available: {rounds}",
                  file=sys.stderr)
            sys.exit(1)
        print(f"\nAggregating round {target_round} only ...", flush=True)
        t_r, s_r, p_r = aggregate([target_round], target_meta, alias_to_target)
        out = PHASE7_DIR / f"extraction_success_round_{target_round:02d}.csv"
        write_success_csv(out, t_r, s_r, target_meta)
        print(f"  papers={len(p_r)}, targets tried={len(t_r)}, "
              f"succeeded={len(s_r)} → {out}")

    # Cumulative CSV
    print(f"\nAggregating cumulative across rounds {rounds} ...", flush=True)
    t_c, s_c, p_c = aggregate(rounds, target_meta, alias_to_target)
    out_c = PHASE7_DIR / "extraction_success_cumulative.csv"
    write_success_csv(out_c, t_c, s_c, target_meta)
    print(f"  papers={len(p_c)}, targets tried={len(t_c)}, "
          f"succeeded={len(s_c)} → {out_c}")

    # Coverage per round
    cov = PHASE7_DIR / "coverage_per_round.csv"
    write_coverage_per_round(cov, target_meta, alias_to_target)
    print(f"\nCoverage per round → {cov}")


if __name__ == "__main__":
    main()
