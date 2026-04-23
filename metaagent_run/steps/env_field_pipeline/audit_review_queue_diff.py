"""比对新旧 review_queue 的 LLM 提议差异（run 在 phase3a-norm propose 之后）。

比对维度：
    - 新提议 members (frozenset) 完全命中旧（含 decision） → exact_match
    - 部分重合（共享 ≥1 成员）                          → partial_overlap
    - 全新提议（members 不与任何旧提议重合）             → new_only
    - 旧 review 里有但新提议里没出现                     → missing_in_new

阈值：如果 new_only + partial_overlap > 旧 approve 数的 20%，
      自动暂停（退出码 2），要求人工审核。
"""
from pathlib import Path
import pandas as pd
import sys

from metaagent_run.steps.env_field_pipeline import config

OLD_REVIEW = config.OUTPUT_DIR / "env3_norm_review_queue.pre_expansion.csv"
NEW_REVIEW = config.OUTPUT_DIR / "env3_norm_review_queue.csv"
DIFF_REPORT = config.OUTPUT_DIR / "env3_norm_review_diff.csv"


def _members_set(s: str) -> frozenset[str]:
    if not isinstance(s, str):
        return frozenset()
    return frozenset(m.strip() for m in s.split(";") if m.strip())


def _load(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise RuntimeError(f"missing: {path}")
    return pd.read_csv(path)


def run() -> int:
    old = _load(OLD_REVIEW)
    new = _load(NEW_REVIEW)

    old_ms = [(_members_set(r.members), r) for r in old.itertuples(index=False)]
    new_ms = [(_members_set(r.members), r) for r in new.itertuples(index=False)]

    exact = []
    partial = []
    new_only = []

    for ms_new, r_new in new_ms:
        hit_exact = None
        hit_partial = []
        for ms_old, r_old in old_ms:
            if ms_new == ms_old:
                hit_exact = r_old
                break
            if ms_new & ms_old:
                hit_partial.append(r_old)
        if hit_exact is not None:
            exact.append((r_new, hit_exact))
        elif hit_partial:
            partial.append((r_new, hit_partial))
        else:
            new_only.append(r_new)

    # 旧里有但新里没出现（用 new 的 ms 集）
    new_ms_set = [ms for ms, _ in new_ms]
    missing = [r for ms, r in old_ms if not any(ms & ms_n for ms_n in new_ms_set)]

    approve_count = int((old.get("decision", "").astype(str).str.lower().isin(
        {"approve", "approved", "yes", "y", "true", "t", "1", "custom"}
    )).sum()) if "decision" in old.columns else 0

    n_new_plus_partial = len(new_only) + len(partial)
    threshold = max(5, int(0.20 * max(1, approve_count)))

    print("=" * 60)
    print("REVIEW QUEUE DIFF AUDIT")
    print("=" * 60)
    print(f"Old review queue: {len(old)} proposals (approved={approve_count})")
    print(f"New review queue: {len(new)} proposals")
    print(f"  exact match:     {len(exact)}")
    print(f"  partial overlap: {len(partial)}")
    print(f"  new only:        {len(new_only)}")
    print(f"  missing in new:  {len(missing)}")
    print()
    print(f"Threshold (20% of old approved = {threshold}):")
    print(f"  new + partial = {n_new_plus_partial}")

    # 写差异报告
    rows = []
    for r_new, r_old in exact:
        rows.append({
            "status": "exact_match",
            "family": getattr(r_new, "family", ""),
            "subtype": getattr(r_new, "subtype", ""),
            "target_name_new": getattr(r_new, "target_name", ""),
            "members_new": getattr(r_new, "members", ""),
            "target_name_old": getattr(r_old, "target_name", ""),
            "decision_old": getattr(r_old, "decision", ""),
        })
    for r_new, olds in partial:
        rows.append({
            "status": "partial_overlap",
            "family": getattr(r_new, "family", ""),
            "subtype": getattr(r_new, "subtype", ""),
            "target_name_new": getattr(r_new, "target_name", ""),
            "members_new": getattr(r_new, "members", ""),
            "target_name_old": "|".join(getattr(o, "target_name", "") for o in olds),
            "decision_old": "|".join(str(getattr(o, "decision", "")) for o in olds),
        })
    for r_new in new_only:
        rows.append({
            "status": "new_only",
            "family": getattr(r_new, "family", ""),
            "subtype": getattr(r_new, "subtype", ""),
            "target_name_new": getattr(r_new, "target_name", ""),
            "members_new": getattr(r_new, "members", ""),
            "target_name_old": "",
            "decision_old": "",
        })
    for r_old in missing:
        rows.append({
            "status": "missing_in_new",
            "family": getattr(r_old, "family", ""),
            "subtype": getattr(r_old, "subtype", ""),
            "target_name_new": "",
            "members_new": "",
            "target_name_old": getattr(r_old, "target_name", ""),
            "decision_old": getattr(r_old, "decision", ""),
        })
    pd.DataFrame(rows).to_csv(DIFF_REPORT, index=False)
    print(f"\nDiff report → {DIFF_REPORT}")

    if n_new_plus_partial > threshold:
        print(f"\n⚠️ DIFF EXCEEDS THRESHOLD. PAUSED for manual review.")
        return 2
    print("\n✓ Diff within threshold. Safe to auto-apply with old decisions.")
    return 0


if __name__ == "__main__":
    sys.exit(run())
