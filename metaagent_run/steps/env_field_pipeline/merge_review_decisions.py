"""基于路线 α 自动合并旧 review_queue 的 decision 到新 review_queue。

规则：
    - exact_match     → 继承旧 decision（按 frozenset(members) 精确匹配）
    - partial_overlap → 继承第一个有重合的旧提议的 decision（通常都是 approve）
    - new_only        → REJECT_LIST 里命中的 reject，其余 approve
    - 用户明确建议 reject 的 new_only（7 条）通过 (subtype, target_name) 硬编码匹配
"""
from pathlib import Path
import pandas as pd

from metaagent_run.steps.env_field_pipeline import config

OLD_REVIEW = config.OUTPUT_DIR / "env3_norm_review_queue.pre_expansion.csv"
NEW_REVIEW = config.OUTPUT_DIR / "env3_norm_review_queue.csv"

# 用户明确 reject 的 new_only 提议，按 (subtype, target_name_new) 识别
REJECT_NEW_ONLY: set[tuple[str, str]] = {
    ("time_duration", "annual"),
    ("time_point", "sunset_time"),
    ("vertical_position", "oxycline_depth"),
    ("material_medium_type", "ice_type"),
    ("sampling_site", "location"),
    ("trace_chemistry", "metals"),
    ("material_medium_type", "sample_type"),
}


def _members_set(s):
    if not isinstance(s, str):
        return frozenset()
    return frozenset(m.strip() for m in s.split(";") if m.strip())


def run() -> None:
    old = pd.read_csv(OLD_REVIEW)
    new = pd.read_csv(NEW_REVIEW)

    old_records = [
        (_members_set(r.members), str(getattr(r, "decision", "pending") or "pending").strip().lower(), r)
        for r in old.itertuples(index=False)
    ]

    stats = {"exact": 0, "partial_approve": 0, "partial_reject": 0,
             "new_approve": 0, "new_reject": 0}

    decisions = []
    for nr in new.itertuples(index=False):
        ms_new = _members_set(nr.members)
        # exact
        exact_hit = next((rec for rec in old_records if rec[0] == ms_new), None)
        if exact_hit is not None:
            decisions.append(exact_hit[1] if exact_hit[1] else "approve")
            stats["exact"] += 1
            continue
        # partial
        partial_hits = [rec for rec in old_records if rec[0] & ms_new]
        if partial_hits:
            # 若任一旧 decision = approve，继承 approve；否则 reject
            any_approve = any(
                h[1] in {"approve", "approved", "yes", "y", "true", "t", "1", "custom"}
                for h in partial_hits
            )
            if any_approve:
                decisions.append("approve")
                stats["partial_approve"] += 1
            else:
                decisions.append("reject")
                stats["partial_reject"] += 1
            continue
        # new_only
        key = (str(nr.subtype).strip(), str(nr.target_name).strip())
        if key in REJECT_NEW_ONLY:
            decisions.append("reject")
            stats["new_reject"] += 1
        else:
            decisions.append("approve")
            stats["new_approve"] += 1

    new["decision"] = decisions
    new.to_csv(NEW_REVIEW, index=False)

    print("=" * 60)
    print("DECISION MERGE SUMMARY")
    print("=" * 60)
    for k, v in stats.items():
        print(f"  {k:20s}: {v}")
    total_approve = stats["exact"] + stats["partial_approve"] + stats["new_approve"]
    total_reject = stats["partial_reject"] + stats["new_reject"]
    print(f"  TOTAL approve     : {total_approve}")
    print(f"  TOTAL reject      : {total_reject}")
    print(f"  Grand total       : {len(new)}")
    print(f"\nWrote decisions → {NEW_REVIEW}")


if __name__ == "__main__":
    run()
