"""审计：展开后的 new phase3 输出里，44 条 manual_shortcode_fix 是否还需要。

流程：
    1. 读旧 env3_final_annotations.csv（shortcode_fix 已应用的最终版本）和
       env3_final_annotations.before_shortcode_fix.csv（fix 前的 phase3 LLM 原判）。
    2. 用两者 diff 还原出 44 条 fix 的"目标判定"：{(raw_key, raw_key_after_fix): (fix_subtype, fix_qk, fix_bag)}
    3. 读新 env3_structured_annotations.csv（展开后 phase3 重跑的结果）。
    4. 对每条 fix 的原 raw_key（这里 shortcode_fix 作用在 raw_key 级别，所以 raw_key_original 对得上）:
       - 查新 phase3 的 subtype/qk/bag
       - 与 fix 目标比对
       - 一致 → 可退役；不一致 → 还需应用
    5. 输出退役/仍需表。
"""
from pathlib import Path
import pandas as pd

from metaagent_run.steps.env_field_pipeline import config

OLD_FIX_BEFORE = config.OUTPUT_DIR / "env3_final_annotations.before_shortcode_fix.csv"
OLD_FIX_AFTER = config.OUTPUT_DIR / "env3_final_annotations.csv"    # 有 raw_key_original? 可能没
NEW_PHASE3 = config.OUTPUT_DIR / "env3_structured_annotations.csv"
RETIRED = config.OUTPUT_DIR / "shortcode_fix_retired.csv"
STILL_NEEDED = config.OUTPUT_DIR / "shortcode_fix_still_needed.csv"


def run() -> None:
    before = pd.read_csv(OLD_FIX_BEFORE)
    after = pd.read_csv(OLD_FIX_AFTER)
    new = pd.read_csv(NEW_PHASE3)

    # 旧版 fix 以 raw_key 为 join 键（before/after 两份都是基于原始 raw_key）
    join_key = "raw_key"
    b = before[[join_key, "family", "subtype", "quantity_kind", "modifier_bag"]].rename(
        columns={"family": "bfam", "subtype": "bsub", "quantity_kind": "bqk", "modifier_bag": "bbag"}
    )
    a = after[[join_key, "family", "subtype", "quantity_kind", "modifier_bag"]].rename(
        columns={"family": "afam", "subtype": "asub", "quantity_kind": "aqk", "modifier_bag": "abag"}
    )
    ba = b.merge(a, on=join_key, how="inner")
    # "被 fix" 的条目：subtype 或 qk 或 bag 有变
    fixed = ba[
        (ba["bsub"] != ba["asub"])
        | (ba["bqk"] != ba["aqk"])
        | (ba["bbag"].fillna("") != ba["abag"].fillna(""))
    ].copy()
    print(f"Detected {len(fixed)} shortcode_fix rules (from old before/after diff)")

    # 新 phase3 annotation：用 raw_key_original 作对应键（因为 shortcode_fix 是对原始 raw_key 说的）
    if "raw_key_original" in new.columns:
        new_key = new[["raw_key_original", "family", "subtype", "quantity_kind", "modifier_bag"]].rename(
            columns={"raw_key_original": "raw_key",
                     "family": "nfam", "subtype": "nsub", "quantity_kind": "nqk", "modifier_bag": "nbag"}
        )
    else:
        new_key = new[["raw_key", "family", "subtype", "quantity_kind", "modifier_bag"]].rename(
            columns={"family": "nfam", "subtype": "nsub", "quantity_kind": "nqk", "modifier_bag": "nbag"}
        )

    joined = fixed.merge(new_key, on=join_key, how="left")

    # 判断：new phase3 的判定是否已经符合 fix 目标
    joined["new_matches_fix"] = (
        (joined["asub"] == joined["nsub"])
        & (joined["aqk"] == joined["nqk"])
        & (joined["abag"].fillna("") == joined["nbag"].fillna(""))
    )
    retired = joined[joined["new_matches_fix"]].copy()
    still = joined[~joined["new_matches_fix"] & joined["nsub"].notna()].copy()

    retired[["raw_key", "asub", "aqk", "abag", "nsub", "nqk", "nbag"]].to_csv(
        RETIRED, index=False
    )
    still[["raw_key", "bsub", "bqk", "bbag", "asub", "aqk", "abag",
           "nsub", "nqk", "nbag"]].to_csv(STILL_NEEDED, index=False)

    print("=" * 60)
    print("SHORTCODE FIX RESIDUE AUDIT")
    print("=" * 60)
    print(f"Total fix rules:     {len(fixed)}")
    print(f"  Retired (new phase3 already matches fix target): {len(retired)}")
    print(f"  Still needed (new phase3 still deviates):        {len(still)}")
    print(f"\nReports:")
    print(f"  {RETIRED}")
    print(f"  {STILL_NEEDED}")
    print()
    print("Retired (top 20):")
    print(retired[["raw_key", "asub", "aqk"]].head(20).to_string(index=False))
    print()
    print("Still needed (all):")
    print(still[["raw_key", "asub", "aqk", "nsub", "nqk"]].to_string(index=False))


if __name__ == "__main__":
    run()
