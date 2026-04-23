"""Step6 — Cross-paper conflict resolution module.

输入: step5_output.json + step4b/step3/pmid_year 上游产物
输出: step6_resolved_output.json + step6_stats.json

主键: (raw_accession, canonical_slot)，按仲裁规则从多候选中选出权威值。
"""
