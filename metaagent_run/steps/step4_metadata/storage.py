import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

LOGGER = logging.getLogger(__name__)


def save_checkpoint(
    path: str,
    mapped_results: List[Dict[str, Any]],
    failed_fields: List[str],
    processed_count: int,
    total_count: int,
) -> None:
    data = {
        "mapped_results": mapped_results,
        "failed_fields": failed_fields,
        "processed_count": processed_count,
        "total_count": total_count,
    }
    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
    LOGGER.info("[Checkpoint] processed=%d/%d", processed_count, total_count)


def load_checkpoint(path: str) -> Optional[Dict[str, Any]]:
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as file:
        data = json.load(file)
    LOGGER.info(
        "[Checkpoint] 恢复：processed=%d/%d，failed=%d",
        data.get("processed_count", 0),
        data.get("total_count", 0),
        len(data.get("failed_fields", [])),
    )
    return data


def save_outputs(
    mapped_results: List[Dict[str, Any]],
    failed_fields: List[str],
    freq_data: Dict[str, Dict[str, Any]],
    mixs_standards: List[Dict[str, str]],
    output_dir: Path,
    mapping_filename: str = "step4b_mapping_result.json",
    unmapped_filename: str = "step4b_unmapped_fields.json",
    frequency_filename: str = "step4b_frequency_by_mixs_term.json",
) -> None:
    truly_mapped = [record for record in mapped_results if record["mixs_slot"] != "UNMAPPED"]
    final_unmapped = [record for record in mapped_results if record["mixs_slot"] == "UNMAPPED"]
    veto_unmapped = [record for record in final_unmapped if record.get("veto_forced_unmapped")]
    llm_unmapped = [record for record in final_unmapped if not record.get("veto_forced_unmapped")]
    unique_slots = {record["mixs_slot"] for record in truly_mapped}

    mapping_file = output_dir / mapping_filename
    with mapping_file.open("w", encoding="utf-8") as file:
        json.dump(
            {
                "summary": {
                    "total_input_fields": len(mapped_results) + len(failed_fields),
                    "mapped_to_mixs": len(truly_mapped),
                    "unmapped_by_llm": len(llm_unmapped),
                    "forced_unmapped_by_veto": len(veto_unmapped),
                    "unmapped_final": len(final_unmapped),
                    "failed_to_process": len(failed_fields),
                    "unique_mixs_terms_covered": len(unique_slots),
                    "total_mixs_terms_available": len(mixs_standards),
                    "mixs_coverage": "{:.1f}%".format(
                        len(unique_slots) / len(mixs_standards) * 100 if mixs_standards else 0
                    ),
                },
                "mapped": truly_mapped,
                "unmapped": final_unmapped,
                "failed": failed_fields,
            },
            file,
            ensure_ascii=False,
            indent=2,
        )
    LOGGER.info("映射表: %s", mapping_file)

    unmapped_file = output_dir / unmapped_filename
    with unmapped_file.open("w", encoding="utf-8") as file:
        json.dump(
            {
                "unmapped_by_llm": [
                    {"field": record["field"], "reason": record.get("reason", "")}
                    for record in llm_unmapped
                ],
                "forced_unmapped_by_veto": [
                    {
                        "field": record["field"],
                        "reason": record.get("reason", ""),
                        "original_mixs_slot": record.get("original_mixs_slot", ""),
                        "original_mixs_title": record.get("original_mixs_title", ""),
                    }
                    for record in veto_unmapped
                ],
                "failed_to_process": failed_fields,
            },
            file,
            ensure_ascii=False,
            indent=2,
        )
    LOGGER.info("未映射字段: %s", unmapped_file)

    freq_file = output_dir / frequency_filename
    with freq_file.open("w", encoding="utf-8") as file:
        json.dump(freq_data, file, ensure_ascii=False, indent=2)
    LOGGER.info("频次数据: %s", freq_file)
