"""CLI 入口：python -m metaagent_run.steps.step5.new"""

import argparse
import asyncio
import sys

from .config import load_runtime_config


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Step5_test: paper-level accession-metadata extraction")

    # 主输入/输出（均可选，默认从 config 读取）
    p.add_argument("--input", dest="input_file", default=None,
                   help="target_env_v2_relation_input.json")
    p.add_argument("--output", dest="output_file", default=None,
                   help="输出文件路径")

    # 上游产物
    p.add_argument("--relation", dest="relation_file", default=None,
                   help="step2 relation output JSON")
    p.add_argument("--accession", dest="accession_file", default=None,
                   help="step3 accession output JSON")
    p.add_argument("--accession-list", dest="accession_list_file", default=None,
                   help="外部 DB 验证 accession list (TSV)")
    p.add_argument("--expanded-metadata", dest="expanded_metadata_file", default=None,
                   help="pmid_run_merged_data_expanded.json")
    p.add_argument("--env-tag", dest="env_tag_file", default=None,
                   help="step4 env_tag output JSON")
    p.add_argument("--env-targets", dest="env_extraction_targets_file", default=None,
                   help="env_extraction_targets.json")
    p.add_argument("--schema-discovery", dest="schema_discovery_file", default=None,
                   help="schema_discovery_result_review-gpt-mixs.json (synonym_groups)")

    # 运行参数
    p.add_argument("--paper-concurrency", dest="paper_concurrency", type=int, default=None)

    return p.parse_args()


def main() -> None:
    args = parse_args()

    overrides = {
        k: v for k, v in vars(args).items()
        if v is not None and k != "paper_concurrency"
    }
    config = load_runtime_config(**overrides)

    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if callable(reconfigure):
        reconfigure(line_buffering=True)

    from .orchestrator import main_async

    asyncio.run(
        main_async(
            input_file=config.input_file,
            output_file=config.output_file,
            paper_concurrency=args.paper_concurrency,
            runtime_config=config,
        )
    )


if __name__ == "__main__":
    main()
