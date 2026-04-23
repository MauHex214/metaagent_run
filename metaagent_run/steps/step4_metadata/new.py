import argparse
import asyncio
import sys

from .config import load_runtime_config


def parse_args(config) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LLM 映射样本元数据字段到 MIxS 标准 + Tier 分级")
    parser.add_argument("--fields", default=str(config.fields_file))
    parser.add_argument("--mixs", default=str(config.mixs_file))
    parser.add_argument("--pmid-index", dest="pmid_index_file", default=str(config.pmid_index_file))
    parser.add_argument("--pmid-env-index", dest="pmid_env_index_file", default=str(config.pmid_env_index_file))
    parser.add_argument("--paper-env-map", dest="paper_env_map_file", default=str(config.paper_env_map_file))
    parser.add_argument("--discovery", default=str(config.discovery_file))
    parser.add_argument("--output-dir", default=str(config.output_dir))
    parser.add_argument("--batch-size", type=int, default=config.batch_size)
    parser.add_argument("--max-retries", type=int, default=config.max_retries_per_batch)
    parser.add_argument("--request-interval", type=float, default=config.request_interval)
    parser.add_argument("--no-resume", action="store_true")
    # Unified review decisions file replaces legacy exclusion/veto/correction lists.
    # --exclusion-list / --mapping-veto-list kept as deprecated aliases for the
    # same file so that external wrappers don't break.
    parser.add_argument("--review-decisions",
                        default=str(config.mapping_review_decisions_file))
    parser.add_argument("--exclusion-list",
                        default=str(config.mapping_review_decisions_file),
                        help="[deprecated] alias of --review-decisions")
    parser.add_argument("--mapping-veto-list",
                        default=str(config.mapping_review_decisions_file),
                        help="[deprecated] alias of --review-decisions")
    parser.add_argument("--tier2-min-pmid", type=int, default=config.tier2_min_pmid)
    parser.add_argument("--top-n-freq", type=int, default=config.top_n_freq)
    parser.add_argument("--top-n-fanout", type=int, default=config.top_n_fanout)
    parser.add_argument("--min-fanout", type=int, default=config.min_fanout)
    return parser.parse_args()


def main() -> None:
    config = load_runtime_config()
    args = parse_args(config)
    from .orchestrator import main_async

    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if callable(reconfigure):
        reconfigure(line_buffering=True)

    asyncio.run(
        main_async(
            fields_file=args.fields,
            mixs_file=args.mixs,
            pmid_index_file=args.pmid_index_file,
            pmid_env_index_file=args.pmid_env_index_file,
            paper_env_map_file=args.paper_env_map_file,
            discovery_file=args.discovery,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            max_retries=args.max_retries,
            request_interval=args.request_interval,
            no_resume=args.no_resume,
            exclusion_list=args.exclusion_list,
            mapping_veto_list=args.mapping_veto_list,
            tier2_min_pmid=args.tier2_min_pmid,
            top_n_freq=args.top_n_freq,
            top_n_fanout=args.top_n_fanout,
            min_fanout=args.min_fanout,
            runtime_config=config,
        )
    )


if __name__ == "__main__":
    main()
