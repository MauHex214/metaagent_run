import argparse
import asyncio
import sys

from .config import RuntimeConfig, load_runtime_config


def parse_args(config: RuntimeConfig) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run step2 pipeline (relation extraction + discovery builder)"
    )
    _ = parser.add_argument(
        "--input", dest="full_text_file", default=str(config.full_text_json)
    )
    _ = parser.add_argument(
        "--relation", dest="relation_file", default=str(config.relation_output_file)
    )
    _ = parser.add_argument(
        "--pmid-year", dest="pmid_year_file", default=str(config.pmid_year_txt)
    )
    _ = parser.add_argument(
        "--output", dest="discovery_output_file", default=str(config.discovery_out)
    )
    _ = parser.add_argument(
        "--relation-max-concurrency",
        dest="relation_max_concurrency",
        type=int,
        default=None,
    )
    _ = parser.add_argument(
        "--skip-relation",
        dest="skip_relation",
        action="store_true",
        help="Skip relation extraction and use --relation file directly.",
    )
    return parser.parse_args()


def main() -> None:
    config = load_runtime_config()
    args = parse_args(config)
    from .orchestrator import main_async

    run_relation = not args.skip_relation

    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if callable(reconfigure):
        reconfigure(line_buffering=True)

    asyncio.run(
        main_async(
            full_text_file=args.full_text_file,
            relation_file=args.relation_file,
            pmid_year_file=args.pmid_year_file,
            discovery_output_file=args.discovery_output_file,
            run_relation=run_relation,
            relation_max_concurrency=args.relation_max_concurrency,
            runtime_config=config,
        )
    )


if __name__ == "__main__":
    main()
