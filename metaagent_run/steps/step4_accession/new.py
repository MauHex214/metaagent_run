import argparse
import asyncio
import sys

from .config import load_runtime_config


def parse_args(config) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run metaagent step4 pipeline")
    parser.add_argument("--input", dest="input_file", default=config.input_file)
    parser.add_argument("--output", dest="output_file", default=config.output_file)
    parser.add_argument(
        "--max-concurrency",
        dest="max_concurrency",
        type=int,
        default=None,
    )
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
            input_file=args.input_file,
            output_file=args.output_file,
            max_concurrency=args.max_concurrency,
            runtime_config=config,
        )
    )


if __name__ == "__main__":
    main()
