import argparse
import asyncio
import sys
from typing import Optional

from .config import load_runtime_config

_REVIEW_USE_CONFIG_DEFAULT = "__USE_CONFIG_DEFAULT__"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run step3 schema discovery pipeline")
    parser.add_argument("--input", dest="input_file", default=None)
    parser.add_argument("--output", dest="output_file", default=None)
    parser.add_argument("--checkpoint", dest="checkpoint_file", default=None)
    parser.add_argument(
        "--review",
        dest="review_file",
        nargs="?",
        const=_REVIEW_USE_CONFIG_DEFAULT,
        default=None,
        help=(
            "Apply canonical review post-processing after the main pipeline "
            "writes its output. With no value, uses the path in "
            "config.canonical_review_decisions_file; with a value, overrides. "
            "Omitted (default) → no review step."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_runtime_config()
    from .orchestrator import main_async

    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if callable(reconfigure):
        reconfigure(line_buffering=True)

    review_file: Optional[str]
    if args.review_file is None:
        review_file = None
    elif args.review_file == _REVIEW_USE_CONFIG_DEFAULT:
        review_file = str(config.canonical_review_decisions_file)
    else:
        review_file = args.review_file

    asyncio.run(
        main_async(
            input_file=args.input_file or str(config.input_file),
            output_file=args.output_file or str(config.output_file),
            checkpoint_file=args.checkpoint_file or str(config.checkpoint_file),
            runtime_config=config,
            review_file=review_file,
        )
    )


if __name__ == "__main__":
    main()
