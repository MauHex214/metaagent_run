import argparse
import asyncio
import sys

from .config import load_runtime_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run step3 schema discovery pipeline")
    parser.add_argument("--input", dest="input_file", default=None)
    parser.add_argument("--output", dest="output_file", default=None)
    parser.add_argument("--checkpoint", dest="checkpoint_file", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_runtime_config()
    from .orchestrator import main_async

    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if callable(reconfigure):
        reconfigure(line_buffering=True)

    asyncio.run(
        main_async(
            input_file=args.input_file or str(config.input_file),
            output_file=args.output_file or str(config.output_file),
            checkpoint_file=args.checkpoint_file or str(config.checkpoint_file),
            runtime_config=config,
        )
    )


if __name__ == "__main__":
    main()
