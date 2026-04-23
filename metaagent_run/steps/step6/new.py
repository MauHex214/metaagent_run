"""CLI 入口：python -m metaagent_run.steps.step6.new"""

import argparse
import logging

from .config import load_runtime_config
from .orchestrator import run


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Step6: cross-paper conflict resolution",
    )
    p.add_argument(
        "--input-dir", required=True,
        help="目录路径，应包含 step5_output.json 和上游产物",
    )
    p.add_argument(
        "--output-dir", default=None,
        help="输出目录（默认与 --input-dir 相同）",
    )
    p.add_argument("--log-level", default="INFO")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    cfg = load_runtime_config(
        input_dir=args.input_dir,
        output_dir=args.output_dir or args.input_dir,
    )
    run(cfg)


if __name__ == "__main__":
    main()
