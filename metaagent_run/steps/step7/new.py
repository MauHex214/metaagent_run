"""CLI 入口：python -m metaagent_run.steps.step7.new

子命令：
  build-cde   构建 CDE（自动 Tier 1 + LLM Tier 2）；测通阶段默认 auto-merge
  run         主流程：load step6 → normalize → hoist → write
  merge-only  生产用：合并已有 tier1 + tier2_reviewed
"""

import argparse
import logging
from pathlib import Path

from .config import load_runtime_config


def _add_common_io(p: argparse.ArgumentParser) -> None:
    p.add_argument("--input-dir", required=True)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--log-level", default="INFO")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Step7: Key/Value normalization + Sample-level hoist",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_build = sub.add_parser("build-cde", help="Build CDE (Tier 1 autogen + Tier 2 LLM)")
    _add_common_io(p_build)
    p_build.add_argument(
        "--no-merge", action="store_true",
        help="Stop after writing tier1_autogen + tier2_suggestions; do NOT auto-merge",
    )

    p_run = sub.add_parser("run", help="Run main pipeline (load step6 → normalize → hoist)")
    _add_common_io(p_run)

    p_merge = sub.add_parser("merge-only",
                             help="Merge existing Tier 1 autogen + Tier 2 reviewed CDE files")
    _add_common_io(p_merge)
    p_merge.add_argument("--tier1", required=True, help="Path to cde_tier1_autogen.json")
    p_merge.add_argument("--tier2", required=True, help="Path to tier2_cde_reviewed.json")

    return parser.parse_args()


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

    if args.cmd == "build-cde":
        from .cde_builder import build_cde
        build_cde(cfg, auto_merge=not args.no_merge)
    elif args.cmd == "run":
        from .orchestrator import run
        run(cfg)
    elif args.cmd == "merge-only":
        from .cde_builder import merge_only
        merge_only(cfg, Path(args.tier1), Path(args.tier2))
    else:
        raise SystemExit("Unknown subcommand: %s" % args.cmd)


if __name__ == "__main__":
    main()
