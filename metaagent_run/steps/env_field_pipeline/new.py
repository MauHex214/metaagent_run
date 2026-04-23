"""env_field_pipeline 统一入口。

用法：
    python3 -m metaagent_run.steps.env_field_pipeline.new <phase>

phase 取值：0 / 1 / 2 / 3 / 3-norm-propose / 3-norm-apply /
           4 / 4-rerun-mixs / 4-rename-only / 5 / 5-calibrate /
           6 / manual-patch
"""
import argparse
import logging
import sys


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


def main() -> None:
    _setup_logging()
    parser = argparse.ArgumentParser(description="env_field_pipeline 统一入口")
    parser.add_argument(
        "phase",
        choices=["0", "1", "2", "3", "3-norm-propose", "3-norm-apply",
                 "4", "4-rerun-mixs", "4-rename-only",
                 "5-calibrate", "5",
                 "6",
                 "manual-patch"],
        help="要执行的环节编号",
    )
    args = parser.parse_args()

    if args.phase == "0":
        from . import phase0
        phase0.run()
    elif args.phase == "1":
        from . import phase1
        phase1.run()
    elif args.phase == "2":
        from . import phase2
        phase2.run()
    elif args.phase == "3":
        from . import phase3
        phase3.run()
    elif args.phase == "3-norm-propose":
        from . import phase3_norm
        phase3_norm.propose()
    elif args.phase == "3-norm-apply":
        from . import phase3_norm
        phase3_norm.apply_merges()
    elif args.phase == "4":
        from . import phase4
        phase4.run()
    elif args.phase == "4-rerun-mixs":
        from . import phase4
        phase4.rerun_mixs()
    elif args.phase == "4-rename-only":
        from . import phase4
        phase4.rename_only()
    elif args.phase == "5-calibrate":
        from . import phase5
        phase5.calibrate()
    elif args.phase == "5":
        from . import phase5
        phase5.run()
    elif args.phase == "6":
        from . import phase6
        phase6.run()
    elif args.phase == "manual-patch":
        from . import phase_manual_patch
        phase_manual_patch.run()
    else:
        print(f"未知 phase: {args.phase}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
