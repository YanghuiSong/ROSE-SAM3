"""Evaluate the fixed ROSE-SAM3 inference pipeline with MMSegmentation."""
from __future__ import annotations

import argparse
import os
import os.path as osp
import sys

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))

from mmengine.config import Config, DictAction
from mmengine.runner import Runner

import custom_datasets  # noqa: F401
import rose_sam3_segmentor  # noqa: F401


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate the ROSE-SAM3 protocol")
    parser.add_argument("config", help="Path to the evaluation config")
    parser.add_argument("--out", type=str, help="Directory for offline predictions")
    parser.add_argument(
        "--cfg-options",
        nargs="+",
        action=DictAction,
        help="Override evaluation config values",
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    cfg.launcher = "none"
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    if args.out is not None:
        cfg.test_evaluator["output_dir"] = args.out
    cfg.work_dir = os.environ.get(
        "ROSE_WORK_DIR",
        osp.join("./work_dirs", osp.splitext(osp.basename(args.config))[0]),
    )
    print(osp.abspath(osp.curdir))
    print(f"Model type: {cfg.model.type}")
    runner = Runner.from_cfg(cfg)
    results = runner.test()
    if runner.rank == 0:
        print(results)


if __name__ == "__main__":
    main()
