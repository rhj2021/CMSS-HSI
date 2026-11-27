#!/usr/bin/env python3
"""Run the CMSS benchmark and component studies."""

from __future__ import annotations

import argparse
from pathlib import Path


def main(argv=None):
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(root / "configs/benchmark.yaml"))
    parser.add_argument(
        "--stage",
        required=True,
        choices=[
            "inspect",
            "prepare",
            "clean",
            "baselines",
            "geometry",
            "k",
            "selector",
            "summarize",
            "all",
        ],
    )
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--detectors", nargs="+", choices=["GETNET", "GLAFormer"])
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument(
        "--progress", action=argparse.BooleanOptionalAction, default=None
    )
    parser.add_argument(
        "--geometry-only",
        action="store_true",
        help="Measure geometry without detector preparation or inference.",
    )
    args = parser.parse_args(argv)
    if args.threads < 1:
        parser.error("--threads must be positive")

    import torch
    from threadpoolctl import threadpool_limits
    from experiments.benchmark.engine import BenchmarkRunner
    from experiments.benchmark.reports import summarize

    torch.set_num_threads(args.threads)
    try:
        import cv2

        cv2.setNumThreads(args.threads)
    except ImportError:
        pass
    with threadpool_limits(limits=args.threads):
        with BenchmarkRunner(
            args.config,
            root,
            device=args.device,
            datasets=args.datasets,
            seeds=args.seeds,
            detectors=args.detectors,
            data_root=args.data_root,
            output=args.output,
            geometry_only=args.geometry_only,
            progress=args.progress,
        ) as runner:
            if args.stage == "inspect":
                runner.inspect()
            if args.stage in {"prepare", "all"} and not args.geometry_only:
                runner.prepare()
            suites = (
                ["clean", "baselines", "geometry", "k", "selector"]
                if args.stage == "all"
                else [args.stage]
            )
            for suite in suites:
                if suite in {"clean", "baselines", "geometry", "k", "selector"}:
                    runner.evaluate(suite)
            summary = summarize(runner.root, runner.cfg)
            print("[SUMMARY]", summary)


if __name__ == "__main__":
    main()
