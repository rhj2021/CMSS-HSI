"""Run the published weight-reliability and spectral-representation controls."""

from __future__ import annotations

import argparse
import importlib.metadata
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage", choices=["all", "weights", "features"], default="all"
    )
    parser.add_argument("--main-output", type=Path, default=Path("results/benchmark"))
    parser.add_argument("--output", type=Path, default=Path("results/controls"))
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--device")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--datasets", nargs="+")
    parser.add_argument("--detectors", nargs="+", choices=["GETNET", "GLAFormer"])
    parser.add_argument("--seeds", nargs="+", type=int)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")

    import torch
    from threadpoolctl import threadpool_limits
    from experiments.controls.source import Source
    from experiments.controls.experiment import Experiment
    from experiments.controls.reports import summarize
    from experiments.controls.util import (
        require,
        read_json,
        write_json,
        sha_file,
        beneath,
    )

    source_root = args.main_output.expanduser().resolve()
    output = args.output.expanduser().resolve()
    require(
        not beneath(output, source_root) and not beneath(source_root, output),
        "Benchmark and control outputs must be disjoint directories",
    )
    source = Source(source_root, args.data_root, args.device, args.threads)
    args.datasets = args.datasets or list(source.cfg["project"]["datasets"])
    args.detectors = args.detectors or list(source.cfg["benchmark"]["detectors"])
    args.seeds = args.seeds or list(source.cfg["project"]["seeds"])
    args.transforms = [item["name"] for item in source.cfg["transforms"]]
    for values, configured, name in (
        (args.datasets, source.cfg["project"]["datasets"], "datasets"),
        (args.detectors, source.cfg["benchmark"]["detectors"], "detectors"),
        (args.seeds, source.cfg["project"]["seeds"], "seeds"),
    ):
        require(
            len(values) == len(set(values)) and set(values) <= set(configured),
            f"--{name} must be a unique subset of the benchmark configuration",
        )
    package_root = Path(__file__).resolve().parent / "experiments/controls"
    signature = dict(
        benchmark_protocol=source.protocol,
        benchmark_files={
            name: sha_file(source_root / name)
            for name in (
                "config.yaml",
                "manifest.json",
                "metrics/conditions.csv",
                "metrics/geometry.csv",
            )
        },
        code={path.name: sha_file(path) for path in sorted(package_root.glob("*.py"))},
        datasets=args.datasets,
        detectors=args.detectors,
        seeds=args.seeds,
        transforms=args.transforms,
        fractions=[0.25, 0.5],
        intervention_seeds=[1001, 1002, 1003],
        block_size=24,
        device=source.device,
        threads=args.threads,
        versions={
            name: importlib.metadata.version(name)
            for name in ("numpy", "scipy", "scikit-learn", "torch", "pandas")
        },
    )
    marker = output / "manifest.json"
    if marker.exists():
        manifest = read_json(marker)
        require(
            manifest["signature"] == signature,
            "Control output belongs to different inputs, code or settings; choose a new --output",
        )
    else:
        require(
            not output.exists() or not any(output.iterdir()),
            "Choose an empty control output directory",
        )
        output.mkdir(parents=True, exist_ok=True)
        manifest = dict(signature=signature, completed_stages=[])
        write_json(marker, manifest)
    torch.set_num_threads(args.threads)
    with threadpool_limits(limits=args.threads):
        source.initial_preflight(
            args.datasets, args.transforms, args.seeds, args.detectors
        )
        Experiment(source, args, output).execute(args.stage)
        done = set(manifest["completed_stages"])
        done.update(["weights", "features"] if args.stage == "all" else [args.stage])
        tables = summarize(output, args, done)
        manifest["completed_stages"] = sorted(done)
        write_json(marker, manifest)
    print(f"Control CSV tables: {tables}")


if __name__ == "__main__":
    main()
