"""Measure complete preprocessing and selector-off representation controls."""

from __future__ import annotations

import argparse
import importlib.metadata
import platform
from pathlib import Path
import time
import zlib


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage", choices=["all", "complete", "features"], default="all"
    )
    parser.add_argument("--main-output", type=Path, default=Path("results/benchmark"))
    parser.add_argument("--output", type=Path, default=Path("results/timing"))
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--device")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--datasets", nargs="+")
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")

    import cv2
    import numpy as np
    import pandas as pd
    import torch
    import yaml
    from threadpoolctl import threadpool_info, threadpool_limits
    from cmss.data import load_dataset
    from experiments.benchmark.core import CandidateStore, full_cmss
    from experiments.benchmark.registration import roma_model
    from experiments.benchmark.transforms import apply_extended
    from experiments.controls.core import Variant, estimate
    from experiments.controls.util import (
        require,
        read_json,
        write_json,
        write_csv,
        sha_file,
        sha_array,
        beneath,
    )

    main_output = args.main_output.expanduser().resolve()
    output = args.output.expanduser().resolve()
    require(
        not beneath(output, main_output) and not beneath(main_output, output),
        "Benchmark and timing outputs must be disjoint directories",
    )
    cfg = yaml.safe_load((main_output / "config.yaml").read_text(encoding="utf-8"))
    main_manifest = read_json(main_output / "manifest.json")
    data_root = (
        Path(args.data_root or cfg["project"]["data_root"]).expanduser().resolve()
    )
    datasets = args.datasets or list(cfg["project"]["datasets"])
    require(
        len(datasets) == len(set(datasets))
        and set(datasets) <= set(cfg["project"]["datasets"]),
        "--datasets must be a unique subset of the benchmark datasets",
    )
    device = torch.device(args.device or main_manifest["environment"]["device"])
    require(device.type in {"cpu", "cuda"}, "Use a CPU or CUDA device")
    if device.type == "cuda":
        require(torch.cuda.is_available(), "CUDA is unavailable")
        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        torch.cuda.set_device(device)
    device_string = str(device)
    torch.set_num_threads(args.threads)
    cv2.setNumThreads(args.threads)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    runtime = cfg["benchmark"]["runtime"]
    warmup, repeats = int(runtime["warmup"]), int(runtime["repeats"])
    require(
        warmup >= 1 and repeats >= 2,
        "Timing requires a warm-up and repeated measurements",
    )
    require(
        runtime["transforms"] == ["row_shift_4"], "Timing uses native-sized row drift"
    )
    correction = cfg["correction"]
    methods = list(runtime["methods"])
    if "roma_affine" not in methods:
        methods.append("roma_affine")
    spec = next(item for item in cfg["transforms"] if item["name"] == "row_shift_4")
    cpu = platform.processor()
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        cpu = next(
            (
                line.split(":", 1)[1].strip()
                for line in cpuinfo.read_text().splitlines()
                if line.startswith("model name")
            ),
            cpu,
        )

    with threadpool_limits(limits=args.threads):
        environment = dict(
            python=platform.python_version(),
            cpu=cpu,
            device=device_string,
            gpu=torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            cuda=torch.version.cuda,
            torch_threads=torch.get_num_threads(),
            opencv_threads=cv2.getNumThreads(),
            threadpools=threadpool_info(),
            versions={
                name: importlib.metadata.version(name)
                for name in ("numpy", "scipy", "scikit-learn", "torch", "pandas")
            },
        )
        code_root = Path(__file__).resolve().parent
        source_files = [Path(__file__).resolve()]
        source_files += sorted((code_root / "cmss").rglob("*.py"))
        source_files += sorted((code_root / "experiments/benchmark").glob("*.py"))
        source_files += [code_root / "experiments/controls/core.py"]
        signature = dict(
            benchmark_protocol=main_manifest["protocol_hash"],
            config_sha256=sha_file(main_output / "config.yaml"),
            code={
                str(path.relative_to(code_root)): sha_file(path)
                for path in source_files
            },
            datasets=datasets,
            transform=spec,
            warmup=warmup,
            repeats=repeats,
            methods=methods,
            environment=environment,
        )
        marker = output / "manifest.json"
        if marker.exists():
            require(
                read_json(marker)["signature"] == signature,
                "Timing output belongs to different inputs, code or settings; choose a new --output",
            )
        else:
            require(
                not output.exists() or not any(output.iterdir()),
                "Choose an empty timing output directory",
            )
            output.mkdir(parents=True, exist_ok=True)
            write_json(marker, dict(signature=signature))
        write_json(output / "environment.json", environment)
        records = {}
        scopes = ["complete", "features"] if args.stage == "all" else [args.stage]
        for scope in scopes:
            path = output / f"{scope}.csv"
            records[scope] = (
                pd.read_csv(path).to_dict("records") if path.exists() else []
            )
        if "complete" in scopes and "roma_affine" in methods:
            roma_model(device_string)

        def synchronize_device():
            if device.type == "cuda":
                torch.cuda.synchronize(device)

        for name in datasets:
            ds = load_dataset(
                name, data_root, cfg.get("dataset_options", {}).get(name, {})
            )
            inventory = read_json(main_output / "inventory" / f"{name}.json")
            hashes = {key: sha_array(getattr(ds, key)) for key in ("t1", "t2", "gt")}
            require(
                hashes == inventory["full_hashes"],
                f"Timing data differs from benchmark: {name}",
            )
            seed = int(
                zlib.crc32(f"{name}:{spec['name']}:{correction['seed']}".encode())
                & 0x7FFFFFFF
            )
            moving = apply_extended(ds.t2, spec, device_string, seed).moving
            for scope in scopes:
                names = methods if scope == "complete" else ["pca1", "pca3", "full"]
                for method in names:
                    variant = (
                        Variant(
                            method, "features", representation=method, selector=False
                        )
                        if scope == "features"
                        else None
                    )
                    done = {
                        int(row["repetition"])
                        for row in records[scope]
                        if row["dataset"] == name and row["method"] == method
                    }
                    if set(range(repeats)) <= done:
                        continue

                    def measure():
                        synchronize_device()
                        start = time.perf_counter()
                        if scope == "features":
                            result = estimate(
                                ds.t1,
                                moving,
                                correction,
                                device_string,
                                variant,
                                f"{name}:row_shift_4",
                                collect=False,
                            )
                        elif method == "cmss":
                            result = full_cmss(
                                ds.t1, moving, correction, device_string, False
                            )
                        else:
                            store = CandidateStore(
                                output / "unused",
                                ds.t1,
                                moving,
                                correction,
                                cfg["benchmark"]["external"],
                                device_string,
                                "timing",
                                False,
                                use_disk=False,
                            )
                            result = store.get(method)
                        synchronize_device()
                        seconds = time.perf_counter() - start
                        status = (
                            "ok"
                            if scope == "features"
                            else result.metadata.get("registration_status", "ok")
                        )
                        del result
                        return seconds, status

                    for _ in range(warmup):
                        measure()
                    for repeat in range(repeats):
                        if repeat in done:
                            continue
                        seconds, status = measure()
                        records[scope].append(
                            dict(
                                dataset=name,
                                transform=spec["name"],
                                method=method,
                                repetition=repeat,
                                seconds=seconds,
                                scope=scope,
                                selector_enabled=scope == "complete"
                                and method == "cmss",
                                height=ds.t1.shape[0],
                                width=ds.t1.shape[1],
                                bands=ds.t1.shape[2],
                                device=device_string,
                                threads=args.threads,
                                registration_status=status,
                            )
                        )
                        write_csv(output / f"{scope}.csv", records[scope])
                    print(f"Timed {scope}: {name}/{method}", flush=True)
            del ds, moving
        for scope in scopes:
            frame = pd.DataFrame(records[scope])
            require(
                not frame.duplicated(["dataset", "method", "repetition"]).any(),
                "Duplicate timing repetitions",
            )
            summary = (
                frame.groupby(["dataset", "method"])
                .seconds.agg(["mean", "std", "count"])
                .reset_index()
            )
            summary.columns = [
                "dataset",
                "method",
                "mean_seconds",
                "sample_sd_seconds",
                "repetitions",
            ]
            write_csv(output / f"{scope}_summary.csv", summary)
    print(f"Timing CSV tables: {output}")


if __name__ == "__main__":
    main()
