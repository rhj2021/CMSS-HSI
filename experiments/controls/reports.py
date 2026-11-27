"""CSV summaries with separate detector and intervention averaging."""

from pathlib import Path

import pandas as pd

from .experiment import variants
from .util import read_json, require, write_csv


def summarize(output, args, families):
    output = Path(output)
    weights, features = variants()
    selected = (weights if "weights" in families else []) + (
        features if "features" in families else []
    )
    geometries, quality, metrics, coverage = [], [], [], []
    for scene in args.datasets:
        for transform in args.transforms:
            for variant in selected:
                folder = output / "cases" / scene / transform / variant.name
                metadata_path = folder / "trace.json"
                require(
                    metadata_path.exists(), f"Missing control result: {metadata_path}"
                )
                metadata = read_json(metadata_path)
                geometries.append(metadata["geometry"])
                quality.extend(metadata["weight_quality_rows"])
                phases = ["post"] if variant.selector else ["pre"]
                if variant.name == "uniform":
                    phases.insert(0, "pre")
                count = 0
                for phase in phases:
                    for detector in args.detectors:
                        for seed in args.seeds:
                            path = (
                                folder / f"metrics_{phase}_{detector}_seed{seed}.json"
                            )
                            require(path.exists(), f"Missing detector result: {path}")
                            metrics.append(read_json(path))
                            count += 1
                coverage.append(
                    dict(
                        dataset=scene,
                        transform=transform,
                        variant=variant.name,
                        geometric_cases=1,
                        detection_rows=count,
                        expected_detection_rows=len(phases)
                        * len(args.detectors)
                        * len(args.seeds),
                    )
                )
    geometry = pd.DataFrame(geometries)
    detection = pd.DataFrame(metrics)
    require(
        not geometry.duplicated(["dataset", "transform", "variant"]).any(),
        "Duplicate geometric controls",
    )
    require(
        not detection.duplicated(
            ["dataset", "transform", "variant", "phase", "detector", "seed"]
        ).any(),
        "Duplicate detector controls",
    )
    tables = output / "tables"
    write_csv(tables / "geometry_per_case.csv", geometry)
    write_csv(tables / "weight_quality_per_iteration.csv", quality)
    write_csv(tables / "detection_per_trial.csv", detection)
    write_csv(tables / "coverage.csv", coverage)

    geometric_rows = []
    for (family, group, repeat), frame in geometry.groupby(
        ["experiment", "group", "intervention_seed"]
    ):
        row = dict(
            experiment=family,
            group=group,
            intervention_seed=int(repeat),
            geometric_cases=len(frame),
        )
        for field in (
            "EPE_pre",
            "EPE_post",
            "EPE_identity",
            "used_AP_mean",
            "used_AUROC_mean",
            "achieved_fraction_of_domain",
            "initial_effective_changed_pixels",
        ):
            if field in frame:
                values = pd.to_numeric(frame[field], errors="coerce")
                row[field] = float(values.mean()) if values.notna().any() else None
        geometric_rows.append(row)
    per_repeat = pd.DataFrame(geometric_rows)
    write_csv(tables / "geometry_by_intervention_seed.csv", per_repeat)
    summary = []
    names = {"used_AP_mean": "AP", "used_AUROC_mean": "AUROC"}
    for (family, group), frame in per_repeat.groupby(["experiment", "group"]):
        row = dict(
            experiment=family,
            group=group,
            intervention_repeats=len(frame),
            geometric_cases_per_repeat=int(frame.geometric_cases.min()),
        )
        for field in (
            "EPE_pre",
            "EPE_post",
            "EPE_identity",
            "used_AP_mean",
            "used_AUROC_mean",
            "achieved_fraction_of_domain",
        ):
            if field not in frame:
                continue
            values = pd.to_numeric(frame[field], errors="coerce").dropna()
            label = names.get(field, field)
            row[label + "_mean"] = float(values.mean()) if len(values) else None
            row[label + "_sd"] = float(values.std(ddof=1)) if len(values) > 1 else None
        summary.append(row)
    write_csv(tables / "geometry_summary.csv", summary)

    recovery = []
    for (family, group, repeat, phase), frame in detection.groupby(
        ["experiment", "group", "intervention_seed", "phase"]
    ):
        valid = frame[frame.gap_valid.astype(bool)]
        recovery.append(
            dict(
                experiment=family,
                group=group,
                intervention_seed=int(repeat),
                phase=phase,
                total_trials=len(frame),
                valid_trials=len(valid),
                positive=int((valid.delta_F1 > 0).sum()),
                P05_GapRec=float(valid.GapRec.quantile(0.05)) if len(valid) else None,
                median_delta_F1_all=float(frame.delta_F1.median()),
                median_delta_F1_valid=(
                    float(valid.delta_F1.median()) if len(valid) else None
                ),
            )
        )
    write_csv(tables / "recovery_by_intervention_seed.csv", recovery)
    keys = [
        "experiment",
        "group",
        "intervention_seed",
        "phase",
        "dataset",
        "detector",
        "seed",
    ]
    per_detector_seed = (
        detection.groupby(keys)[["F1", "IoU", "BoundaryF1"]].mean().reset_index()
    )
    write_csv(tables / "detection_per_detector_seed.csv", per_detector_seed)
    combined = per_detector_seed.groupby(keys[:-1])[["F1", "IoU", "BoundaryF1"]].agg(
        ["mean", "std", "count"]
    )
    combined.columns = [f"{metric}_{stat}" for metric, stat in combined.columns]
    write_csv(tables / "detection_mean_sd.csv", combined.reset_index())

    weight_quality = pd.DataFrame(quality)
    if len(weight_quality) and "used_prevalence" in weight_quality:
        reference = weight_quality[
            (weight_quality.variant == "irmad_default") & (weight_quality.iteration > 1)
        ]
        if len(reference):
            prevalence = reference.groupby(
                ["dataset", "transform"]
            ).used_prevalence.mean()
            write_csv(
                tables / "constant_score_reference.csv",
                [
                    dict(
                        cases=len(prevalence),
                        mean_change_prevalence=float(prevalence.mean()),
                    )
                ],
            )
    return tables
