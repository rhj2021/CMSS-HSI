"""Benchmark summaries with separate detector-seed and geometry statistics."""

from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

METRICS = ["F1", "IoU", "BoundaryF1"]


def summarize_conditions(df: pd.DataFrame) -> pd.DataFrame:
    keys = [
        x
        for x in ["suite", "dataset", "baseline", "transform", "correction"]
        if x in df
    ]
    if df.duplicated(keys + ["seed"]).any():
        raise ValueError(
            "Duplicate condition/seed rows: refusing to average mixed runs"
        )
    out = df.groupby(keys, sort=False)[METRICS].agg(["mean", "std", "count"])
    out.columns = [f"{a}_{b}" for a, b in out.columns]
    return out.reset_index()


def summarize(root, cfg):
    root = Path(root)
    out = root / "summary"
    out.mkdir(parents=True, exist_ok=True)
    p = root / "metrics/conditions.csv"
    if p.exists():
        d = pd.read_csv(p)
        keys = ["suite", "dataset", "seed", "baseline", "transform"]
        threshold_counts = d.groupby(keys).threshold.nunique()
        if (threshold_counts > 1).any():
            raise ValueError(
                "Different detector thresholds within a matched comparison"
            )
        if (d.groupby(keys).N.nunique() > 1).any():
            raise ValueError(
                "Different evaluation supports within a matched comparison"
            )
        summarize_conditions(d).to_csv(
            out / "per_perturbation_mean_std.csv", index=False
        )
        clean = d[d.correction == "clean_reference"][keys + ["F1"]].rename(
            columns={"F1": "clean_F1"}
        )
        mis = d[d.correction == "misaligned"][keys + ["F1"]].rename(
            columns={"F1": "mis_F1"}
        )
        paired = d.merge(clean, on=keys, validate="many_to_one").merge(
            mis, on=keys, validate="many_to_one"
        )
        gap = paired.clean_F1 - paired.mis_F1
        paired["delta_F1_vs_misaligned"] = paired.F1 - paired.mis_F1
        paired["delta_F1_vs_clean"] = paired.F1 - paired.clean_F1
        paired["GapRec"] = np.where(
            (gap > float(cfg["report"]["gap_min_denominator"]))
            & (paired["transform"] != "clean_no_harm"),
            paired.delta_F1_vs_misaligned / gap,
            np.nan,
        )
        paired.to_csv(out / "paired_per_seed.csv", index=False)
        rows = []
        for (suite, method), g in paired[
            ~paired.correction.isin(["clean_reference", "misaligned", "oracle"])
        ].groupby(["suite", "correction"]):
            pert = g[g["transform"] != "clean_no_harm"]
            c = g[g["transform"] == "clean_no_harm"]
            v = pert.GapRec.dropna()
            rows.append(
                dict(
                    suite=suite,
                    correction=method,
                    n_perturbed_trials=len(pert),
                    valid_GapRec_n=len(v),
                    positive_GapRec_n=int((v > 0).sum()),
                    median_GapRec=float(v.median()) if len(v) else np.nan,
                    p05_GapRec=float(v.quantile(0.05)) if len(v) else np.nan,
                    median_delta_F1=(
                        float(pert.delta_F1_vs_misaligned.median())
                        if len(pert)
                        else np.nan
                    ),
                    clean_n=len(c),
                    clean_negative_n=int((c.delta_F1_vs_clean < -1e-12).sum()),
                    clean_worst_delta_F1=(
                        float(c.delta_F1_vs_clean.min()) if len(c) else np.nan
                    ),
                    clean_median_delta_F1=(
                        float(c.delta_F1_vs_clean.median()) if len(c) else np.nan
                    ),
                )
            )
        pd.DataFrame(rows).to_csv(out / "robustness_and_safety.csv", index=False)
        paired[paired["transform"] == "clean_no_harm"].to_csv(
            out / "clean_per_seed.csv", index=False
        )
        cover = (
            d.groupby(["suite", "dataset", "baseline", "transform", "correction"])
            .seed.agg(["nunique", lambda x: ",".join(map(str, sorted(set(x))))])
            .reset_index()
        )
        cover.columns = [
            "suite",
            "dataset",
            "baseline",
            "transform",
            "correction",
            "seed_count",
            "seeds",
        ]
        cover["configured_seed_count"] = len(cfg["project"]["seeds"])
        cover["all_configured_seeds"] = cover.seed_count == len(cfg["project"]["seeds"])
        cover.to_csv(out / "coverage.csv", index=False)
    gpath = root / "metrics/geometry.csv"
    if gpath.exists():
        g = pd.read_csv(gpath)
        g.to_csv(out / "geometry_per_case.csv", index=False)
        pert = g[g["transform"] != "clean_no_harm"]
        if len(pert):
            mean = pert.groupby(["suite", "correction"]).agg(
                n_corrections=("flow_epe_mean", "count"),
                mean_EPE=("flow_epe_mean", "mean"),
                median_of_case_mean_EPE=("flow_epe_mean", "median"),
                max_case_mean_EPE=("flow_epe_mean", "max"),
            )
            mean.to_csv(out / "geometry_summary.csv")
        g.groupby(
            ["suite", "correction", "selector_choice"], dropna=False
        ).size().rename("count").reset_index().to_csv(
            out / "selector_choices.csv", index=False
        )
        g[g.registration_status != "ok"].to_csv(
            out / "registration_failures.csv", index=False
        )
        g[g["transform"] == "clean_no_harm"].to_csv(
            out / "clean_geometry_checks.csv", index=False
        )
    if p.exists():
        main_tables(d, cfg, out)
        native_tables(d, out)
    return out


def main_tables(frame, cfg, out):
    """Average the three perturbations within each seed before summarizing seeds."""
    names = set(cfg["report"]["main_transforms"])
    main = frame[(frame["suite"] == "baselines") & frame["transform"].isin(names)]
    if main.empty:
        return
    keys = ["dataset", "baseline", "correction", "seed"]
    means = main.groupby(keys)[METRICS].mean()
    counts = main.groupby(keys)["transform"].nunique().rename("n_perturbations")
    per_seed = means.join(counts).reset_index()
    per_seed["expected_perturbations"] = len(names)
    per_seed.to_csv(out / "main_per_seed.csv", index=False)
    complete = per_seed[per_seed["n_perturbations"] == len(names)]
    if complete.empty:
        return
    summary = complete.groupby(["dataset", "baseline", "correction"])[METRICS].agg(
        ["mean", "std", "count"]
    )
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary = summary.reset_index()
    summary["configured_seed_count"] = len(cfg["project"]["seeds"])
    summary.to_csv(out / "main_mean_sd.csv", index=False)


def native_tables(frame, out):
    """Summarize original-input identity and CMSS results across detector seeds."""
    native = frame[
        frame["suite"].eq("clean")
        & frame["transform"].eq("clean_no_harm")
        & frame["correction"].isin(["misaligned", "cmss"])
    ].copy()
    if native.empty:
        return
    native["method"] = native["correction"].replace({"misaligned": "identity"})
    keys = ["dataset", "seed", "baseline", "method"]
    if native.duplicated(keys).any():
        raise ValueError("Duplicate original-input detector/seed/method results.")
    columns = keys + ["threshold", "N", "TP", "TN", "FP", "FN"] + METRICS
    native[columns].to_csv(out / "native_per_seed.csv", index=False)
    rows, paired = [], []
    for (dataset, detector, method), group in native.groupby(
        ["dataset", "baseline", "method"], sort=False
    ):
        record = dict(
            dataset=dataset,
            baseline=detector,
            method=method,
            n_seeds=int(group["seed"].nunique()),
            seeds=",".join(map(str, sorted(group["seed"].unique()))),
            N_min=int(group["N"].min()),
            N_max=int(group["N"].max()),
        )
        for metric in METRICS:
            values = group[metric].astype(float)
            record[metric + "_mean"] = float(values.mean())
            record[metric + "_std"] = (
                float(values.std(ddof=1)) if len(values) > 1 else np.nan
            )
        rows.append(record)
    for (dataset, seed, detector), group in native.groupby(
        ["dataset", "seed", "baseline"], sort=False
    ):
        identity = group[group["method"] == "identity"]
        if len(identity) != 1:
            raise ValueError(
                f"Missing or duplicate identity reference: {dataset}/{seed}/{detector}."
            )
        identity = identity.iloc[0]
        if group["N"].nunique() != 1 or group["threshold"].nunique() != 1:
            raise ValueError(
                "Original-input comparisons use different supports or thresholds."
            )
        for _, row in group.iterrows():
            record = dict(
                dataset=dataset,
                seed=seed,
                baseline=detector,
                method=row["method"],
                N=int(row.N),
            )
            for metric in METRICS:
                record["delta_" + metric] = float(row[metric] - identity[metric])
            paired.append(record)
    pd.DataFrame(rows).to_csv(out / "native_mean_sd.csv", index=False)
    pd.DataFrame(paired).to_csv(out / "native_paired_differences.csv", index=False)
