"""Matched controls with fixed detector checkpoints and decision thresholds."""

from __future__ import annotations
from dataclasses import asdict
from pathlib import Path
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from .util import (
    require,
    read_json,
    write_json,
    write_npz,
    sha_array,
    sha_file,
    object_hash,
    close,
    log,
)
from .core import Variant, Trace, run, restore


def variants():
    weights = [
        Variant("irmad_default", "weights"),
        Variant("uniform", "weights", weighting="uniform"),
    ]
    for fraction in (0.25, 0.5):
        for seed in (1001, 1002, 1003):
            weights.append(
                Variant(
                    f"block_flip_p{int(round(fraction * 100)):02d}_r{seed}",
                    "weights",
                    weighting="block_flip",
                    fraction=fraction,
                    intervention_seed=seed,
                    block_size=24,
                )
            )
    features = [
        Variant(name, "features", representation=name, selector=False)
        for name in ("pca1", "pca3", "full")
    ]
    return weights, features


def quality(score, gt, mask, threshold):
    valid = np.asarray(mask, bool) & (gt >= 0) & np.isfinite(score)
    y = gt[valid].astype(np.int8)
    z = np.asarray(score[valid], np.float64)
    npos = int((y == 1).sum())
    nneg = int((y == 0).sum())
    if not npos or not nneg:
        return dict(
            AP=None,
            AUROC=None,
            N=int(len(y)),
            changed=npos,
            unchanged=nneg,
            prevalence=float(y.mean()) if len(y) else None,
            FNR=None,
            FPR=None,
        )
    pred = z > threshold
    return dict(
        AP=float(average_precision_score(y, z)),
        AUROC=float(roc_auc_score(y, z)),
        N=int(len(y)),
        changed=npos,
        unchanged=nneg,
        prevalence=float(npos / len(y)),
        FNR=float(np.sum(~pred & (y == 1)) / npos),
        FPR=float(np.sum(pred & (y == 0)) / nneg),
    )


def summarize_trace(case, trace, variant, cfg):
    """Evaluate geometry and internal change responses after correction."""
    from cmss.corrections import flow_errors

    valid = case.valid
    gt = case.dataset.gt
    rows = []
    eps = float(cfg["weighting"]["epsilon"])
    change_thr = (1 - eps) / 2
    for stage in trace.stages:
        it = stage["iteration"]
        rec = dict(
            dataset=case.dataset.name,
            transform=case.transform,
            variant=variant.name,
            experiment=variant.experiment,
            group=variant.group,
            intervention_seed=variant.intervention_seed,
            **stage,
        )
        if variant.weighting != "uniform":
            for kind, key in (
                ("used", f"weights_used_{it}"),
                ("clean", f"weights_clean_{it}"),
                ("injected", f"weights_injected_{it}"),
            ):
                q = quality(1 - trace.arrays[key], gt, valid, change_thr)
                rec.update({f"{kind}_{k}": v for k, v in q.items()})
        rec["EPE_after_update"] = flow_errors(
            trace.arrays[f"flow_update_{it}"], case.truth, valid
        )["flow_epe_mean"]
        rows.append(rec)
    epre = flow_errors(trace.raw.flow, case.truth, valid)
    epost = flow_errors(trace.selected.flow, case.truth, valid)
    selected = trace.selected.metadata
    used = [r for r in rows if r["iteration"] > 1]

    def mean_field(rr, k):
        x = [r.get(k) for r in rr if r.get(k) is not None]
        return float(np.mean(x)) if x else None

    geom = dict(
        dataset=case.dataset.name,
        transform=case.transform,
        variant=variant.name,
        group=variant.group,
        experiment=variant.experiment,
        representation=variant.representation,
        weighting=variant.weighting,
        requested_fraction=variant.fraction,
        intervention_seed=variant.intervention_seed,
        EPE_pre=epre["flow_epe_mean"],
        EPE_post=epost["flow_epe_mean"],
        EPE_identity=flow_errors(np.zeros_like(case.truth), case.truth, valid)[
            "flow_epe_mean"
        ],
        selector_enabled=variant.selector,
        selector_choice=selected.get("selector_choice", "disabled"),
        used_AP_mean=mean_field(used, "used_AP"),
        used_AUROC_mean=mean_field(used, "used_AUROC"),
        final_AP=rows[-1].get("injected_AP"),
        final_AUROC=rows[-1].get("injected_AUROC"),
        final_FNR=rows[-1].get("injected_FNR"),
        final_FPR=rows[-1].get("injected_FPR"),
        geometry_mask_sha256=sha_array(valid),
        pre_flow_sha256=sha_array(trace.raw.flow),
        post_flow_sha256=sha_array(trace.selected.flow),
        raw_horizontal_noninteger_fraction=float(
            np.mean(
                np.abs(
                    trace.arrays["raw_row_dx"] - np.round(trace.arrays["raw_row_dx"])
                )
                > 1e-6
            )
        ),
        raw_global_vertical=float(trace.arrays["raw_global_dy"]),
    )
    geom.update(trace.intervention)
    for k, v in selected.items():
        if k.startswith("selector_") and not isinstance(v, (dict, list)):
            geom[k] = v
    return geom, rows


class Experiment:
    def __init__(self, source, args, output):
        self.source = source
        self.args = args
        self.out = Path(output)
        self.weights, self.features = variants()
        self.reference_cache = {}

    def pca(self, case):
        from cmss.corrections import global_pca_translation

        c = self.source.correction
        return global_pca_translation(
            case.dataset.t1,
            case.moving,
            device=self.source.device,
            seed=int(c["seed"]),
            n_components=int(c["pca_components"]),
            global_radius=int(c["global_radius"]),
        )

    def obtain(self, case, variant, pca):
        """Load a verified cached trace or compute it in the output directory."""
        folder = self.out / "cases" / case.dataset.name / case.transform / variant.name
        meta_path = folder / "trace.json"
        arr_path = folder / "trace.npz"
        if meta_path.is_file() and arr_path.is_file():
            meta = read_json(meta_path)
            require(
                meta["source_protocol"] == self.source.protocol
                and meta["case_signature"] == case.signature,
                f"Cached input configuration differs: {folder}",
            )
            require(
                meta["variant"] == asdict(variant), f"Cached variant mismatch {folder}"
            )
            require(
                meta["arrays_sha256"] == sha_file(arr_path),
                f"Corrupt trace array cache {arr_path}",
            )
            with np.load(arr_path, allow_pickle=False) as d:
                arrays = {k: d[k] for k in d.files}
            trace = restore(
                case.dataset.t1,
                case.moving,
                self.source.correction,
                self.source.device,
                meta,
                arrays,
                pca,
            )
            return trace, folder, meta
        log(f"GEOMETRY {case.dataset.name}/{case.transform}/{variant.name}")
        trace = run(
            case.dataset.t1,
            case.moving,
            self.source.correction,
            self.source.device,
            variant,
            f"{case.dataset.name}:{case.transform}",
            pca=pca,
            collect=True,
        )
        geom, rows = summarize_trace(case, trace, variant, self.source.correction)
        write_npz(arr_path, **trace.arrays)
        meta = dict(
            variant=asdict(variant),
            source_protocol=self.source.protocol,
            case_signature=case.signature,
            arrays_sha256=sha_file(arr_path),
            raw_cube_sha256=sha_array(trace.raw.corrected),
            selected_cube_sha256=sha_array(trace.selected.corrected),
            raw_metadata=trace.raw.metadata,
            selected_metadata=trace.selected.metadata,
            stages=trace.stages,
            intervention=trace.intervention,
            geometry=geom,
            weight_quality_rows=rows,
        )
        write_json(meta_path, meta)
        return trace, folder, read_json(meta_path)

    def prediction(self, case, cube, detector, seed, mask, indices):
        _, _, checkpoint_hash = self.source.checkpoint(
            case.dataset.name, detector, seed
        )
        expected_threshold = float(
            self.source.row(
                case.dataset.name, case.transform, detector, seed, "misaligned"
            ).threshold
        )
        signature = dict(
            cube=sha_array(cube),
            indices=sha_array(indices),
            checkpoint=checkpoint_hash,
            protocol=self.source.protocol,
            threshold=expected_threshold,
        )
        key = object_hash(signature)
        path = self.out / "predictions" / key[:2] / (key + ".npz")
        metadata_path = path.with_suffix(".json")
        if path.exists() and metadata_path.exists():
            metadata = read_json(metadata_path)
            require(
                metadata["signature"] == signature
                and metadata["sha256"] == sha_file(path),
                f"Prediction cache changed: {path}",
            )
            with np.load(path, allow_pickle=False) as saved:
                require(
                    np.array_equal(saved["indices"], indices),
                    "Prediction support changed",
                )
                probability = saved["prob"].astype(np.float32)
                threshold = float(saved["threshold"].item())
        else:
            probability, threshold = self.source.predict(
                case, cube, detector, seed, indices
            )
            write_npz(
                path,
                indices=indices,
                prob=probability,
                threshold=np.asarray(threshold, np.float64),
            )
            write_json(metadata_path, dict(signature=signature, sha256=sha_file(path)))
        require(
            close(threshold, expected_threshold, atol=1e-10, rtol=0),
            "Cached decision threshold changed",
        )
        metrics = self.source.check_prediction(
            case, probability, threshold, indices, mask, detector, seed
        )
        return dict(
            metrics,
            threshold=threshold,
            prediction_file=str(path.relative_to(self.out)),
            checkpoint_files_hash=checkpoint_hash,
            indices_sha256=signature["indices"],
            cube_sha256=signature["cube"],
        )

    def references(self, case, detector, seed, mask, indices):
        key = (case.dataset.name, case.transform, detector, seed)
        if key not in self.reference_cache:
            self.reference_cache[key] = {
                "clean_reference": self.prediction(
                    case, case.dataset.t2, detector, seed, mask, indices
                ),
                "misaligned": self.prediction(
                    case, case.moving, detector, seed, mask, indices
                ),
            }
        return self.reference_cache[key]

    def evaluate(self, case, trace, variant, folder):
        phases = (
            [("post", trace.selected.corrected)]
            if variant.selector
            else [("pre", trace.raw.corrected)]
        )
        if variant.name == "uniform":
            phases.insert(0, ("pre", trace.raw.corrected))
        cutoff = float(self.source.cfg["report"]["gap_min_denominator"])
        for detector in self.args.detectors:
            for seed in self.args.seeds:
                mask, indices = self.source.support(case, detector, seed)
                refs = self.references(case, detector, seed, mask, indices)
                clean = float(refs["clean_reference"]["F1"])
                misaligned = float(refs["misaligned"]["F1"])
                gap = clean - misaligned
                for phase, cube in phases:
                    metrics = self.prediction(case, cube, detector, seed, mask, indices)
                    delta = float(metrics["F1"] - misaligned)
                    record = dict(
                        dataset=case.dataset.name,
                        transform=case.transform,
                        variant=variant.name,
                        group=variant.group,
                        experiment=variant.experiment,
                        phase=phase,
                        detector=detector,
                        seed=seed,
                        intervention_seed=variant.intervention_seed,
                        requested_fraction=variant.fraction,
                        representation=variant.representation,
                        weighting=variant.weighting,
                        clean_F1=clean,
                        misaligned_F1=misaligned,
                        initial_drop=gap,
                        delta_F1=delta,
                        gap_valid=bool(gap > cutoff),
                        GapRec=delta / gap if gap > cutoff else None,
                        **metrics,
                    )
                    write_json(
                        folder / f"metrics_{phase}_{detector}_seed{seed}.json", record
                    )

    def save_feature_default(self, case, default, variant):
        trace = Trace(
            default.raw, default.raw, dict(default.arrays), default.stages, {}, "pca3"
        )
        trace.arrays["flow_post"] = default.raw.flow.copy()
        folder = self.out / "cases" / case.dataset.name / case.transform / variant.name
        geometry, rows = summarize_trace(case, trace, variant, self.source.correction)
        write_npz(folder / "trace.npz", **trace.arrays)
        metadata = dict(
            variant=asdict(variant),
            source_protocol=self.source.protocol,
            case_signature=case.signature,
            arrays_sha256=sha_file(folder / "trace.npz"),
            raw_cube_sha256=sha_array(trace.raw.corrected),
            selected_cube_sha256=sha_array(trace.selected.corrected),
            raw_metadata=trace.raw.metadata,
            selected_metadata=trace.selected.metadata,
            stages=trace.stages,
            intervention={},
            geometry=geometry,
            weight_quality_rows=rows,
        )
        write_json(folder / "trace.json", metadata)
        return trace, folder

    def execute(self, stage):
        for name in self.args.datasets:
            for transform in self.args.transforms:
                case = self.source.case(name, transform)
                pca = self.pca(case)
                default, default_folder, _ = self.obtain(case, self.weights[0], pca)
                self.source.default_check(case, default)
                if stage in {"all", "weights"}:
                    for variant in self.weights:
                        if variant.name == "irmad_default":
                            trace, folder = default, default_folder
                        else:
                            trace, folder, _ = self.obtain(case, variant, pca)
                        if variant.name == "uniform":
                            self.source.raw_baseline_check(
                                case, trace, "scanline_unweighted"
                            )
                        self.evaluate(case, trace, variant, folder)
                        if variant.name != "irmad_default":
                            del trace
                if stage in {"all", "features"}:
                    for variant in self.features:
                        if variant.name == "pca3":
                            trace, folder = self.save_feature_default(
                                case, default, variant
                            )
                        else:
                            trace, folder, _ = self.obtain(case, variant, pca)
                        self.evaluate(case, trace, variant, folder)
                        del trace
                del default, case, pca
            self.reference_cache.clear()
            self.source.release_dataset(name)
