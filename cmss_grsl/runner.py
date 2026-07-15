from __future__ import annotations
import json, traceback, zlib
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
import yaml
from tqdm.auto import tqdm

from .baselines.getnet_adapter import predict_getnet, train_getnet_clean
from .baselines.glaformer import predict_glaformer, train_glaformer
from .baselines.irmad import run_irmad
from .corrections import cmss_correct, global_pca_translation, select_safe_cmss
from .data import DatasetBundle, load_dataset
from .metrics import binary_metrics
from .reporting import summarize_and_package
from .splits import balanced_indices, make_spatial_block_split
from .transforms import apply_transform, oracle_realign
from .utils import ensure_dir, load_yaml, save_json


class CMSSRunner:
    def __init__(self, config_path: str | Path, device_override: str | None = None, force: bool = False) -> None:
        self.config_path = Path(config_path).resolve()
        self.cfg = load_yaml(self.config_path)
        self.project_cfg = self.cfg['project']
        if device_override:
            self.project_cfg['device'] = device_override
        self.device = str(self.project_cfg.get('device', 'cuda:0'))
        self.force = bool(force)
        self.root = ensure_dir(self.project_cfg.get('output_dir', 'results/cmss_grsl'))
        self.cache_dir = ensure_dir(self.root / 'cache')
        self.ckpt_dir = ensure_dir(self.root / 'checkpoints')
        self.metrics_dir = ensure_dir(self.root / 'metrics')
        self.summary_dir = ensure_dir(self.root / 'summary')
        self.inventory_dir = ensure_dir(self.root / 'inventory')
        self.data_root = Path(self.project_cfg.get('data_root', 'data'))
        self.status_rows: list[dict[str, Any]] = []
        with open(self.root / 'resolved_config.yaml', 'w', encoding='utf-8') as f:
            yaml.safe_dump(self.cfg, f, sort_keys=False, allow_unicode=True)

    @property
    def datasets(self) -> list[str]:
        return [str(x) for x in self.project_cfg.get('datasets', ['River', 'Farmland', 'Hermiston'])]

    @property
    def seeds(self) -> list[int]:
        return [int(x) for x in self.project_cfg.get('seeds', [0, 1, 2])]

    def _progress_enabled(self) -> bool:
        return bool(self.project_cfg.get('progress', True))

    def _bar(self, iterable, **kwargs):
        kwargs.setdefault('dynamic_ncols', True)
        kwargs.setdefault('disable', not self._progress_enabled())
        return tqdm(iterable, **kwargs)

    def _with_progress(self, cfg: dict[str, Any], desc: str) -> dict[str, Any]:
        out = dict(cfg)
        out['progress'] = self._progress_enabled()
        out['progress_desc'] = desc
        return out

    def _dataset(self, name: str) -> DatasetBundle:
        options = self.cfg.get('dataset_options', {}).get(name, {})
        return load_dataset(name, self.data_root, options)

    def _record_status(self, stage: str, dataset: str, item: str, seed: int | None, status: str, message: str = '') -> None:
        self.status_rows.append({'stage': stage, 'dataset': dataset, 'item': item, 'seed': '' if seed is None else int(seed), 'status': status, 'message': message[:4000]})

    def _flush_status(self) -> None:
        if self.status_rows:
            pd.DataFrame(self.status_rows).to_csv(self.metrics_dir / 'run_status.csv', index=False)

    def inspect(self) -> None:
        rows = []
        for name in self._bar(self.datasets, desc='inspect datasets', unit='dataset'):
            try:
                ds = self._dataset(name)
                save_json(self.inventory_dir / f'{name}.json', ds.metadata)
                rows.append({'dataset': name, 'height': ds.t1.shape[0], 'width': ds.t1.shape[1], 'bands': ds.t1.shape[2], 'unchanged': int(np.sum(ds.gt == 0)), 'changed': int(np.sum(ds.gt == 1)), 'ignored': int(np.sum(ds.gt < 0))})
                self._record_status('inspect', name, 'dataset', None, 'ok')
            except Exception as exc:
                self._record_status('inspect', name, 'dataset', None, 'failed', f'{exc}\n{traceback.format_exc()}')
        pd.DataFrame(rows).to_csv(self.metrics_dir / 'dataset_inventory.csv', index=False)
        self._flush_status()

    def _split_path(self, dataset: str, seed: int) -> Path:
        return self.cache_dir / dataset / 'splits' / f'seed{seed}.npz'

    def _prepare_split(self, ds: DatasetBundle, seed: int) -> None:
        path = self._split_path(ds.name, seed)
        if path.exists() and not self.force:
            self._record_status('prepare', ds.name, 'split', seed, 'cached')
            return
        cfg = self.cfg['split']
        split = make_spatial_block_split(ds.gt, seed=seed, block_size=int(cfg.get('block_size', 24)), train_fraction=float(cfg.get('train_fraction', 0.15)), val_fraction=float(cfg.get('val_fraction', 0.10)), guard=int(cfg.get('guard', 3)))
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, train=split.train_mask, val=split.val_mask, test=split.test_mask)
        save_json(path.with_suffix('.json'), split.metadata)
        self._record_status('prepare', ds.name, 'split', seed, 'ok')

    def _transform_dir(self, dataset: str, transform_name: str) -> Path:
        return self.cache_dir / dataset / 'transforms' / transform_name

    def _transform_seed(self, dataset: str, transform_name: str) -> int:
        token = f"{dataset}:{transform_name}:{self.cfg.get('correction', {}).get('seed', 31415)}"
        return int(zlib.crc32(token.encode('utf-8')) & 0x7FFFFFFF)

    def _prepare_transform(self, ds: DatasetBundle, spec: dict[str, Any]) -> None:
        name = str(spec['name'])
        out = self._transform_dir(ds.name, name)
        required = [out / 'moving.npy', out / 'flow.npy', out / 'oracle.npy', out / 'metadata.json']
        if all(p.exists() for p in required) and not self.force:
            self._record_status('prepare', ds.name, f'transform:{name}', None, 'cached')
            return
        out.mkdir(parents=True, exist_ok=True)
        tr = apply_transform(ds.t2, spec, device=self.device, seed=self._transform_seed(ds.name, name))
        oracle, oracle_valid = oracle_realign(tr.moving, tr.forward_flow, device=self.device)
        np.save(out / 'moving.npy', tr.moving.astype(np.float32))
        np.save(out / 'flow.npy', tr.forward_flow.astype(np.float32))
        np.save(out / 'oracle.npy', oracle.astype(np.float32))
        meta = {**tr.metadata, 'oracle_valid_fraction': float((tr.valid_reference & oracle_valid).mean())}
        save_json(out / 'metadata.json', meta)
        self._record_status('prepare', ds.name, f'transform:{name}', None, 'ok')

    def prepare(self) -> None:
        for name in self._bar(self.datasets, desc='prepare datasets', unit='dataset'):
            try:
                ds = self._dataset(name)
                for seed in self._bar(self.seeds, desc=f'prepare splits {name}', unit='seed', leave=False):
                    self._prepare_split(ds, seed)
                for raw in self._bar(self.cfg.get('transforms', []), desc=f'prepare transforms {name}', unit='transform', leave=False):
                    self._prepare_transform(ds, dict(raw))
            except Exception as exc:
                self._record_status('prepare', name, 'dataset', None, 'failed', f'{exc}\n{traceback.format_exc()}')
        self._flush_status()

    def _load_split(self, dataset: str, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        d = np.load(self._split_path(dataset, seed))
        return d['train'].astype(bool), d['val'].astype(bool), d['test'].astype(bool)

    def train(self) -> None:
        split_cfg = self.cfg['split']
        for name in self._bar(self.datasets, desc='train datasets', unit='dataset'):
            try:
                ds = self._dataset(name)
            except Exception as exc:
                self._record_status('train', name, 'dataset', None, 'failed', f'{exc}\n{traceback.format_exc()}')
                continue
            for seed in self._bar(self.seeds, desc=f'train seeds {name}', unit='seed', leave=False):
                try:
                    train_mask, val_mask, _ = self._load_split(name, seed)
                    train_idx = balanced_indices(
                        train_mask,
                        ds.gt,
                        max_per_class=int(split_cfg.get('max_train_per_class', 7000)),
                        seed=seed,
                    )
                    val_idx = balanced_indices(
                        val_mask,
                        ds.gt,
                        max_per_class=int(split_cfg.get('max_val_per_class', 4000)),
                        seed=seed + 1000,
                    )
                except Exception as exc:
                    self._record_status('train', name, 'split', seed, 'failed', f'{exc}\n{traceback.format_exc()}')
                    continue

                gla_cfg = self.cfg.get('baselines', {}).get('GLAFormer', {})
                if bool(gla_cfg.get('enabled', True)):
                    out = self.ckpt_dir / name / f'seed{seed}' / 'GLAFormer'
                    if (out / 'best.pt').exists() and not self.force:
                        self._record_status('train', name, 'GLAFormer', seed, 'cached')
                    else:
                        try:
                            train_glaformer(ds.t1, ds.t2, ds.gt, train_idx, val_idx, out, self._with_progress(gla_cfg, f'train {name}/seed{seed}/GLAFormer'), self.device, seed)
                            self._record_status('train', name, 'GLAFormer', seed, 'ok')
                        except Exception as exc:
                            self._record_status('train', name, 'GLAFormer', seed, 'failed', f'{exc}\n{traceback.format_exc()}')
                get_cfg = self.cfg.get('baselines', {}).get('GETNET', {})
                if bool(get_cfg.get('enabled', True)):
                    out = self.ckpt_dir / name / f'seed{seed}' / 'GETNET'
                    if (out / 'best.pt').exists() and not self.force:
                        self._record_status('train', name, 'GETNET', seed, 'cached')
                    else:
                        try:
                            train_getnet_clean(ds.t1, ds.t2, ds.gt, val_idx, out, self._with_progress(get_cfg, f'train {name}/seed{seed}/GETNET'), self.device, seed)
                            self._record_status('train', name, 'GETNET', seed, 'ok')
                        except Exception as exc:
                            self._record_status('train', name, 'GETNET', seed, 'failed', f'{exc}\n{traceback.format_exc()}')
        self._flush_status()

    @staticmethod
    def _fill_map(shape: tuple[int, int], indices: np.ndarray, values: np.ndarray) -> np.ndarray:
        out = np.full(shape, np.nan, dtype=np.float32)
        out.reshape(-1)[indices] = values
        return out

    def _save_viz_prediction(self, dataset: str, seed: int, baseline: str, transform: str, correction: str, indices: np.ndarray, prob: np.ndarray, threshold: float) -> None:
        viz_dir = self.cache_dir / dataset / 'viz'
        viz_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            viz_dir / f'seed{seed}_{baseline}_{transform}_{correction}.npz',
            indices=indices.astype(np.int64),
            prob=prob.astype(np.float32),
            threshold=np.asarray(threshold, dtype=np.float32),
        )

    def _deep_prediction_path(self, dataset: str, seed: int, baseline: str, key: str) -> Path:
        return self.cache_dir / dataset / 'predictions' / f'seed{seed}' / baseline / f'{key}.npz'

    def _predict_deep(self, baseline: str, ds: DatasetBundle, seed: int, t2_condition: np.ndarray, indices: np.ndarray, key: str) -> tuple[np.ndarray, float]:
        path = self._deep_prediction_path(ds.name, seed, baseline, key)
        if path.exists() and not self.force:
            d = np.load(path)
            if np.array_equal(d['indices'], indices):
                return d['prob'].astype(np.float32), float(d['threshold'])
        base = self.ckpt_dir / ds.name / f'seed{seed}' / baseline
        if baseline == 'GLAFormer':
            cfg = self.cfg['baselines']['GLAFormer']
            prob, out_idx, threshold = predict_glaformer(base / 'best.pt', base / 'normalizer.npz', ds.t1, t2_condition, ds.gt, indices, int(cfg.get('eval_batch_size', 128)), self.device, int(cfg.get('num_workers', 0)), progress_desc=f'predict {baseline} {ds.name}/seed{seed}/{key}', show_progress=self._progress_enabled())
        elif baseline == 'GETNET':
            cfg = self.cfg['baselines']['GETNET']
            prob, out_idx, threshold = predict_getnet(base / 'best.pt', base / 'normalizer.npz', ds.t1, t2_condition, indices, int(cfg.get('eval_batch_size', 128)), self.device, int(cfg.get('num_workers', 0)), progress_desc=f'predict {baseline} {ds.name}/seed{seed}/{key}', show_progress=self._progress_enabled())
        else:
            raise ValueError(baseline)
        if not np.array_equal(out_idx, indices):
            lookup = {int(v): i for i, v in enumerate(out_idx)}
            prob = np.asarray([prob[lookup[int(i)]] for i in indices], dtype=np.float32)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, indices=indices.astype(np.int64), prob=prob.astype(np.float32), threshold=np.asarray(threshold, dtype=np.float32))
        return prob, float(threshold)

    def _predict_irmad(self, ds: DatasetBundle, t2_condition: np.ndarray, key: str) -> tuple[np.ndarray, np.ndarray, float, dict[str, Any]]:
        path = self.cache_dir / ds.name / 'irmad' / f'{key}.npz'
        if path.exists() and not self.force:
            d = np.load(path, allow_pickle=True)
            return d['score'].astype(np.float32), d['pred'].astype(np.uint8), float(d['threshold']), json.loads(str(d['metadata'].item()))
        cfg = self.cfg['baselines']['IRMAD']
        r = run_irmad(ds.t1, t2_condition, pca_components=int(cfg.get('pca_components', 20)), max_iter=int(cfg.get('max_iter', 30)), tolerance=float(cfg.get('tolerance', 1e-3)), regularization=float(cfg.get('regularization', 1e-3)), sample_size=int(cfg.get('sample_size', 50000)), seed=0, show_progress=self._progress_enabled(), progress_desc=f'IRMAD {ds.name}/{key}')
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, score=r.score.astype(np.float32), pred=r.pred.astype(np.uint8), threshold=np.asarray(r.threshold, dtype=np.float32), metadata=np.asarray(json.dumps(r.metadata)))
        return r.score, r.pred, float(r.threshold), r.metadata

    def _condition_row(self, dataset: str, seed: int, baseline: str, transform: str, correction: str, gt: np.ndarray, mask: np.ndarray, prob: np.ndarray, threshold: float) -> dict[str, Any]:
        return {'dataset': dataset, 'seed': int(seed), 'baseline': baseline, 'transform': transform, 'correction': correction, 'threshold': float(threshold), 'evaluation_fraction': float(mask.mean()), **binary_metrics(prob >= threshold, gt, mask=mask, prob=prob)}

    def _correction_path(self, dataset: str, transform: str, method: str) -> Path:
        return self.cache_dir / dataset / 'corrections' / transform / method

    def _compute_corrections(self, ds: DatasetBundle, transform: str, moving: np.ndarray, true_flow: np.ndarray) -> dict[str, dict[str, Any]]:
        cfg = self.cfg.get('correction', {})
        out: dict[str, dict[str, Any]] = {}
        methods = cfg.get('methods', ['pc1_translation', 'pca3_translation', 'scanline_unweighted', 'cmss_k1', 'cmss'])
        for method in self._bar(methods, desc=f'correct {ds.name}/{transform}', unit='method', leave=False):
            method = str(method)
            cdir = self._correction_path(ds.name, transform, str(method))
            cdir.mkdir(parents=True, exist_ok=True)
            img_path = cdir / 'corrected.npy'
            meta_path = cdir / 'metadata.json'
            flow_path = cdir / 'flow.npy'
            if img_path.exists() and meta_path.exists() and flow_path.exists() and not self.force:
                corrected = np.load(img_path).astype(np.float32)
                flow = np.load(flow_path).astype(np.float32)
                meta = json.loads(meta_path.read_text(encoding='utf-8'))
            else:
                if method == 'pc1_translation':
                    r = global_pca_translation(ds.t1, moving, true_flow, device=self.device, seed=int(cfg.get('seed', 31415)), n_components=1, global_radius=int(cfg.get('global_radius', 6)))
                elif method == 'pca3_translation':
                    r = global_pca_translation(ds.t1, moving, true_flow, device=self.device, seed=int(cfg.get('seed', 31415)), n_components=3, global_radius=int(cfg.get('global_radius', 6)))
                elif method == 'scanline_unweighted':
                    r = cmss_correct(ds.t1, moving, true_flow, device=self.device, seed=int(cfg.get('seed', 31415)), n_components=int(cfg.get('pca_components', 3)), global_radius=int(cfg.get('global_radius', 6)), row_radius=int(cfg.get('row_radius', 3)), smooth_sigma=float(cfg.get('smooth_sigma', 6.0)), outer_iters=int(cfg.get('outer_iters', 3)), use_weights=False, show_progress=self._progress_enabled(), progress_desc=f'CMSS-unweighted {ds.name}/{transform}')
                elif method == 'cmss_k1':
                    # Single-component CMSS ablation.
                    r = cmss_correct(ds.t1, moving, true_flow, device=self.device, seed=int(cfg.get('seed', 31415)), n_components=1, global_radius=int(cfg.get('global_radius', 6)), row_radius=int(cfg.get('row_radius', 3)), smooth_sigma=float(cfg.get('smooth_sigma', 6.0)), outer_iters=int(cfg.get('outer_iters', 3)), use_weights=True, show_progress=self._progress_enabled(), progress_desc=f'CMSS-K1 {ds.name}/{transform}')
                    r.metadata['method'] = 'CMSS-K1'
                    r.metadata['ablation'] = 'single_component_cmss'
                elif method == 'cmss':
                    raw = cmss_correct(ds.t1, moving, true_flow, device=self.device, seed=int(cfg.get('seed', 31415)), n_components=int(cfg.get('pca_components', 3)), global_radius=int(cfg.get('global_radius', 6)), row_radius=int(cfg.get('row_radius', 3)), smooth_sigma=float(cfg.get('smooth_sigma', 6.0)), outer_iters=int(cfg.get('outer_iters', 3)), use_weights=True, show_progress=self._progress_enabled(), progress_desc=f'CMSS {ds.name}/{transform}')
                    # Safe selection is applied only to the full CMSS method.
                    pca_ref = out.get('pca3_translation')
                    pca_res = None
                    if pca_ref is not None:
                        from .corrections import CorrectionResult
                        pca_res = CorrectionResult(pca_ref['corrected'], pca_ref['flow'], pca_ref['metadata'])
                    r = select_safe_cmss(
                        ds.t1,
                        moving,
                        raw,
                        pca_result=pca_res,
                        true_flow=true_flow,
                        seed=int(cfg.get('seed', 31415)),
                        n_components=int(cfg.get('pca_components', 3)),
                        cfg=cfg.get('safe_selector', {}),
                    )
                else:
                    raise ValueError(method)
                corrected, flow, meta = r.corrected, r.flow, r.metadata
                np.save(img_path, corrected.astype(np.float32))
                np.save(flow_path, flow.astype(np.float32))
                save_json(meta_path, meta)
            out[str(method)] = {'corrected': corrected, 'flow': flow, 'metadata': meta}
        return out

    def _evaluate_clean_noharm(self, ds: DatasetBundle, condition_rows: list[dict[str, Any]], correction_rows: list[dict[str, Any]]) -> None:
        """Evaluate whether synchronization harms already registered image pairs.

        These rows use transform='clean_no_harm' and are summarized separately
        rather than mixed into gap-recovery, because there is no misalignment gap.
        """
        exp_cfg = self.cfg.get('experiments', {})
        if not bool(exp_cfg.get('clean_no_harm', False)):
            return
        tname = str(exp_cfg.get('clean_no_harm_transform', 'clean_no_harm'))
        true_flow = np.zeros(ds.t1.shape[:2] + (2,), dtype=np.float32)
        try:
            corrections = self._compute_corrections(ds, tname, ds.t2, true_flow)
            for method, item in corrections.items():
                correction_rows.append({'dataset': ds.name, 'transform': tname, 'correction': method, **item['metadata']})
        except Exception as exc:
            self._record_status('evaluate', ds.name, f'transform:{tname}', None, 'failed', f'{exc}\n{traceback.format_exc()}')
            return

        for seed in self._bar(self.seeds, desc=f'evaluate clean no-harm {ds.name}', unit='seed', leave=False):
            try:
                _, _, test_mask = self._load_split(ds.name, seed)
                test_idx = balanced_indices(test_mask, ds.gt, max_per_class=None, seed=seed)
                eval_mask = np.zeros(ds.gt.shape, dtype=bool)
                eval_mask.reshape(-1)[test_idx] = True
            except Exception as exc:
                self._record_status('evaluate', ds.name, 'clean_no_harm_split', seed, 'failed', f'{exc}\n{traceback.format_exc()}')
                continue

            for baseline in self._bar(['GLAFormer', 'GETNET'], desc=f'clean no-harm deep {ds.name}/seed{seed}', unit='baseline', leave=False):
                bcfg = self.cfg.get('baselines', {}).get(baseline, {})
                if not bool(bcfg.get('enabled', True)):
                    continue
                checkpoint = self.ckpt_dir / ds.name / f'seed{seed}' / baseline / 'best.pt'
                if not checkpoint.exists():
                    self._record_status('evaluate', ds.name, baseline, seed, 'missing_checkpoint')
                    continue
                try:
                    prob, thr = self._predict_deep(baseline, ds, seed, ds.t2, test_idx, f'{tname}_clean_reference')
                    condition_rows.append(self._condition_row(ds.name, seed, baseline, tname, 'clean_reference', ds.gt, eval_mask, self._fill_map(ds.gt.shape, test_idx, prob), thr))
                    self._save_viz_prediction(ds.name, seed, baseline, tname, 'clean_reference', test_idx, prob, thr)
                    for method, item in self._bar(corrections.items(), desc=f'clean no-harm corrections {baseline} {ds.name}/seed{seed}', unit='method', leave=False):
                        prob, thr = self._predict_deep(baseline, ds, seed, item['corrected'], test_idx, f'{tname}_{method}')
                        condition_rows.append(self._condition_row(ds.name, seed, baseline, tname, method, ds.gt, eval_mask, self._fill_map(ds.gt.shape, test_idx, prob), thr))
                        self._save_viz_prediction(ds.name, seed, baseline, tname, method, test_idx, prob, thr)
                    self._record_status('evaluate', ds.name, f'{baseline}:clean_no_harm', seed, 'ok')
                except Exception as exc:
                    self._record_status('evaluate', ds.name, f'{baseline}:clean_no_harm', seed, 'failed', f'{exc}\n{traceback.format_exc()}')

            icfg = self.cfg.get('baselines', {}).get('IRMAD', {})
            if bool(icfg.get('enabled', True)):
                try:
                    irmad_conditions = [('clean_reference', ds.t2)] + [(m, x['corrected']) for m, x in corrections.items()]
                    for corr_name, cube in self._bar(irmad_conditions, desc=f'IRMAD clean no-harm {ds.name}/seed{seed}', unit='condition', leave=False):
                        score, pred, thr, meta = self._predict_irmad(ds, cube, f'{tname}_{corr_name}')
                        condition_rows.append({
                            'dataset': ds.name,
                            'seed': int(seed),
                            'baseline': 'IRMAD',
                            'transform': tname,
                            'correction': corr_name,
                            'threshold': float(thr),
                            'evaluation_fraction': float(eval_mask.mean()),
                            **binary_metrics(pred, ds.gt, mask=eval_mask, prob=score),
                        })
                    self._record_status('evaluate', ds.name, 'IRMAD:clean_no_harm', seed, 'ok')
                except Exception as exc:
                    self._record_status('evaluate', ds.name, 'IRMAD:clean_no_harm', seed, 'failed', f'{exc}\n{traceback.format_exc()}')

    def evaluate(self) -> None:
        condition_rows = []
        correction_rows = []
        for name in self._bar(self.datasets, desc='evaluate datasets', unit='dataset'):
            try:
                ds = self._dataset(name)
            except Exception as exc:
                self._record_status('evaluate', name, 'dataset', None, 'failed', f'{exc}\n{traceback.format_exc()}')
                continue
            self._evaluate_clean_noharm(ds, condition_rows, correction_rows)
            for raw_spec in self._bar(self.cfg.get('transforms', []), desc=f'evaluate transforms {name}', unit='transform', leave=False):
                spec = dict(raw_spec)
                tname = str(spec['name'])
                try:
                    trdir = self._transform_dir(name, tname)
                    moving = np.load(trdir / 'moving.npy').astype(np.float32)
                    true_flow = np.load(trdir / 'flow.npy').astype(np.float32)
                    oracle = np.load(trdir / 'oracle.npy').astype(np.float32)
                    corrections = self._compute_corrections(ds, tname, moving, true_flow)
                    for method, item in corrections.items():
                        correction_rows.append({'dataset': name, 'transform': tname, 'correction': method, **item['metadata']})
                except Exception as exc:
                    self._record_status('evaluate', name, f'transform:{tname}', None, 'failed', f'{exc}\n{traceback.format_exc()}')
                    continue

                for seed in self._bar(self.seeds, desc=f'evaluate seeds {name}/{tname}', unit='seed', leave=False):
                    try:
                        _, _, test_mask = self._load_split(name, seed)
                        test_idx = balanced_indices(test_mask, ds.gt, max_per_class=None, seed=seed)
                        eval_mask = np.zeros(ds.gt.shape, dtype=bool)
                        eval_mask.reshape(-1)[test_idx] = True
                    except Exception as exc:
                        self._record_status('evaluate', name, 'split', seed, 'failed', f'{exc}\n{traceback.format_exc()}')
                        continue

                    # Reuse the clean-trained detector and fixed test mask.
                    for baseline in self._bar(['GLAFormer', 'GETNET'], desc=f'deep baselines {name}/{tname}/seed{seed}', unit='baseline', leave=False):
                        bcfg = self.cfg.get('baselines', {}).get(baseline, {})
                        if not bool(bcfg.get('enabled', True)):
                            continue
                        checkpoint = self.ckpt_dir / name / f'seed{seed}' / baseline / 'best.pt'
                        if not checkpoint.exists():
                            self._record_status('evaluate', name, baseline, seed, 'missing_checkpoint')
                            continue
                        try:
                            prob, thr = self._predict_deep(baseline, ds, seed, ds.t2, test_idx, 'clean_reference')
                            condition_rows.append(self._condition_row(name, seed, baseline, tname, 'clean_reference', ds.gt, eval_mask, self._fill_map(ds.gt.shape, test_idx, prob), thr))
                            self._save_viz_prediction(name, seed, baseline, tname, 'clean_reference', test_idx, prob, thr)

                            prob, thr = self._predict_deep(baseline, ds, seed, moving, test_idx, f'{tname}_misaligned')
                            condition_rows.append(self._condition_row(name, seed, baseline, tname, 'misaligned', ds.gt, eval_mask, self._fill_map(ds.gt.shape, test_idx, prob), thr))
                            self._save_viz_prediction(name, seed, baseline, tname, 'misaligned', test_idx, prob, thr)

                            prob, thr = self._predict_deep(baseline, ds, seed, oracle, test_idx, f'{tname}_oracle')
                            condition_rows.append(self._condition_row(name, seed, baseline, tname, 'oracle', ds.gt, eval_mask, self._fill_map(ds.gt.shape, test_idx, prob), thr))
                            self._save_viz_prediction(name, seed, baseline, tname, 'oracle', test_idx, prob, thr)

                            for method, item in self._bar(corrections.items(), desc=f'eval corrections {baseline} {name}/{tname}/seed{seed}', unit='method', leave=False):
                                prob, thr = self._predict_deep(baseline, ds, seed, item['corrected'], test_idx, f'{tname}_{method}')
                                condition_rows.append(self._condition_row(name, seed, baseline, tname, method, ds.gt, eval_mask, self._fill_map(ds.gt.shape, test_idx, prob), thr))
                                self._save_viz_prediction(name, seed, baseline, tname, method, test_idx, prob, thr)
                            self._record_status('evaluate', name, baseline, seed, 'ok')
                        except Exception as exc:
                            self._record_status('evaluate', name, baseline, seed, 'failed', f'{exc}\n{traceback.format_exc()}')

                    # IR-MAD is evaluated on each seed-specific test mask.
                    icfg = self.cfg.get('baselines', {}).get('IRMAD', {})
                    if bool(icfg.get('enabled', True)):
                        try:
                            irmad_conditions = [('clean_reference', ds.t2), ('misaligned', moving), ('oracle', oracle)] + [(m, x['corrected']) for m, x in corrections.items()]
                            for corr_name, cube in self._bar(irmad_conditions, desc=f'IRMAD {name}/{tname}/seed{seed}', unit='condition', leave=False):
                                score, pred, thr, meta = self._predict_irmad(ds, cube, f'{tname}_{corr_name}')
                                condition_rows.append({
                                    'dataset': name,
                                    'seed': int(seed),
                                    'baseline': 'IRMAD',
                                    'transform': tname,
                                    'correction': corr_name,
                                    'threshold': float(thr),
                                    'evaluation_fraction': float(eval_mask.mean()),
                                    **binary_metrics(pred, ds.gt, mask=eval_mask, prob=score),
                                })
                            self._record_status('evaluate', name, 'IRMAD', seed, 'ok')
                        except Exception as exc:
                            self._record_status('evaluate', name, 'IRMAD', seed, 'failed', f'{exc}\n{traceback.format_exc()}')
        if condition_rows:
            pd.DataFrame(condition_rows).to_csv(self.metrics_dir / 'condition_metrics.csv', index=False)
        if correction_rows:
            pd.DataFrame(correction_rows).to_csv(self.metrics_dir / 'correction_metrics.csv', index=False)
        self._flush_status()

    def summarize(self) -> Path:
        return summarize_and_package(self.root, self.cfg)
