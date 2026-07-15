from __future__ import annotations
import json, zipfile
from pathlib import Path
from typing import Any
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from .utils import save_json, stable_hash_dict


def med(x):
    a = np.asarray(list(x), float)
    a = a[np.isfinite(a)]
    return float(np.median(a)) if len(a) else float('nan')


def finite_rate(values, predicate) -> float:
    a = np.asarray(list(values), float)
    a = a[np.isfinite(a)]
    return float(np.mean(predicate(a))) if len(a) else float('nan')


def finite_pair_rate(left, right, predicate) -> float:
    a = np.asarray(list(left), float)
    b = np.asarray(list(right), float)
    valid = np.isfinite(a) & np.isfinite(b)
    return float(np.mean(predicate(a[valid], b[valid]))) if np.any(valid) else float('nan')


def df_to_markdown_safe(df: pd.DataFrame) -> str:
    """Return a Markdown-compatible table without making tabulate mandatory."""
    if df is None or df.empty:
        return ''
    try:
        return df.to_markdown(index=False)
    except ImportError:
        return '```\n' + df.to_string(index=False) + '\n```'


def _gap_recovery(clean: float, degraded: float, corrected: float, min_gap: float) -> float:
    gap = clean - degraded
    return float('nan') if gap <= min_gap else float((corrected - degraded) / gap)


def gap_table(m: pd.DataFrame, min_gap: float = 1e-8) -> pd.DataFrame:
    rows = []
    if m.empty:
        return pd.DataFrame()
    # Clean-pair rows do not contain a misaligned reference.
    for key, g in m[m['transform'] != 'clean_no_harm'].groupby(['dataset', 'seed', 'baseline', 'transform']):
        d = {str(r.correction): r for _, r in g.iterrows()}
        if 'clean_reference' not in d or 'misaligned' not in d:
            continue
        clean = float(d['clean_reference'].F1)
        degraded = float(d['misaligned'].F1)
        r = dict(zip(['dataset', 'seed', 'baseline', 'transform'], key))
        r.update(clean_F1=clean, misaligned_F1=degraded, misalignment_drop=clean - degraded)
        for corr in ['oracle', 'pc1_translation', 'pca3_translation', 'scanline_unweighted', 'cmss_k1', 'cmss']:
            val = float(d[corr].F1) if corr in d else float('nan')
            r[f'{corr}_F1'] = val
            r[f'{corr}_recovery'] = _gap_recovery(clean, degraded, val, min_gap)
        rows.append(r)
    return pd.DataFrame(rows)


def clean_noharm_table(m: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if m.empty or 'transform' not in m.columns:
        return pd.DataFrame()
    x = m[m['transform'] == 'clean_no_harm']
    if x.empty:
        return pd.DataFrame()
    for key, g in x.groupby(['dataset', 'seed', 'baseline']):
        d = {str(r.correction): r for _, r in g.iterrows()}
        if 'clean_reference' not in d:
            continue
        clean = float(d['clean_reference'].F1)
        row = dict(zip(['dataset', 'seed', 'baseline'], key))
        row['clean_F1'] = clean
        for corr in ['pc1_translation', 'pca3_translation', 'scanline_unweighted', 'cmss_k1', 'cmss']:
            val = float(d[corr].F1) if corr in d else float('nan')
            row[f'{corr}_F1'] = val
            row[f'{corr}_delta_F1'] = val - clean if np.isfinite(val) else float('nan')
        rows.append(row)
    return pd.DataFrame(rows)


def _main_filter(gap: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    report_cfg = cfg.get('report', {})
    baselines = [str(x) for x in report_cfg.get('strong_baselines', ['GLAFormer', 'GETNET'])]
    transforms = [str(x) for x in report_cfg.get('main_transforms', [t.get('name') for t in cfg.get('transforms', [])])]
    x = gap.copy()
    if baselines:
        x = x[x['baseline'].isin(baselines)]
    if transforms:
        x = x[x['transform'].isin(transforms)]
    return x


def summarize_and_package(root: str | Path, cfg: dict[str, Any]) -> Path:
    root = Path(root)
    s = root / 'summary'
    s.mkdir(parents=True, exist_ok=True)
    mpath = root / 'metrics' / 'condition_metrics.csv'
    cpath = root / 'metrics' / 'correction_metrics.csv'
    m = pd.read_csv(mpath) if mpath.exists() and mpath.stat().st_size else pd.DataFrame()
    c = pd.read_csv(cpath) if cpath.exists() and cpath.stat().st_size else pd.DataFrame()
    report_cfg = cfg.get('report', {})
    min_gap = float(report_cfg.get('gap_min_denominator', 1e-8))

    gap = gap_table(m, min_gap=min_gap)
    gap.to_csv(s / 'gap_recovery.csv', index=False)
    main_gap = _main_filter(gap, cfg)
    main_gap.to_csv(s / 'main_gap_recovery.csv', index=False)

    noharm = clean_noharm_table(m)
    noharm.to_csv(s / 'clean_noharm.csv', index=False)

    if not m.empty:
        m.groupby(['dataset', 'baseline', 'transform', 'correction'], as_index=False).agg(
            F1_mean=('F1', 'mean'), F1_std=('F1', 'std'), IoU_mean=('IoU', 'mean'),
            BoundaryF1_mean=('BoundaryF1', 'mean'), Kappa_mean=('Kappa', 'mean')
        ).to_csv(s / 'scenario_summary.csv', index=False)
    if not c.empty:
        c.groupby(['dataset', 'transform', 'correction'], as_index=False).mean(numeric_only=True).to_csv(s / 'correction_summary.csv', index=False)
        if 'selector_choice' in c.columns:
            sel = c[c['correction'].astype(str) == 'cmss'].copy()
            if not sel.empty:
                sel.groupby(['dataset', 'transform', 'selector_choice', 'selector_reason'], dropna=False).size().reset_index(name='count').to_csv(s / 'cmss_selector_summary.csv', index=False)

    paper_rows = []
    if not main_gap.empty:
        for ds, g in main_gap.groupby('dataset'):
            paper_rows.append({
                'dataset': ds,
                'drop_median': med(g['misalignment_drop']),
                'oracle_recovery_median': med(g['oracle_recovery']),
                'cmss_recovery_median': med(g['cmss_recovery']),
                'cmss_k1_recovery_median': med(g['cmss_k1_recovery']),
                'scanline_recovery_median': med(g['scanline_unweighted_recovery']),
                'pca3_recovery_median': med(g['pca3_translation_recovery']),
                'cmss_positive_rate': finite_rate(g['cmss_recovery'], lambda a: a > 0),
                'cmss_gt_scanline_rate': finite_pair_rate(g['cmss_recovery'], g['scanline_unweighted_recovery'], lambda a, b: a > b),
                'cmss_gt_pca3_rate': finite_pair_rate(g['cmss_recovery'], g['pca3_translation_recovery'], lambda a, b: a > b),
            })
    pd.DataFrame(paper_rows).to_csv(s / 'paper_overview.csv', index=False)

    ab_rows = []
    if not main_gap.empty:
        for name, col in [
            ('PCA3 translation', 'pca3_translation_recovery'),
            ('Unweighted scanline', 'scanline_unweighted_recovery'),
            ('CMSS-K1 single component', 'cmss_k1_recovery'),
            ('CMSS full', 'cmss_recovery'),
            ('Oracle', 'oracle_recovery'),
        ]:
            ab_rows.append({'method': name, 'median_recovery': med(main_gap[col]), 'positive_rate': finite_rate(main_gap[col], lambda a: a > 0)})
    pd.DataFrame(ab_rows).to_csv(s / 'ablation_summary.csv', index=False)

    noharm_rows = []
    if not noharm.empty:
        for method in ['pc1_translation', 'pca3_translation', 'scanline_unweighted', 'cmss_k1', 'cmss']:
            col = f'{method}_delta_F1'
            if col in noharm.columns:
                noharm_rows.append({'method': method, 'median_delta_F1': med(noharm[col]), 'worst_delta_F1': float(np.nanmin(np.asarray(noharm[col], float)))})
    pd.DataFrame(noharm_rows).to_csv(s / 'clean_noharm_summary.csv', index=False)

    fig_dir = s / 'figures'
    fig_dir.mkdir(exist_ok=True)
    if paper_rows:
        df = pd.DataFrame(paper_rows)
        fig, ax = plt.subplots(figsize=(8.5, 4.2))
        x = np.arange(len(df))
        w = 0.15
        ax.bar(x - 2*w, df['drop_median'], width=w, label='Drop')
        ax.bar(x - 1*w, df['pca3_recovery_median'], width=w, label='PCA3-Trans')
        ax.bar(x + 0*w, df['scanline_recovery_median'], width=w, label='Scanline-Unw')
        ax.bar(x + 1*w, df['cmss_k1_recovery_median'], width=w, label='CMSS-K1')
        ax.bar(x + 2*w, df['cmss_recovery_median'], width=w, label='CMSS')
        ax.set_xticks(x)
        ax.set_xticklabels(df['dataset'])
        ax.set_ylabel('Median F1 drop / recovery')
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(fig_dir / 'cmss_recovery_overview.png', dpi=180)
        plt.close(fig)

    report = ['# CMSS experiment summary', '']
    if paper_rows:
        report += ['## Main protocol dataset-level medians', '', df_to_markdown_safe(pd.DataFrame(paper_rows)), '']
    if ab_rows:
        report += ['## Necessary ablations', '', df_to_markdown_safe(pd.DataFrame(ab_rows)), '']
    if noharm_rows:
        report += ['## Clean no-harm summary', '', df_to_markdown_safe(pd.DataFrame(noharm_rows)), '']
    selector_path = s / 'cmss_selector_summary.csv'
    if selector_path.exists():
        try:
            report += ['## CMSS safe-selector choices', '', df_to_markdown_safe(pd.read_csv(selector_path)), '']
        except Exception:
            pass
    report += [
        '## Interpretation',
        '',
        f'- Gap recovery is reported only when `clean_F1 - misaligned_F1 > {min_gap}`; weaker-degradation cases should be discussed with ΔF1 instead.',
        '- Main protocol uses the report.strong_baselines and report.main_transforms fields, excluding out-of-model affine stress cases.',
        '- `scanline_unweighted` isolates the scanline model; `cmss_k1` isolates the multi-component feature contribution; `cmss` is the full method.',
        '- `clean_noharm.csv` checks whether synchronization damages already registered image pairs.',
    ]
    (s / 'report.md').write_text('\n'.join(report), encoding='utf-8')
    save_json(s / 'summary_manifest.json', {
        'config_hash': stable_hash_dict(cfg),
        'condition_rows': len(m),
        'correction_rows': len(c),
        'gap_rows': len(gap),
        'main_gap_rows': len(main_gap),
        'clean_noharm_rows': len(noharm),
        'gap_min_denominator': min_gap,
    })
    out = root / 'feedback_bundle.zip'
    include = [root / 'metrics', root / 'summary', root / 'inventory', root / 'resolved_config.yaml']
    with zipfile.ZipFile(out, 'w', zipfile.ZIP_DEFLATED) as z:
        for base in include:
            if base.exists():
                if base.is_file():
                    z.write(base, str(base.relative_to(root)))
                else:
                    for p in base.rglob('*'):
                        if p.is_file():
                            z.write(p, str(p.relative_to(root)))
        z.writestr('manifest.json', json.dumps({'included': ['metrics', 'summary', 'inventory', 'resolved_config.yaml']}, indent=2))
    return out
