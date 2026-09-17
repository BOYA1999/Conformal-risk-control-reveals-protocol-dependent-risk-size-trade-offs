import hashlib
import json
import os
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd

PACKAGE = Path(__file__).resolve().parents[3]
WORK = Path(os.environ.get('MOLXAI_WORK_ROOT', '../molxai-work')).resolve()
SOURCE = WORK / 'independent_reference/results'
OUT = Path(os.environ.get('MOLXAI_OUTPUT_DIR', '../molxai-check/liver_calibration')).resolve()
if OUT == WORK or WORK in OUT.parents or OUT == PACKAGE or PACKAGE in OUT.parents:
    raise SystemExit('Use a new output directory outside the package and input workspace.')
OUT.mkdir(parents=True, exist_ok=False)
GRID = np.linspace(0, 1, 101)
METRICS = ['risk', 'mean_atom_fraction', 'precision', 'iou', 'full_molecule_fraction']
POLICIES = ['index_tiebreak_v1', 'include_all_exact_ties_v2']
KEYS = ['cell_id', 'target', 'method', 'policy', 'alpha']


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path):
    with np.load(path, allow_pickle=False) as data:
        rows = [(data['scores'][a:b].astype(float), data['masks'][a:b].astype(bool)) for a, b in zip(data['ptr'][:-1], data['ptr'][1:])]
        return data['source_ids'].copy(), rows


def curves(rows, policy):
    output = []
    for scores, mask in rows:
        assert np.isfinite(scores).all() and mask.any() and len(scores) == len(mask)
        order = np.lexsort((np.arange(len(scores)), -scores))
        counts = np.ceil(GRID * len(scores)).astype(int)
        if policy == POLICIES[1]:
            counts[1:] = np.searchsorted(-scores[order], -scores[order][counts[1:] - 1], side='right')
        hits = np.r_[0, np.cumsum(mask[order])][counts]
        output.append(np.stack([1 - hits / mask.sum(), counts / len(scores), np.divide(hits, counts, out=np.zeros(101), where=counts > 0), hits / (counts + mask.sum() - hits), (counts == len(scores)).astype(float)], axis=-1))
    values = np.asarray(output)
    assert np.all(np.diff(values[:, :, 0], axis=1) <= 1e-15)
    assert np.all(values[:, 0, 0] == 1) and np.all(values[:, 100, 0] == 0)
    return values


def direct(scores, mask, q, policy):
    k = int(np.ceil(float(GRID[q]) * len(scores)))
    order = sorted(range(len(scores)), key=lambda i: (-float(scores[i]), i))
    chosen = set(order[:k])
    if k and policy == POLICIES[1]:
        cutoff = float(scores[order[k - 1]])
        chosen = {i for i, value in enumerate(scores) if float(value) >= cutoff}
    reference = set(np.flatnonzero(mask).tolist())
    hits = len(chosen & reference)
    return np.array([1 - hits / len(reference), len(chosen) / len(scores), hits / len(chosen) if chosen else 0, hits / len(chosen | reference), float(len(chosen) == len(scores))])


def interval(values, draws):
    return float(values.mean()), *np.quantile(values[draws].mean(1), [.025, .975]).tolist()


def main():
    started = time.time()
    original_path = SOURCE / 'cell_metrics.csv'
    original = pd.read_csv(original_path)
    assert len(original) == 288 and not original.duplicated(KEYS).any()
    files = sorted(SOURCE.glob('*_calibration_scores.npz'))
    assert len(files) == 48
    hashes = {'cell_metrics.csv': digest(original_path), 'summary.csv': digest(SOURCE / 'summary.csv')}
    cell_rows, molecule_rows = [], []
    common_ids, common_reference = {}, {}
    direct_checks, direct_error = 0, 0.0
    for path in files:
        cell_id = path.name.split('_observed_class_')[0] if '_observed_class_' in path.name else path.name.split('_hepatotoxic_class_')[0]
        target = 'observed_class' if '_observed_class_' in path.name else 'hepatotoxic_class'
        method = path.name.removeprefix(f'{cell_id}_{target}_').removesuffix('_calibration_scores.npz')
        identity = {'cell_id': cell_id, 'target': target, 'method': method}
        loaded = {}
        for split in ['calibration', 'test']:
            current = SOURCE / f'{cell_id}_{target}_{method}_{split}_scores.npz'
            hashes[current.name] = digest(current)
            ids, rows = load(current)
            assert len(ids) == (33 if split == 'calibration' else 31) and len(set(ids)) == len(ids)
            if split in common_ids:
                assert np.array_equal(ids, common_ids[split])
                assert all(np.array_equal(row[1], ref) for row, ref in zip(rows, common_reference[split]))
            common_ids[split], common_reference[split] = ids, [row[1] for row in rows]
            loaded[split] = rows
        assert not set(common_ids['calibration']) & set(common_ids['test'])
        for policy in POLICIES:
            cal, test = curves(loaded['calibration'], policy), curves(loaded['test'], policy)
            empirical = cal[:, :, 0].mean(0)
            corrected = (cal[:, :, 0].sum(0) + 1) / 34
            check_q = {0, 100}
            for alpha in [.05, .10, .20]:
                selected = {}
                for rule, criterion in [('corrected', corrected), ('naive', empirical)]:
                    feasible = np.flatnonzero(criterion <= alpha)
                    q = int(feasible[0]) if len(feasible) else 100
                    selected[rule] = q
                    check_q.update([q, max(0, q - 1)])
                    row = {**identity, 'policy': policy, 'alpha': alpha, 'rule': rule, 'q': q, 'lambda': float(GRID[q]), 'empirical_calibration_risk': float(empirical[q]), 'corrected_calibration_risk': float(corrected[q]), 'fallback_full': not bool(len(feasible)), 'n_calibration': 33, 'n_test': 31}
                    row.update(zip(METRICS, test[:, q, :].mean(0)))
                    cell_rows.append(row)
                    for source_id, values in zip(common_ids['test'], test[:, q, :]):
                        molecule_rows.append({**identity, 'policy': policy, 'alpha': alpha, 'rule': rule, 'source_index': int(source_id), **dict(zip(METRICS, values))})
                assert selected['corrected'] >= selected['naive']
            for split, values in [('calibration', cal), ('test', test)]:
                for i, (scores, mask) in enumerate(loaded[split]):
                    for q in check_q:
                        error = float(np.max(np.abs(direct(scores, mask, q, policy) - values[i, q])))
                        direct_error = max(direct_error, error)
                        direct_checks += 1
    assert direct_error < 1e-12
    cells, molecules = pd.DataFrame(cell_rows), pd.DataFrame(molecule_rows)
    joined = original.merge(cells[cells.rule == 'corrected'], on=KEYS, suffixes=('_old', '_new'), validate='one_to_one')
    errors = {metric: float(np.max(np.abs(joined[metric + '_old'] - joined[metric + '_new']))) for metric in METRICS[:-1] + ['corrected_calibration_risk']}
    errors['selected_fraction'] = float(np.max(np.abs(joined.selected_fraction - joined['lambda'])))
    assert max(errors.values()) < 1e-12 and len(joined) == 288
    paired = cells[cells.rule == 'corrected'].merge(cells[cells.rule == 'naive'], on=KEYS, suffixes=('_corrected', '_naive'), validate='one_to_one')
    paired['same_point'] = paired.q_corrected == paired.q_naive
    for key in ['q', 'lambda'] + METRICS:
        paired['delta_' + key] = paired[key + '_corrected'] - paired[key + '_naive']
    unique_ids = sorted(common_ids['test'].tolist())
    draws = np.random.default_rng(20260912).integers(0, 31, size=(2000, 31))
    summaries = []
    group_keys = ['target', 'policy', 'method', 'alpha']
    for keys, group in molecules.groupby(group_keys):
        identity = dict(zip(group_keys, keys))
        points = paired.loc[(paired[group_keys] == pd.Series(identity)).all(axis=1)]
        assert len(points) == 6
        row = {**identity, 'models': 6, 'n_calibration': 33, 'n_test': 31, 'bootstrap_replicates': 2000, 'same_point_cells': int(points.same_point.sum()), 'same_point_fraction': float(points.same_point.mean())}
        means = {rule: frame.groupby('source_index')[METRICS].mean().reindex(unique_ids) for rule, frame in group.groupby('rule')}
        for metric in METRICS:
            for label, values in [('corrected', means['corrected'][metric].to_numpy()), ('naive', means['naive'][metric].to_numpy()), ('delta', (means['corrected'][metric] - means['naive'][metric]).to_numpy())]:
                for suffix, value in zip(['', '_ci_low', '_ci_high'], interval(values, draws)):
                    row[f'{label}_{metric}{suffix}'] = value
        for key in ['q', 'lambda']:
            for label in ['corrected', 'naive']:
                row[f'mean_{key}_{label}'] = float(points[f'{key}_{label}'].mean())
            row['mean_delta_' + key] = float(points['delta_' + key].mean())
        summaries.append(row)
    summary = pd.DataFrame(summaries)
    old_summary = pd.read_csv(SOURCE / 'summary.csv')
    sj = old_summary.merge(summary[summary.alpha == .1], on=['target', 'policy', 'method', 'alpha'], validate='one_to_one')
    summary_error = max(float(np.max(np.abs(sj[metric + suffix] - sj['corrected_' + metric + suffix]))) for metric in METRICS[:-1] for suffix in ['', '_ci_low', '_ci_high'])
    assert summary_error < 1e-12
    for name, data in [('cell_metrics', cells), ('molecule_metrics', molecules), ('paired_cells', paired), ('summary', summary)]:
        data.to_csv(OUT / f'{name}.csv', index=False)
    primary = summary[(summary.policy == POLICIES[0]) & (summary.alpha == .1)]
    primary.to_csv(OUT / 'primary_table.csv', index=False)
    report = {'status': 'PASS', 'score_combinations': 48, 'score_archives': 96, 'policy_alpha_combinations': 288, 'selector_rows': len(cells), 'molecule_rows': len(molecules), 'original_corrected_rows_reproduced': len(joined), 'original_metric_max_errors': errors, 'original_summary_interval_max_error': summary_error, 'direct_set_checks': direct_checks, 'direct_set_max_error': direct_error, 'same_point_primary_cells': int(paired[(paired.policy == POLICIES[0]) & (paired.alpha == .1)].same_point.sum()), 'primary_combinations': 48, 'runtime_seconds': time.time() - started, 'python': platform.python_version(), 'numpy': np.__version__, 'pandas': pd.__version__, 'bootstrap': {'seed': 20260912, 'replicates': 2000, 'unit': '31 shared test molecules', 'conditioning': 'fixed calibration and predictors', 'paired_delta': 'corrected minus naive before resampling'}, 'script_sha256': digest(Path(__file__)), 'input_sha256': hashes}
    assert all(digest(SOURCE / name) == value for name, value in hashes.items())
    (OUT / 'validation.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(primary[['target', 'method', 'same_point_cells', 'corrected_risk', 'naive_risk', 'corrected_mean_atom_fraction', 'naive_mean_atom_fraction', 'delta_mean_atom_fraction', 'delta_mean_atom_fraction_ci_low', 'delta_mean_atom_fraction_ci_high']].to_string(index=False))
    print(json.dumps({k: v for k, v in report.items() if k != 'input_sha256'}, indent=2))


if __name__ == '__main__':
    main()
