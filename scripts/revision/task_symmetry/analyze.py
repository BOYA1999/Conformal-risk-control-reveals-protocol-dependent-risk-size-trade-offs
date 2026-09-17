import hashlib
import json
import os
import platform
from pathlib import Path

import numpy as np
import pandas as pd

PACKAGE = Path(__file__).resolve().parents[3]
SOURCE = Path(os.environ.get('MOLXAI_SOURCE_DIR', PACKAGE / 'results/full_grid')).resolve()
OUT = Path(os.environ.get('MOLXAI_OUTPUT_DIR', '../molxai-check/task_symmetry')).resolve()
if OUT == SOURCE or OUT == PACKAGE or PACKAGE in OUT.parents:
    raise SystemExit('Use an output directory outside the package and input directory.')
OUT.mkdir(parents=True, exist_ok=False)
METRICS = ['risk', 'mean_atom_fraction', 'precision', 'iou', 'random_atom_fraction', 'saving_vs_random']


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    path = SOURCE / 'comparison_cells.csv'
    source_hash = digest(path)
    cells = pd.read_csv(path)
    assert len(cells) == 264 and cells.alpha.eq(.1).all()
    assert not cells.duplicated(['cell_id', 'method']).any()
    assert cells.groupby(['task', 'method']).size().eq(6).all()
    assert set(cells.method) == {'ig', 'gradinput', 'saliency', 'atom_occlusion'}
    assert set(cells.task) == {'B', 'P', 'X', 'indole', 'PAINS', 'rings-count', 'rings-max', 'benzene', 'logic7', 'logic8', 'logic10'}
    tasks = cells.groupby(['family', 'task', 'method'])[METRICS].mean().reset_index()
    reference = tasks[tasks.method == 'ig']
    task_diffs = tasks[tasks.method != 'ig'].merge(reference, on=['family', 'task'], suffixes=('', '_ig'), validate='many_to_one')
    for metric in METRICS:
        task_diffs['delta_' + metric] = task_diffs[metric] - task_diffs[metric + '_ig']
    summaries, differences = [], []
    direct_error = 0.0
    for panel, selected in [('eleven_tasks', cells), ('eight_tasks', cells[~cells.task.isin(['B', 'P', 'X'])])]:
        assert selected.groupby('method').size().eq(66 if panel == 'eleven_tasks' else 48).all()
        for family, data in [('all', selected)] + list(selected.groupby('family')):
            task_means = data.groupby(['task', 'method'])[METRICS].mean().reset_index()
            for method, subset in data.groupby('method'):
                macro = task_means[task_means.method == method][METRICS].mean()
                direct = np.array([sum(float(row[metric]) for row in subset.to_dict('records')) / len(subset) for metric in METRICS])
                direct_error = max(direct_error, float(np.max(np.abs(direct - macro.to_numpy()))))
                summaries.append({'panel': panel, 'family': family, 'method': method, 'tasks': subset.task.nunique(), 'cells': len(subset), **macro.to_dict(), 'risk_pass_cells': int(subset.risk.le(.1).sum()), 'size_pass_cells': int(subset.mean_atom_fraction.lt(.8).sum()), 'joint_pass_cells': int((subset.risk.le(.1) & subset.mean_atom_fraction.lt(.8)).sum()), 'origin': '|'.join(sorted(subset.origin.unique()))})
            ref = task_means[task_means.method == 'ig'].set_index('task')[METRICS].sort_index()
            draws = np.random.default_rng(20260913).integers(0, len(ref), size=(5000, len(ref)))
            for method in ['atom_occlusion', 'saliency', 'gradinput']:
                delta = task_means[task_means.method == method].set_index('task')[METRICS].reindex(ref.index) - ref
                row = {'panel': panel, 'family': family, 'method': method, 'reference': 'ig', 'tasks': len(ref), 'bootstrap_replicates': 5000, 'smaller_tasks': int(delta.mean_atom_fraction.lt(0).sum()), 'larger_tasks': int(delta.mean_atom_fraction.gt(0).sum()), 'equal_size_tasks': int(delta.mean_atom_fraction.eq(0).sum())}
                for metric in METRICS:
                    values = delta[metric].to_numpy()
                    low, high = np.quantile(values[draws].mean(1), [.025, .975])
                    row.update({f'delta_{metric}': float(values.mean()), f'delta_{metric}_ci_low': float(low), f'delta_{metric}_ci_high': float(high)})
                differences.append(row)
    summary, differences = pd.DataFrame(summaries), pd.DataFrame(differences)
    primary = summary[(summary.panel == 'eleven_tasks') & (summary.family == 'all')]
    original = pd.read_csv(SOURCE / 'method_summary.csv')
    join = primary.merge(original, on='method', suffixes=('', '_original'), validate='one_to_one')
    errors = {metric: float(np.max(np.abs(join[metric] - join['macro_task_' + metric]))) for metric in METRICS}
    assert max(errors.values()) < 1e-12 and direct_error < 1e-12
    assert (join.risk_pass_cells == join.risk_pass_cells_original).all()
    assert (join.size_pass_cells == join.efficiency_pass_cells).all()
    assert (join.joint_pass_cells == join.joint_pass_cells_original).all()
    for name, data in [('summary', summary), ('paired_task_differences', task_diffs), ('paired_summary', differences), ('task_means', tasks)]:
        data.to_csv(OUT / f'{name}.csv', index=False)
    assert digest(path) == source_hash
    report = {'status': 'PASS', 'input_rows': 264, 'methods': 4, 'primary_cells_per_method': 66, 'excluded_panel_cells_per_method': 48, 'summary_rows': len(summary), 'task_mean_rows': len(tasks), 'task_difference_rows': len(task_diffs), 'paired_summary_rows': len(differences), 'original_summary_max_errors': errors, 'direct_cell_mean_max_error': direct_error, 'source_sha256': source_hash, 'original_summary_sha256': digest(SOURCE / 'method_summary.csv'), 'script_sha256': digest(Path(__file__)), 'python': platform.python_version(), 'numpy': np.__version__, 'pandas': pd.__version__, 'bootstrap': '5000 paired task-cluster draws, seed20260913; six models/seeds averaged first; descriptive fixed-panel sensitivity, not benchmark-family population inference'}
    (OUT / 'validation.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(summary[summary.family == 'all'][['panel', 'method', 'risk', 'mean_atom_fraction', 'iou', 'size_pass_cells', 'joint_pass_cells']].to_string(index=False))
    print(differences[(differences.panel == 'eight_tasks') & (differences.method == 'atom_occlusion')][['family', 'smaller_tasks', 'larger_tasks', 'delta_mean_atom_fraction', 'delta_mean_atom_fraction_ci_low', 'delta_mean_atom_fraction_ci_high']].to_string(index=False))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
