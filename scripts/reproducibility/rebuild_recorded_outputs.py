import argparse
import importlib.util
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PACKAGE = Path(__file__).resolve().parents[2]


def liver(work, output, replay):
    source = work / 'independent_reference/results'
    expected = pd.read_csv(source / 'cell_metrics.csv')
    rows, molecules = [], []
    keys = ['cell_id', 'target', 'method', 'policy', 'alpha']
    metrics = ['risk', 'mean_atom_fraction', 'precision', 'iou', 'tie_inflation']
    maximum = 0.0
    for (cell, target, method), group in expected.groupby(keys[:3]):
        parts = []
        for split in ['calibration', 'test']:
            with np.load(source / f'{cell}_{target}_{method}_{split}_scores.npz', allow_pickle=False) as cache:
                ptr = cache['ptr']
                scores = [cache['scores'][a:b].copy() for a, b in zip(ptr[:-1], ptr[1:])]
                masks = [cache['masks'][a:b].astype(bool) for a, b in zip(ptr[:-1], ptr[1:])]
                ids = cache['source_ids'].copy()
            assert len(scores) == (33 if split == 'calibration' else 31)
            parts.extend([scores, masks])
        for policy in replay.POLICIES:
            actual = replay.replay(*parts, policy, [.05, .1, .2])
            for alpha, result in actual.items():
                wanted = group[(group.policy == policy) & (group.alpha == float(alpha))].iloc[0]
                changes = [abs(result[key] - wanted[key]) for key in metrics]
                changes += [abs(result['fraction'] - wanted.selected_fraction), abs(result['corrected_risk'] - wanted.corrected_calibration_risk)]
                maximum = max(maximum, *changes)
                assert max(changes) < 1e-10, (cell, target, method, policy, alpha)
                rows.append(dict(cell_id=cell, target=target, method=method, policy=policy, **result))
            fraction = actual['0.10']['fraction']
            for identifier, scores, mask in zip(ids, parts[2], parts[3]):
                value = replay.direct_metrics([scores], [mask], fraction, policy == replay.POLICIES[1])
                value['equal_size_random_risk'] = 1 - value['mean_atom_fraction']
                value['risk_minus_random'] = value['risk'] - value['equal_size_random_risk']
                molecules.append(dict(cell_id=cell, target=target, method=method, policy=policy, source_index=int(identifier), selected_fraction=fraction, **value))
    assert len(rows) == 288 and len(molecules) == 2976
    frame = pd.DataFrame(molecules)
    ids = sorted(frame.source_index.unique())
    draws = np.random.default_rng(20260912).integers(0, 31, size=(2000, 31))
    summaries = []
    for group_keys, group in frame.groupby(['target', 'policy', 'method']):
        result = dict(zip(['target', 'policy', 'method'], group_keys))
        result.update(alpha=.1, models=6, test_molecules=31, calibration_molecules=33, bootstrap_replicates=2000)
        means = group.groupby('source_index')[metrics + ['equal_size_random_risk', 'risk_minus_random']].mean().reindex(ids)
        for key in means:
            values = means[key].to_numpy()
            result[key] = values.mean()
            result[key + '_ci_low'], result[key + '_ci_high'] = np.quantile(values[draws].mean(1), [.025, .975])
        result.update(risk_pass_cells=int((group.groupby('cell_id').risk.mean() <= .1).sum()), efficiency_pass_cells=int((group.groupby('cell_id').mean_atom_fraction.mean() < .8).sum()), mean_selected_fraction=group.groupby('cell_id').selected_fraction.first().mean())
        summaries.append(result)
    actual = pd.DataFrame(summaries).sort_values(['target', 'policy', 'method']).reset_index(drop=True)
    wanted = pd.read_csv(source / 'summary.csv').sort_values(['target', 'policy', 'method']).reset_index(drop=True)
    numeric = wanted.select_dtypes(include='number').columns
    delta = np.max(np.abs(actual[numeric].to_numpy() - wanted[numeric].to_numpy()))
    assert delta < 1e-10, delta
    target = output / 'independent_reference'
    target.mkdir()
    pd.DataFrame(rows).to_csv(target / 'cell_metrics_reconstructed.csv', index=False)
    actual.to_csv(target / 'summary_reconstructed.csv', index=False)
    return dict(cells=6, metric_rows=288, molecule_records_reconstructed=2976, test_molecules=31, maximum_metric_difference=maximum, maximum_summary_difference=float(delta), bootstrap='2000 shared molecule draws; seed20260912; conditional on fixed calibration and checkpoints')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--work-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    work, output = args.work_root.resolve(), args.output.resolve()
    if output.exists() or output == work or work in output.parents or PACKAGE == output or PACKAGE in output.parents:
        raise SystemExit('Use a new output directory outside the input workspace and code package.')
    output.mkdir(parents=True)
    module_path = PACKAGE / 'scripts/full_grid/validate_and_compare.py'
    spec = importlib.util.spec_from_file_location('recorded_replay', module_path)
    replay = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(replay)
    target = output / 'full_grid'
    for part in ['cells', 'scores']:
        shutil.copytree(work / 'full_grid' / part, target / part)
    replay.HERE = target
    replay.ORIGINAL = work / 'artifacts/experiment/gradient_grid_main/cells'
    replay.SUBSET = work / 'experiments/established_subset/cells'
    replay.CODE = work / 'full_grid/recorded_source'
    replay.CONTRACT = replay.CODE / 'run_contract.json'
    sys.argv = [str(module_path)]
    replay.main()
    p20 = json.loads((target / 'validation.json').read_text())
    p21 = liver(work, output, replay)
    record = {'status': 'PASS_RECORDED_SCORE_AND_STATISTIC_RECONSTRUCTION', 'p20_cells': p20['cells'], 'p20_maximum_metric_difference': p20['maximum_metric_replay_difference'], 'p21': p21, 'boundary': 'Frozen score/statistic reconstruction; no original-source atom mapping validation, new attribution, model training, new scientific validation, or public release.'}
    (output / 'reconstruction_report.json').write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(record))


if __name__ == '__main__':
    main()
