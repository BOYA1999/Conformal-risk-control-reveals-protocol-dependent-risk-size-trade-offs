import csv
import itertools
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
METRICS = ['risk', 'mean_atom_fraction', 'precision', 'iou', 'tie_inflation', 'equal_size_random_risk']


def read(folder, name):
    with (ROOT / 'results' / folder / name).open(encoding='utf-8') as stream:
        return list(csv.DictReader(stream))


def close(left, right):
    assert math.isfinite(float(left)) and math.isclose(float(left), float(right), abs_tol=1e-10, rel_tol=1e-10), (left, right)


def liver():
    cells = read('independent_reference', 'cell_metrics.csv')
    summary = read('independent_reference', 'summary.csv')
    strata = read('independent_reference', 'class_strata.csv')
    predictors = read('independent_reference', 'predictor_model_table.csv')
    assert [len(cells), len(summary), len(strata), len(predictors)] == [288, 16, 48, 6]
    ids = {f'liver__{model}__seed{seed}' for model, seed in itertools.product(['gin', 'gcn'], [42, 123, 2026])}
    assert {r['cell_id'] for r in predictors} == ids
    expected = set(itertools.product(ids, ['observed_class', 'hepatotoxic_class'], ['gradinput', 'ig', 'saliency', 'atom_occlusion'], ['index_tiebreak_v1', 'include_all_exact_ties_v2'], [.05, .1, .2]))
    assert {(r['cell_id'], r['target'], r['method'], r['policy'], float(r['alpha'])) for r in cells} == expected
    for row in cells:
        assert int(row['n_calibration']) == 33 and int(row['n_test']) == 31
        close(float(row['mean_atom_fraction']) + float(row['equal_size_random_risk']), 1)
    for row in summary:
        group = [r for r in cells if float(r['alpha']) == .1 and all(r[k] == row[k] for k in ['target', 'method', 'policy'])]
        subgroup = [r for r in strata if all(r[k] == row[k] for k in ['target', 'method', 'policy'])]
        assert len(group) == 6 and sorted(int(r['test_molecules']) for r in subgroup) == [4, 12, 15]
        for metric in METRICS:
            close(row[metric], mean(float(r[metric]) for r in group))
            close(row[metric], sum(float(r[metric]) * int(r['test_molecules']) for r in subgroup) / 31)
            assert float(row[metric + '_ci_low']) <= float(row[metric + '_ci_high'])
        close(row['risk_minus_random'], float(row['risk']) - float(row['equal_size_random_risk']))
        close(row['mean_selected_fraction'], mean(float(r['selected_fraction']) for r in group))
        assert int(row['risk_pass_cells']) == sum(float(r['risk']) <= .1 for r in group)
        assert int(row['efficiency_pass_cells']) == sum(float(r['mean_atom_fraction']) < .8 for r in group)
    audit = json.loads((ROOT / 'results/independent_reference/admission_summary.json').read_text())
    assert [audit['source_rows'], audit['admitted_rows'], audit['excluded_rows_total'], audit['external_overlap_rows']] == [587, 552, 35, 30]
    validation = json.loads((ROOT / 'results/independent_reference/validation_summary.json').read_text())
    assert validation['status'] == 'PASS' and validation['all_metric_rows_replayed'] == 288
    print('PASS: independent-reference aggregate means, strata and configuration coverage; bootstrap bounds are recorded local-replay evidence, not recomputed from aggregates')


def full_grid():
    rows = read('full_grid', 'comparison_cells.csv')
    summary = read('full_grid', 'method_summary.csv')
    assert len(rows) == 264 and len(summary) == 4
    expected = {f'{family}__{task}__{model}__seed{seed}' for family, tasks in [('bxaic', ['B', 'P', 'X', 'indole', 'PAINS', 'rings-count', 'rings-max']), ('google', ['benzene', 'logic7', 'logic8', 'logic10'])] for task, model, seed in itertools.product(tasks, ['gin', 'gcn'], [42, 123, 2026])}
    for row in summary:
        group = [r for r in rows if r['method'] == row['method']]
        assert len(group) == int(row['cells']) == 66 and {r['cell_id'] for r in group} == expected
        tasks = defaultdict(list)
        for r in group:
            tasks[r['family'], r['task']].append(r)
        assert len(tasks) == 11 and all(len(v) == 6 for v in tasks.values())
        for metric in ['risk', 'mean_atom_fraction', 'precision', 'iou', 'random_atom_fraction', 'saving_vs_random']:
            close(row['macro_task_' + metric], mean(mean(float(r[metric]) for r in values) for values in tasks.values()))
        assert int(row['risk_pass_cells']) == sum(float(r['risk']) <= .1 for r in group)
        assert int(row['efficiency_pass_cells']) == sum(float(r['mean_atom_fraction']) < .8 for r in group)
        assert int(row['joint_pass_cells']) == sum(float(r['risk']) <= .1 and float(r['mean_atom_fraction']) < .8 for r in group)
    for filename, count in [('historical_subset_cells.csv', 264), ('same_run_subset_cells.csv', 132), ('tie_sensitivity_cells.csv', 132), ('fixed_budget_cells.csv', 528), ('order_reversals.csv', 18), ('policy_order_comparisons.csv', 5), ('paired_comparisons.csv', 44)]:
        assert len(read('full_grid', filename)) == count, filename
    seal = json.loads((ROOT / 'results/full_grid/seal.json').read_text())
    assert seal['status'] == 'PASS' and seal['exact_expected_cells'] == seal['exact_expected_caches'] == 66
    print('PASS: four-method 66-cell grid coverage, macro-task means, pass counts and recorded full-run seal')


if __name__ == '__main__':
    liver()
    full_grid()
