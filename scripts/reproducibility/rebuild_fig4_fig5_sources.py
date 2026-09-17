import argparse
from pathlib import Path

import pandas as pd

PACKAGE = Path(__file__).resolve().parents[2]
parser = argparse.ArgumentParser(description='Rebuild Figures 4 and 5 numerical source tables from structure-free aggregates; no new attribution or image rendering.')
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
out = args.output.resolve()
assert out != PACKAGE and PACKAGE not in out.parents
out.mkdir(parents=True, exist_ok=False)
results = PACKAGE / 'results'
cells = pd.read_csv(results / 'full_grid/comparison_cells.csv')
assert len(cells) == 264 and cells.alpha.eq(.1).all()
tasks = cells.groupby(['family','task','method'])['mean_atom_fraction'].mean().reset_index()
tasks['provenance'] = tasks.method.map({'ig':'P01 historical primary','gradinput':'P01 historical primary','atom_occlusion':'P20 complete partition','saliency':'P20 complete partition'})
random = cells.groupby(['family','task'])['random_atom_fraction'].mean().reset_index().rename(columns={'random_atom_fraction':'mean_atom_fraction'})
random['method'], random['provenance'] = 'random', 'P02 original index policy'
oracle = pd.read_csv(results / 'original_audits/reviewer_oracle_crc/task_oracle_all_alpha.csv')
oracle = oracle[oracle.alpha == .1][['family','task','oracle_test_mean_atom_fraction']].rename(columns={'oracle_test_mean_atom_fraction':'mean_atom_fraction'})
assert len(oracle) == 11
oracle['method'], oracle['provenance'] = 'oracle', 'P02 original index policy'
task_data = pd.concat([tasks,random,oracle], ignore_index=True)
assert len(task_data) == 66 and not task_data.duplicated(['family','task','method']).any()
task_data.to_csv(out / 'fig4_task_source.csv', index=False)
pairs = pd.read_csv(results / 'task_symmetry/paired_summary.csv')
pairs = pairs[(pairs.family == 'all') & (pairs.method == 'atom_occlusion')].copy()
assert len(pairs) == 2
pairs.to_csv(out / 'fig4_interval_source.csv', index=False)
protocol = results / 'protocol_figure'
comparisons = pd.read_csv(protocol / 'inputs/target_comparisons.csv')
comparisons = comparisons[((comparisons['first'] == 'raw__zero__20') & (comparisons['second'] == 'raw__zero__200')) | ((comparisons['first'] == 'raw__zero__200') & (comparisons['second'] == 'margin__zero__200'))].copy()
assert len(comparisons) == 22 and comparisons.groupby('comparison').size().eq(11).all()
comparisons['provenance'] = 'P15/P16 same-run zero-baseline IG; exact ties; 11 seed42 GIN cells'
comparisons.to_csv(out / 'fig5_target_source.csv', index=False)
family = pd.read_csv(protocol / 'inputs/set_family_task_macro.csv')
family = family[family.calibration_loss == 'union'].copy()
assert len(family) == 4
family['oracle_retained'] = family.mean_atom_fraction - family.oracle_excess
family['provenance'] = 'P12/P13 cached-score set-family comparison; 11 seed42 GIN cells'
family.to_csv(out / 'fig5_set_family_source.csv', index=False)
witness = pd.read_csv(protocol / 'inputs/witness_cross_loss_family_macro.csv')
witness = witness[(witness.family == 'google') & (witness.set_family == 'global_index')].copy()
assert len(witness) == 4
witness['provenance'] = 'P14 legal-witness comparison; four Graph Attribution tasks'
witness.to_csv(out / 'fig5_reference_source.csv', index=False)
for path in sorted(out.glob('*.csv')):
    expected = results / ('task_symmetry' if path.name.startswith('fig4') else 'protocol_figure') / 'figure_sources' / path.name
    assert path.read_bytes() == expected.read_bytes(), path.name
print('PASS: all five Figure 4/5 numerical source CSVs are byte identical.')
