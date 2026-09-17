import argparse
from pathlib import Path

import pandas as pd

PACKAGE = Path(__file__).resolve().parents[3]
parser = argparse.ArgumentParser(description='Reconstruct Figure 3 task-macro numerical sources from P22 aggregate cells; no rendering or raw scores required.')
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
out = args.output.resolve()
assert out != PACKAGE and PACKAGE not in out.parents, 'Outputs must be outside the public package.'
out.mkdir(parents=True, exist_ok=False)
data = PACKAGE / 'results/grid_sensitivity'
curves = pd.read_csv(data / 'curves.csv')
points = pd.read_csv(data / 'operating_points.csv')
curves = curves[(curves.grid == 'machine') & (curves.split == 'test')]
points = points[points.grid == 'machine']
metrics = ['risk', 'mean_atom_fraction', 'precision', 'iou']
for kind, frame, axis, expected in [('curve', curves, 'q', 1212), ('point', points, 'alpha', 36)]:
    pieces = []
    for scope in ['all', 'bxaic', 'google']:
        selected = frame if scope == 'all' else frame[frame.family == scope]
        grouped = selected.groupby(['method', 'score_origin', 'family', 'task', axis], as_index=False)[metrics].mean().groupby(['method', 'score_origin', axis], as_index=False)[metrics].mean()
        grouped['scope'] = scope
        pieces.append(grouped)
    result = pd.concat(pieces, ignore_index=True)
    assert len(result) == expected
    name = f'fig3_{kind}_source.csv'
    result.to_csv(out / name, index=False)
    assert (out / name).read_bytes() == (data / 'figure_sources' / name).read_bytes(), name
print('PASS: both Figure 3 numerical source CSVs are byte identical.')
