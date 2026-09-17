import os
import runpy
import sys
from pathlib import Path

import torch


PACKAGE = Path(__file__).resolve().parents[1]
WORK = Path(os.environ.get('MOLXAI_WORK_ROOT', '../molxai-work')).resolve()
if WORK == PACKAGE or PACKAGE in WORK.parents:
    raise SystemExit('Set MOLXAI_WORK_ROOT to a directory outside this repository.')
sys.path.insert(0, str(PACKAGE / 'src'))
sys.argv = [str(PACKAGE / 'src/run_gradient_grid.py'),
    '--bxaic-csv', str(WORK / 'data/raw/bxaic/data.csv'),
    '--bxaic-sdf', str(WORK / 'data/raw/bxaic/explanations.sdf'),
    '--google-root', str(WORK / 'reference/graph-attribution/data'),
    '--contract', str(PACKAGE / 'contracts/experiment_contract.md'),
    '--out-dir', str(WORK / 'artifacts/experiment/gradient_grid_main'),
    '--surface', 'final', '--confirm-final', 'FROZEN_20260809_V0_1',
] + sys.argv[1:]
runpy.run_path(str(PACKAGE / 'src/run_gradient_grid.py'), run_name='__main__')
