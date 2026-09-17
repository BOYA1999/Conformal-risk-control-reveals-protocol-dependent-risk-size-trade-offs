import os
import hashlib
import json
from pathlib import Path

import torch
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from torch_geometric.data import Data

from audit_liver_inputs import read_masks

CODE = Path(__file__).resolve().parent
PACKAGE = CODE.parents[1]
WORK = Path(os.environ.get('MOLXAI_WORK_ROOT', '../molxai-work')).resolve()
HERE = WORK / 'independent_reference'
SCIENCE = WORK


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def identities(mol):
    full = Chem.MolToSmiles(mol, isomericSmiles=False)
    fragments = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
    largest = sorted(fragments, key=lambda m: (-m.GetNumHeavyAtoms(), Chem.MolToSmiles(m, isomericSmiles=False)))[0]
    return full, Chem.MolToSmiles(largest, isomericSmiles=False)


def atom_features(mol):
    def onehot(value, choices):
        return [float(value == c) for c in choices] + [float(value not in choices)]
    return torch.tensor([
        onehot(a.GetSymbol(), ['B', 'C', 'N', 'O', 'F', 'Si', 'P', 'S', 'Cl', 'Br', 'I'])
        + onehot(a.GetDegree(), list(range(6)))
        + onehot(a.GetFormalCharge(), [-2, -1, 0, 1, 2])
        + onehot(a.GetTotalNumHs(), list(range(5)))
        + [float(a.GetIsAromatic()), float(a.IsInRing())]
        for a in mol.GetAtoms()
    ], dtype=torch.float32)


def prepare():
    if WORK == PACKAGE or PACKAGE in WORK.parents:
        raise SystemExit('Set MOLXAI_WORK_ROOT outside this repository.')
    RDLogger.DisableLog('rdApp.warning')
    frame = pd.read_csv(HERE / 'data/raw/Liver.csv')
    records = read_masks()
    molecules = [Chem.MolFromSmiles(s) for s in frame.SMILES]
    keys = [identities(m) for m in molecules]
    assert all(r['SMILES'] == s and int(r['label']) + 1 == int(y) for r, s, y in zip(records, frame.SMILES, frame.label))
    refs = [np.asarray(r['node_atts'], dtype=bool).reshape(-1) for r in records]
    assert all(len(r) == m.GetNumAtoms() for r, m in zip(refs, molecules))
    prior_paths = {'bxaic': SCIENCE / 'data/raw/bxaic/data.csv'}
    prior_paths.update({f'google_{t}': SCIENCE / f'reference/graph-attribution/data/{t}/{t}_smiles.csv' for t in ['benzene', 'logic7', 'logic8', 'logic10']})
    overlaps, invalid = {}, {}
    for name, path in prior_paths.items():
        prior, errors = set(), 0
        for smiles in pd.read_csv(path).smiles:
            m = Chem.MolFromSmiles(smiles)
            if m is None:
                errors += 1
                m = Chem.MolFromSmiles(smiles, sanitize=False)
            if m is not None:
                prior.update(identities(m))
        overlaps[name] = [i for i, k in enumerate(keys) if any(v in prior for v in k)]
        invalid[name] = errors
    excluded = set(i for rows in overlaps.values() for i in rows)
    duplicate_groups = {}
    for i, key in enumerate(keys):
        duplicate_groups.setdefault(key[0], []).append(i)
    duplicates = []
    for rows in duplicate_groups.values():
        if len(rows) > 1:
            conflict = len({int(frame.label[i]) for i in rows}) > 1
            ordered = sorted(rows, key=lambda i: (frame.SPLIT[i] != 'test', i))
            dropped = ordered if conflict else ordered[1:]
            excluded.update(dropped)
            duplicates.append({'rows': rows, 'label_conflict': conflict, 'excluded': dropped})
    allocation = {}
    for i in range(len(frame)):
        if i not in excluded and frame.SPLIT[i] == 'test':
            allocation[i] = 'test'
    for label in [0, 1, 2]:
        rows = [i for i in range(len(frame)) if i not in excluded and frame.SPLIT[i] == 'train' and frame.label[i] == label]
        rows.sort(key=lambda i: hashlib.sha256(('molxai_crc_liver_20260912|' + keys[i][0]).encode()).hexdigest())
        ncal, ndev = round(.20 * len(rows)), round(.15 * len(rows))
        for j, i in enumerate(rows):
            allocation[i] = 'calibration' if j < ncal else 'dev' if j < ncal + ndev else 'fit'
    partitions = {s: [] for s in ['fit', 'dev', 'calibration', 'test']}
    manifest = []
    for i, split in sorted(allocation.items()):
        m = molecules[i]
        edges = [(u, v) for b in m.GetBonds() for u, v in [(b.GetBeginAtomIdx(), b.GetEndAtomIdx()), (b.GetEndAtomIdx(), b.GetBeginAtomIdx())]]
        edge_index = torch.tensor(edges, dtype=torch.long).T.contiguous() if edges else torch.empty((2, 0), dtype=torch.long)
        partitions[split].append(Data(x=atom_features(m), edge_index=edge_index, y=torch.tensor(int(frame.label[i])), rationale_mask=torch.tensor(refs[i]), source_index=torch.tensor(i)))
        manifest.append({'source_index': i, 'split': split, 'label': int(frame.label[i]), 'identity_sha256': hashlib.sha256(keys[i][0].encode()).hexdigest(), 'atoms': m.GetNumAtoms(), 'reference_atoms': int(refs[i].sum())})
    (HERE / 'results').mkdir(exist_ok=True)
    pd.DataFrame(manifest).to_csv(HERE / 'results/split_manifest.csv', index=False)
    counts = {s: {'total': len(gs), 'eligible': sum(bool(g.rationale_mask.any()) for g in gs), 'labels': {str(c): sum(int(g.y) == c for g in gs) for c in [0, 1, 2]}} for s, gs in partitions.items()}
    report = {'status': 'PASS', 'source_rows': len(frame), 'admitted_rows': len(manifest), 'exclusion_rule': 'any full or largest-fragment stereo-insensitive identity overlap with original families; conflicting duplicate groups excluded, otherwise official-test/earliest-row precedence', 'overlaps': overlaps, 'duplicate_groups': duplicates, 'excluded_source_rows': sorted(excluded), 'prior_sanitization_failures': invalid, 'counts': counts, 'feature_dimension': partitions['fit'][0].x.shape[1], 'source_hashes': {p.name: digest(p) for p in (HERE / 'data/raw').iterdir() if p.is_file()}, 'code_sha256': digest(__file__), 'split_sha256': digest(HERE / 'results/split_manifest.csv')}
    (HERE / 'results/data_audit.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    assert all({int(g.y) for g in gs} == {0, 1, 2} for gs in partitions.values())
    assert counts['calibration']['eligible'] > 0 and counts['test']['eligible'] > 0
    return partitions, report


if __name__ == '__main__':
    _, report = prepare()
    print(json.dumps(report, indent=2))
