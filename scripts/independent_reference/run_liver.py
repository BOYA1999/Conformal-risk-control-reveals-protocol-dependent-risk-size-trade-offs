import os
import copy
import json
import sys
import time
from pathlib import Path

import torch
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, average_precision_score
from torch import nn
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader

from prepare_liver import HERE, SCIENCE, CODE, PACKAGE, prepare, digest

sys.path.insert(0, str(CODE.parent / 'full_grid'))
import run_full_grid as grid

grid.DEVICE = torch.device('cpu')
torch.set_num_threads(4)
METHODS = ['gradinput', 'ig', 'saliency', 'atom_occlusion']
RESULTS = HERE / 'results'


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False), encoding='utf-8')


@torch.no_grad()
def predict(model, graphs):
    logits = torch.cat([model(b.x, b.edge_index, b.batch) for b in DataLoader(graphs, batch_size=64, shuffle=False)])
    labels = np.asarray([int(g.y) for g in graphs])
    probabilities = logits.softmax(-1).numpy()
    return {'accuracy': accuracy_score(labels, probabilities.argmax(-1)), 'balanced_accuracy': balanced_accuracy_score(labels, probabilities.argmax(-1)), 'macro_ovr_auroc': roc_auc_score(labels, probabilities, multi_class='ovr', average='macro'), 'macro_ap': average_precision_score(np.eye(3)[labels], probabilities, average='macro'), 'majority_accuracy': float(np.bincount(labels).max() / len(labels))}, probabilities


def train(kind, seed, parts):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = grid.GraphClassifier(kind, parts['fit'][0].x.shape[1])
    model.head = nn.Linear(32, 3)
    labels = torch.tensor([int(g.y) for g in parts['fit']])
    sampler = torch.utils.data.WeightedRandomSampler((1. / torch.bincount(labels).float())[labels], len(labels), replacement=True)
    loader = DataLoader(parts['fit'], batch_size=64, sampler=sampler)
    optimizer = torch.optim.Adam(model.parameters(), lr=.001, weight_decay=.0001)
    best, best_state, best_epoch, stale, rows = -1., None, 0, 0, []
    for epoch in range(1, 101):
        model.train()
        for b in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(model(b.x, b.edge_index, b.batch), b.y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.)
            optimizer.step()
        model.eval()
        metrics, _ = predict(model, parts['dev'])
        score = metrics['macro_ovr_auroc']
        rows.append({'epoch': epoch, **metrics})
        if score > best + 1e-5:
            best, best_state, best_epoch, stale = score, copy.deepcopy(model.state_dict()), epoch, 0
        else:
            stale += 1
        if epoch >= 20 and stale >= 15:
            break
    model.load_state_dict(best_state)
    model.eval()
    return model, {'best_epoch': best_epoch, 'epochs': epoch, 'development_metrics': rows[best_epoch - 1], 'history': rows}


def gradient_rows(model, graphs, method):
    output = []
    for batch in DataLoader(graphs, batch_size=32, shuffle=False):
        gradients = torch.zeros_like(batch.x)
        steps = 20 if method == 'ig' else 1
        for step in range(1, steps + 1):
            x = (batch.x * step / steps).detach().requires_grad_(True)
            logits = model(x, batch.edge_index, batch.batch)
            score = logits[torch.arange(batch.num_graphs), batch.y].sum()
            gradients += torch.autograd.grad(score, x)[0]
        values = (batch.x * gradients / steps).sum(-1).detach().numpy()
        output.extend(values[a:b].copy() for a, b in zip(batch.ptr[:-1], batch.ptr[1:]))
    return output


def evaluate_scores(cal, test, cal_masks, test_masks, policy):
    cal_curves, test_curves = grid.curves(cal, cal_masks, policy), grid.curves(test, test_masks, policy)
    n = len(cal)
    points, molecules = [], []
    for alpha in [.05, .10, .20]:
        corrected = (cal_curves['risk'].sum(0) + 1) / (n + 1)
        feasible = np.flatnonzero(corrected <= alpha)
        index = int(feasible[0]) if len(feasible) else 100
        row = {'alpha': alpha, 'selected_fraction': float(grid.FRACTIONS[index]), 'corrected_calibration_risk': float(corrected[index]), 'fallback_full': not bool(len(feasible)), 'n_calibration': n, 'n_test': len(test)}
        row.update({key: float(values[:, index].mean()) for key, values in test_curves.items()})
        rows = []
        for i, (scores, mask) in enumerate(zip(test, test_masks)):
            values = {key: float(table[i, index]) for key, table in test_curves.items()}
            count = int(round(values['mean_atom_fraction'] * len(scores)))
            values.update({'test_index': i, 'atoms': len(scores), 'reference_atoms': int(mask.sum()), 'selected_atoms': count, 'equal_size_random_risk': 1 - count / len(scores), 'reference_first_schedule_risk': 1 - min(count, int(mask.sum())) / int(mask.sum())})
            rows.append(values)
        row['equal_size_random_risk'] = float(np.mean([r['equal_size_random_risk'] for r in rows]))
        row['reference_first_schedule_risk'] = float(np.mean([r['reference_first_schedule_risk'] for r in rows]))
        for fraction in [.2, .5]:
            j = round(fraction * 100)
            row.update({f'fixed_{int(fraction * 100)}_{key}': float(values[:, j].mean()) for key, values in test_curves.items() if key != 'tie_inflation'})
        points.append(row)
        if alpha == .1:
            molecules = rows
    return points, molecules


def main():
    started = time.time()
    parts, audit = prepare()
    contract = {'run_id': 'liver_external_20260912', 'started_unix': started, 'command': sys.argv, 'plan_sha256': digest(PACKAGE / 'contracts/independent_reference/run_contract.json'), 'code_sha256': digest(__file__), 'adapter_sha256': digest(CODE / 'prepare_liver.py'), 'safe_reader_sha256': digest(CODE / 'audit_liver_inputs.py'), 'metric_code_sha256': digest(CODE.parent / 'full_grid/run_full_grid.py'), 'data': audit, 'environment': {'python': sys.version, 'torch': torch.__version__, 'numpy': np.__version__, 'device': 'cpu', 'threads': 4}, 'targets': ['observed_class', 'hepatotoxic_class'], 'methods': METHODS, 'models': ['gin', 'gcn'], 'seeds': [42, 123, 2026]}
    write(RESULTS / 'run_manifest.json', contract)
    grid.policy_check()
    for kind in ['gin', 'gcn']:
        for seed in [42, 123, 2026]:
            cell = f'liver__{kind}__seed{seed}'
            output = RESULTS / 'cells' / f'{cell}.json'
            if output.exists():
                assert json.loads(output.read_text())['run_code_sha256'] == digest(__file__)
                continue
            begin = time.time()
            print(f'train_start={cell}', flush=True)
            model, training = train(kind, seed, parts)
            predictor = {}
            for split in ['dev', 'calibration', 'test']:
                predictor[split], probabilities = predict(model, parts[split])
                np.savez_compressed(RESULTS / f'{cell}_{split}_predictions.npz', source_ids=np.asarray([int(g.source_index) for g in parts[split]]), labels=np.asarray([int(g.y) for g in parts[split]]), probabilities=probabilities)
            torch.save({'state_dict': model.state_dict(), 'model': kind, 'seed': seed, 'feature_dimension': audit['feature_dimension']}, RESULTS / f'{cell}.pt')
            all_points, all_molecules = [], []
            for target in contract['targets']:
                graphs = {s: [g.clone() for g in parts[s] if bool(g.rationale_mask.any())] for s in ['calibration', 'test']}
                if target == 'hepatotoxic_class':
                    for gs in graphs.values():
                        for g in gs:
                            g.y = torch.tensor(2)
                masks = {s: [g.rationale_mask.numpy() for g in gs] for s, gs in graphs.items()}
                for method in METHODS:
                    score_rows = {}
                    for split, gs in graphs.items():
                        score_rows[split] = gradient_rows(model, gs, method) if method in ['gradinput', 'ig'] else grid.saliency(model, gs) if method == 'saliency' else grid.occlusion(model, gs)
                        np.savez_compressed(RESULTS / f'{cell}_{target}_{method}_{split}_scores.npz', scores=np.concatenate(score_rows[split]), ptr=np.r_[0, np.cumsum([g.num_nodes for g in gs])], masks=np.concatenate(masks[split]), source_ids=np.asarray([int(g.source_index) for g in gs]))
                    for policy in grid.POLICIES:
                        points, rows = evaluate_scores(score_rows['calibration'], score_rows['test'], masks['calibration'], masks['test'], policy)
                        identity = {'cell_id': cell, 'model': kind, 'seed': seed, 'target': target, 'method': method, 'policy': policy}
                        all_points.extend({**identity, **p} for p in points)
                        all_molecules.extend({**identity, 'source_index': int(graphs['test'][r['test_index']].source_index), 'label': int(next(g.y for g in parts['test'] if int(g.source_index) == int(graphs['test'][r['test_index']].source_index))), **r} for r in rows)
            pd.DataFrame(all_points).to_csv(RESULTS / f'{cell}_metrics.csv', index=False)
            pd.DataFrame(all_molecules).to_csv(RESULTS / f'{cell}_molecules.csv', index=False)
            write(output, {'status': 'complete', 'cell_id': cell, 'training': training, 'predictor': predictor, 'seconds': time.time() - begin, 'run_code_sha256': digest(__file__), 'checkpoint_sha256': digest(RESULTS / f'{cell}.pt')})
            print(f'cell_complete={cell} seconds={time.time()-begin:.1f} test={predictor["test"]}', flush=True)
    files = list((RESULTS / 'cells').glob('*.json'))
    assert len(files) == 6
    pd.concat([pd.read_csv(p) for p in sorted(RESULTS.glob('liver__*_metrics.csv'))]).to_csv(RESULTS / 'cell_metrics.csv', index=False)
    pd.concat([pd.read_csv(p) for p in sorted(RESULTS.glob('liver__*_molecules.csv'))]).to_csv(RESULTS / 'molecule_metrics.csv', index=False)
    write(RESULTS / 'completion.json', {'status': 'complete', 'cells': 6, 'seconds': time.time() - started, 'finished_unix': time.time()})


if __name__ == '__main__':
    main()
