import hashlib
import itertools
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

PACKAGE = Path(__file__).resolve().parents[3]
WORK = Path(os.environ.get('MOLXAI_WORK_ROOT', '../molxai-work')).resolve()
HERE = Path(os.environ.get('MOLXAI_GRID_ROOT', WORK / 'revision/grid')).resolve()
P20 = WORK / 'full_grid'
SCIENCE = PACKAGE
METRICS = ["risk", "mean_atom_fraction", "precision", "iou"]
MACHINE = np.linspace(0, 1, 101)


def independent_curves(cache, method, split, exact):
    offsets = cache[f"{split}__offsets"]
    scores, masks = cache[f"{split}__{method}"], cache[f"{split}__rationale"]
    lengths = np.diff(offsets).astype(int)
    counts = np.array([[(q * int(m) + 99) // 100 if exact else int(np.ceil(MACHINE[q] * m)) for q in range(101)] for m in lengths])
    totals = np.zeros((101, 4))
    for i, (start, stop) in enumerate(zip(offsets[:-1], offsets[1:])):
        values, truth = scores[start:stop], masks[start:stop].astype(bool)
        order = sorted(range(len(values)), key=lambda atom: (-float(values[atom]), atom))
        cumulative = np.r_[0, np.cumsum(truth[order])]
        hits, k, r = cumulative[counts[i]], counts[i], int(truth.sum())
        totals[:, 0] += 1 - hits / r
        totals[:, 1] += k / len(values)
        totals[:, 2] += np.divide(hits, k, out=np.zeros(101), where=k > 0)
        totals[:, 3] += hits / (k + r - hits)
    return totals / len(lengths), counts


def summarize():
    points = pd.read_csv(HERE / "operating_points.csv")
    keys = ["method", "score_origin", "grid", "alpha"]
    task = points.groupby(keys + ["family", "task"], as_index=False)[METRICS].mean()
    task.to_csv(HERE / "task_summary.csv", index=False)
    task.groupby(keys + ["family"], as_index=False)[METRICS].mean().to_csv(HERE / "family_summary.csv", index=False)
    rows = []
    for key, group in points.groupby(keys):
        tasks = group.groupby(["family", "task"])[METRICS].mean()
        rows.append({**dict(zip(keys, key)), "cells": len(group), "tasks": len(tasks), **tasks.mean().to_dict(), "risk_pass": int((group.risk <= key[-1]).sum()), "size_pass": int((group.mean_atom_fraction < .8).sum()), "joint_pass": int(((group.risk <= key[-1]) & (group.mean_atom_fraction < .8)).sum())})
    pd.DataFrame(rows).to_csv(HERE / "method_summary.csv", index=False)
    curve = pd.read_csv(HERE / "curves.csv")
    curve.groupby(["method", "score_origin", "grid", "split", "q", "nominal_fraction"], as_index=False)[METRICS].mean().to_csv(HERE / "curve_summary.csv", index=False)
    effect = pd.read_csv(HERE / "grid_effect_cells.csv")
    rows = []
    for key, group in effect.groupby(["method", "alpha", "comparison"]):
        row = dict(zip(["method", "alpha", "comparison"], key))
        row.update(cells=len(group), q_changed_cells=int((group.q_change != 0).sum()), max_abs_q_change=int(group.q_change.abs().max()))
        for split in ["calibration", "test"]:
            row[f"{split}_occurrences"] = int(group[f"{split}_occurrences"].sum())
            row[f"{split}_changed_sets"] = int(group[f"{split}_changed_sets"].sum())
            row[f"{split}_changed_cells"] = int((group[f"{split}_changed_sets"] > 0).sum())
        for metric in METRICS:
            row[f"macro_delta_{metric}"] = float(group.groupby(["family", "task"])[f"delta_{metric}"].mean().mean())
            row[f"max_abs_delta_{metric}"] = float(group[f"delta_{metric}"].abs().max())
        rows.append(row)
    pd.DataFrame(rows).to_csv(HERE / "grid_effect_summary.csv", index=False)
    replay = pd.read_csv(HERE / "historical_replay.csv")
    rows = []
    for key, group in replay.groupby(["method", "score_origin", "alpha"]):
        row = dict(zip(["method", "score_origin", "alpha"], key))
        row.update(cells=len(group), q_changed_cells=int((group.q_change != 0).sum()))
        for metric in METRICS:
            row[f"mean_delta_{metric}"] = float(group[f"delta_{metric}"].mean())
            row[f"max_abs_delta_{metric}"] = float(group[f"delta_{metric}"].abs().max())
        rows.append(row)
    pd.DataFrame(rows).to_csv(HERE / "historical_replay_summary.csv", index=False)
    paired, reversals, thresholds = [], [], []
    for alpha in [.05, .1, .2]:
        panel = points[points.alpha == alpha]
        for grid, group in panel.groupby("grid"):
            for left, right in itertools.combinations(["ig", "gradinput", "atom_occlusion", "saliency"], 2):
                for metric in METRICS:
                    wide = group.pivot(index=["family", "task", "cell_id"], columns="method", values=metric)
                    difference = wide[left] - wide[right]
                    by_task = difference.groupby(level=["family", "task"]).mean().to_numpy()
                    rng = np.random.default_rng(20260913)
                    draws = by_task[rng.integers(0, len(by_task), (2000, len(by_task)))].mean(axis=1)
                    paired.append({"grid": grid, "alpha": alpha, "left": left, "right": right, "metric": metric, "mean_delta": float(by_task.mean()), "q025": float(np.quantile(draws, .025)), "q975": float(np.quantile(draws, .975)), "negative_cells": int((difference < 0).sum()), "positive_cells": int((difference > 0).sum()), "ties": int((difference == 0).sum()), "interval_scope": "descriptive 11 observed tasks from two dependent families"})
            for method, method_rows in group.groupby("method"):
                for threshold in [.5, .6, .7, .8, .9]:
                    size = method_rows.mean_atom_fraction < threshold
                    risk = method_rows.risk <= alpha
                    thresholds.append({"grid": grid, "alpha": alpha, "method": method, "threshold": threshold, "size_pass": int(size.sum()), "risk_pass": int(risk.sum()), "joint_pass": int((size & risk).sum())})
        for left, right in itertools.combinations(["ig", "gradinput", "atom_occlusion", "saliency"], 2):
            for metric in METRICS:
                wide = panel.pivot(index="cell_id", columns=["grid", "method"], values=metric)
                machine = wide["machine", left] - wide["machine", right]
                exact = wide["exact_integer", left] - wide["exact_integer", right]
                reversals.append({"alpha": alpha, "left": left, "right": right, "metric": metric, "strict_reversals": int((machine * exact < 0).sum()), "non_tied_both": int(((machine != 0) & (exact != 0)).sum()), "machine_ties": int((machine == 0).sum()), "exact_ties": int((exact == 0).sum())})
    pd.DataFrame(paired).to_csv(HERE / "paired_method_comparisons.csv", index=False)
    pd.DataFrame(reversals).to_csv(HERE / "method_order_grid.csv", index=False)
    pd.DataFrame(thresholds).to_csv(HERE / "threshold_counts.csv", index=False)


def main():
    recorded_runner = json.loads((HERE / "environment.json").read_text())["runner_sha256"]
    assert hashlib.sha256((HERE / "run_grid.py").read_bytes()).hexdigest().upper() == recorded_runner
    imported = json.loads((P20 / "environment_full.json").read_text())["imported_sources"]
    source_checks = []
    for name, expected in imported.items():
        observed = hashlib.sha256((SCIENCE / "src" / name).read_bytes()).hexdigest().upper()
        assert observed == expected
        source_checks.append({"source": name, "sha256": observed, "matches_frozen_p20": True})
    (HERE / "imported_source_audit.json").write_text(json.dumps(source_checks, indent=2), encoding="utf-8")
    points = pd.read_csv(HERE / "operating_points.csv")
    curve = pd.read_csv(HERE / "curves.csv")
    effects = pd.read_csv(HERE / "grid_effect_cells.csv")
    cardinality = pd.read_csv(HERE / "cardinality_counts.csv")
    paths = sorted((HERE / "cells").glob("*/complete.json"))
    assert len(paths) == 66 and len(points) == 1584 and len(curve) == 106656 and len(effects) == 2376
    assert not curve.duplicated(["cell_id", "method", "grid", "split", "q"]).any()
    assert not points.duplicated(["cell_id", "method", "grid", "alpha"]).any()
    maximum, checks = 0., []
    for path in paths:
        cell_id = path.parent.name
        gradient_path = HERE / "scores" / f"{cell_id}.npz"
        meta = json.loads(path.read_text())
        score_meta = json.loads(gradient_path.with_suffix(".json").read_text())
        assert meta["runner_sha256"] == score_meta["runner_sha256"] == recorded_runner
        assert hashlib.sha256(gradient_path.read_bytes()).hexdigest().upper() == meta["gradient_cache_sha256"]
        assert hashlib.sha256((P20 / "scores" / f"{cell_id}.npz").read_bytes()).hexdigest().upper() == meta["p20_cache_sha256"]
        gradient = dict(np.load(gradient_path, allow_pickle=False))
        p20 = dict(np.load(P20 / "scores" / f"{cell_id}.npz", allow_pickle=False))
        for split in ["calibration", "test"]:
            for suffix in ["source_ids", "labels", "rationale", "offsets"]:
                assert np.array_equal(gradient[f"{split}__{suffix}"], p20[f"{split}__{suffix}"])
        assert not set(p20["calibration__source_ids"]) & set(p20["test__source_ids"])
        for method in ["ig", "gradinput", "atom_occlusion", "saliency"]:
            cache = gradient if method in ["ig", "gradinput"] else p20
            generated = {}
            for split in ["calibration", "test"]:
                for grid in ["machine", "exact_integer"]:
                    values, counts = independent_curves(cache, method, split, grid == "exact_integer")
                    generated[split, grid] = values, counts
                    actual = curve[(curve.cell_id == cell_id) & (curve.method == method) & (curve.split == split) & (curve.grid == grid)].sort_values("q")
                    error = float(np.max(np.abs(values - actual[METRICS].to_numpy())))
                    maximum = max(maximum, error)
                    assert error < 1e-10, (cell_id, method, split, grid, error)
                    assert np.all(np.diff(values[:, 0]) <= 1e-12) and values[0, 0] == 1 and values[-1, 0] == 0
                    for alpha in [.05, .1, .2]:
                        if split == "test":
                            cal, cal_counts = generated["calibration", grid]
                            n = len(cal_counts)
                            corrected = (cal[:, 0] * n + 1) / (n + 1)
                            q = int(np.where(corrected <= alpha)[0][0])
                            row = points[(points.cell_id == cell_id) & (points.method == method) & (points.grid == grid) & (points.alpha == alpha)].iloc[0]
                            assert int(row.q) == q and abs(row.corrected_risk - corrected[q]) < 1e-12
                            assert np.max(np.abs(values[q] - row[METRICS].to_numpy(dtype=float))) < 1e-10
            for _, row in effects[(effects.cell_id == cell_id) & (effects.method == method)].iterrows():
                qa, qb = int(row.machine_q), int(row.exact_q)
                for split in ["calibration", "test"]:
                    machine, mc = generated[split, "machine"]
                    exact, ec = generated[split, "exact_integer"]
                    assert int(np.count_nonzero(mc[:, qa] != ec[:, qb])) == row[f"{split}_changed_sets"]
                delta = generated["test", "exact_integer"][0][qb] - generated["test", "machine"][0][qa]
                assert np.max(np.abs(delta - row[[f"delta_{m}" for m in METRICS]].to_numpy(dtype=float))) < 1e-10
            if method == "ig":
                for split in ["calibration", "test"]:
                    affected = generated[split, "machine"][1] != generated[split, "exact_integer"][1]
                    row = cardinality[(cardinality.cell_id == cell_id) & (cardinality.split == split)].iloc[0]
                    assert row.affected_grid_pairs == int(affected.sum()) and row.affected_occurrences == int(affected.any(axis=1).sum())
        checks.append({"cell_id": cell_id, "status": "PASS"})
        print(f"independent_QA {len(checks)}/66 {cell_id}", flush=True)
    counts = cardinality.groupby("split")["occurrences"].sum().to_dict()
    assert counts == {"calibration": 82866, "test": 81672}
    summarize()
    result = {"status": "PASS", "cells": 66, "curve_rows": len(curve), "operating_points": len(points), "maximum_curve_difference": maximum, "occurrences_per_method": counts, "checks": checks, "independence": "Separate evaluator uses Python stable sorting and cumulative reference hits, rather than runner lexsort/searchsorted; all saved scores and both grids reconstructed, no new scoring or training.", "validator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest().upper()}
    (HERE / "independent_QA.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "checks"}), flush=True)


if __name__ == "__main__":
    main()
