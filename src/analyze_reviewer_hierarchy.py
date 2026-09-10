import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/reviewer_hierarchy_analysis"
BOOTSTRAP = 2000


def partial_spearman(frame, x, y, controls):
    x_rank = rankdata(frame[x])
    y_rank = rankdata(frame[y])
    design = np.column_stack([np.ones(len(frame))] + [rankdata(frame[column]) for column in controls])
    x_resid = x_rank - design @ np.linalg.lstsq(design, x_rank, rcond=None)[0]
    y_resid = y_rank - design @ np.linalg.lstsq(design, y_rank, rcond=None)[0]
    return float(np.corrcoef(x_resid, y_resid)[0, 1])


def standardized_ols(frame, outcome, predictors):
    columns = [outcome] + predictors
    values = frame[columns].to_numpy(float)
    means, scales = values.mean(axis=0), values.std(axis=0)
    if (scales < 1e-12).any():
        return None
    values = (values - means) / scales
    design = np.column_stack([np.ones(len(frame)), values[:, 1:]])
    coefficients = np.linalg.lstsq(design, values[:, 0], rcond=None)[0][1:]
    return dict(zip(predictors, coefficients))


def cluster_sample(frame, rng):
    tasks = frame[["family", "task"]].drop_duplicates().to_records(index=False)
    chosen = rng.integers(0, len(tasks), size=len(tasks))
    pieces = []
    for draw, index in enumerate(chosen):
        family, task = tasks[index]
        piece = frame[(frame["family"] == family) & (frame["task"] == task)].copy()
        piece["bootstrap_cluster"] = draw
        pieces.append(piece)
    return pd.concat(pieces, ignore_index=True)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cells = pd.read_csv(ROOT / "results/reviewer_crc_resampling/cells.csv")
    main = pd.read_csv(ROOT / "results/gradient_grid_phase_a/cell_alpha010.csv")
    main = main[(main["seed"] == 42) & (main["explainer"] == "ig")]
    oracle = pd.read_csv(ROOT / "results/reviewer_oracle_crc/task_oracle_all_alpha.csv")
    oracle = oracle[np.isclose(oracle["alpha"], 0.10)].drop(columns="alpha")
    frame = cells.merge(
        main[[
            "cell_id", "fraction_grid", "mean_atom_fraction", "random_mean_atom_fraction",
            "atom_fraction_saving_vs_random", "test_risk", "test_auroc",
        ]],
        on="cell_id",
        validate="one_to_one",
    ).merge(oracle, on=["family", "task"], validate="many_to_one")
    frame["excess_grid_fraction"] = frame["main_fraction"] - frame["oracle_fraction"]
    frame["excess_realized_atom_fraction"] = frame["mean_atom_fraction"] - frame["oracle_test_mean_atom_fraction"]
    if len(frame) != 22 or frame[["family", "task"]].drop_duplicates().shape[0] != 11:
        raise ValueError("hierarchy analysis requires exactly 22 cells and 11 task clusters")
    if not np.allclose(frame["main_fraction"], frame["fraction_grid"]):
        raise ValueError("cached and main calibrated fractions disagree")
    frame.to_csv(OUT / "cell_stats.csv", index=False)

    metrics = [
        ("spearman_actual_oracle", "spearman", "main_fraction", "oracle_fraction", []),
        ("spearman_actual_mean_rationale", "spearman", "main_fraction", "mean_test_rationale_fraction", []),
        ("spearman_actual_median_rationale", "spearman", "main_fraction", "median_test_rationale_fraction", []),
        ("spearman_actual_mean_atoms", "spearman", "main_fraction", "mean_test_atoms", []),
        ("spearman_excess_fixed20_iou", "spearman", "excess_grid_fraction", "fixed20_iou", []),
        ("spearman_excess_mean_atoms", "spearman", "excess_grid_fraction", "mean_test_atoms", []),
        ("spearman_random_saving_fixed20_iou", "spearman", "atom_fraction_saving_vs_random", "fixed20_iou", []),
        ("partial_actual_oracle_given_atoms", "partial", "main_fraction", "oracle_fraction", ["mean_test_atoms"]),
        ("partial_actual_rationale_given_atoms", "partial", "main_fraction", "mean_test_rationale_fraction", ["mean_test_atoms"]),
        ("partial_excess_iou_given_atoms_rationale", "partial", "excess_grid_fraction", "fixed20_iou", ["mean_test_atoms", "mean_test_rationale_fraction"]),
    ]
    points = {}
    for name, kind, x, y, controls in metrics:
        points[name] = (
            float(spearmanr(frame[x], frame[y]).statistic)
            if kind == "spearman"
            else partial_spearman(frame, x, y, controls)
        )

    models = [
        ("actual_fraction_model", "main_fraction", ["oracle_fraction", "fixed20_iou", "mean_test_atoms"]),
        ("excess_fraction_model", "excess_grid_fraction", ["fixed20_iou", "mean_test_atoms", "mean_test_rationale_fraction"]),
    ]
    point_coefficients = {}
    for model_name, outcome, predictors in models:
        fitted = standardized_ols(frame, outcome, predictors)
        for predictor, coefficient in fitted.items():
            point_coefficients[f"{model_name}:{predictor}"] = float(coefficient)

    rng = np.random.default_rng(20260809)
    bootstrap_rows = []
    for replicate in range(BOOTSTRAP):
        sampled = cluster_sample(frame, rng)
        for name, kind, x, y, controls in metrics:
            value = (
                float(spearmanr(sampled[x], sampled[y]).statistic)
                if kind == "spearman"
                else partial_spearman(sampled, x, y, controls)
            )
            if np.isfinite(value):
                bootstrap_rows.append({"replicate": replicate, "analysis": "correlation", "term": name, "value": value})
        for model_name, outcome, predictors in models:
            fitted = standardized_ols(sampled, outcome, predictors)
            if fitted is not None:
                for predictor, coefficient in fitted.items():
                    if np.isfinite(coefficient):
                        bootstrap_rows.append({
                            "replicate": replicate,
                            "analysis": "standardized_ols",
                            "term": f"{model_name}:{predictor}",
                            "value": float(coefficient),
                        })
    bootstrap = pd.DataFrame(bootstrap_rows)
    bootstrap.to_csv(OUT / "task_cluster_bootstrap.csv.gz", index=False, compression="gzip")

    association_rows = []
    all_points = {**points, **point_coefficients}
    for term, point in all_points.items():
        values = bootstrap.loc[bootstrap["term"] == term, "value"]
        association_rows.append({
            "term": term,
            "analysis": "correlation" if term in points else "standardized_ols",
            "point_estimate": point,
            "bootstrap_valid": int(len(values)),
            "bootstrap_median": float(values.median()),
            "ci95_low": float(values.quantile(0.025)),
            "ci95_high": float(values.quantile(0.975)),
            "same_sign_frequency": float((np.sign(values) == np.sign(point)).mean()),
        })
    associations = pd.DataFrame(association_rows)
    associations.to_csv(OUT / "associations.csv", index=False)
    payload = {
        "status": "complete",
        "cells": len(frame),
        "task_clusters": 11,
        "task_cluster_bootstrap_repetitions": BOOTSTRAP,
        "unit_of_inference": "task-backbone cell; molecule rows used only for cell descriptors",
        "interpretation": "Associational and exploratory; intervals reflect resampling of observed tasks.",
        "associations": json.loads(associations.to_json(orient="records")),
    }
    (OUT / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
