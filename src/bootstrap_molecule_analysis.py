import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results" / "molecule_level_ig_seed42" / "molecules.csv.gz"
OUT = ROOT / "results" / "molecule_level_ig_seed42" / "bootstrap"
REPLICATES = 2000


def bootstrap(values, rng):
    values = np.asarray(values, dtype=float)
    means = np.empty(REPLICATES)
    for start in range(0, REPLICATES, 100):
        stop = min(start + 100, REPLICATES)
        indices = rng.integers(0, len(values), size=(stop - start, len(values)))
        means[start:stop] = values[indices].mean(axis=1)
    return float(values.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975)), means


def holm(pvalues):
    pvalues = np.asarray(pvalues)
    order = np.argsort(pvalues)
    adjusted_sorted = np.maximum.accumulate((len(pvalues) - np.arange(len(pvalues))) * pvalues[order])
    adjusted = np.empty(len(pvalues))
    adjusted[order] = np.minimum(adjusted_sorted, 1.0)
    return adjusted


data = pd.read_csv(SOURCE)
rows = []
for cell_id, frame in data.groupby("cell_id", sort=True):
    rng = np.random.default_rng(int.from_bytes(cell_id.encode()[:8].ljust(8, b"0"), "little"))
    rationale = frame[frame["n_rationale"] > 0]
    risk, risk_low, risk_high, _ = bootstrap(rationale["miss_loss"], rng)
    fidelity, fidelity_low, fidelity_high, fidelity_boot = bootstrap(frame["fidelity_advantage"], rng)
    pvalue = (1 + int((fidelity_boot <= 0).sum())) / (REPLICATES + 1)
    first = frame.iloc[0]
    rows.append({
        "cell_id": cell_id,
        "family": first["family"],
        "task": first["task"],
        "model": first["model"],
        "molecules": len(frame),
        "rationale_molecules": len(rationale),
        "mean_miss_loss": risk,
        "risk_ci_low": risk_low,
        "risk_ci_high": risk_high,
        "risk_point_pass": risk <= 0.10,
        "risk_ci_upper_pass": risk_high <= 0.10,
        "mean_fidelity_advantage": fidelity,
        "fidelity_ci_low": fidelity_low,
        "fidelity_ci_high": fidelity_high,
        "fidelity_p_one_sided": pvalue,
    })
cells = pd.DataFrame(rows)
cells["fidelity_p_holm"] = holm(cells["fidelity_p_one_sided"])
cells["fidelity_holm_positive"] = (cells["mean_fidelity_advantage"] > 0) & (cells["fidelity_p_holm"] <= 0.05)

strata_rows = []
nonnull = data[data["n_rationale"] > 0]
for stratum_type, column, frame in [
    ("rationale_fraction", "rationale_group", nonnull),
    ("atom_size", "atom_size_group", nonnull),
]:
    for stratum, group in frame.groupby(column, sort=True):
        seed = int.from_bytes(hashlib.sha256(f"{stratum_type}:{stratum}".encode()).digest()[:8], "little")
        rng = np.random.default_rng(seed)
        for metric in ["miss_loss", "selected_fraction", "iou", "fidelity_advantage"]:
            point, low, high, _ = bootstrap(group[metric], rng)
            strata_rows.append({
                "stratum_type": stratum_type,
                "stratum": stratum,
                "metric": metric,
                "molecules": len(group),
                "mean": point,
                "ci_low": low,
                "ci_high": high,
                "scope": "pooled_molecule_descriptive",
            })
strata = pd.DataFrame(strata_rows)

OUT.mkdir(parents=True, exist_ok=True)
cells.to_csv(OUT / "cell_bootstrap.csv", index=False)
strata.to_csv(OUT / "strata_bootstrap.csv", index=False)
summary = {
    "status": "PASS",
    "replicates": REPLICATES,
    "cells": len(cells),
    "risk_point_pass_cells": int(cells["risk_point_pass"].sum()),
    "risk_ci_upper_pass_cells": int(cells["risk_ci_upper_pass"].sum()),
    "fidelity_positive_cells": int((cells["mean_fidelity_advantage"] > 0).sum()),
    "fidelity_holm_positive_cells": int(cells["fidelity_holm_positive"].sum()),
    "boundary": "Bootstrap intervals and Holm results are descriptive support and do not replace preregistered gates.",
}
(OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps(summary, indent=2))
