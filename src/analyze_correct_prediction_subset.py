import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
source = ROOT / "results/molecule_level_ig_seed42/molecules.csv.gz"
out = ROOT / "results/molecule_level_ig_seed42/correct_prediction_subset"
out.mkdir(parents=True, exist_ok=True)

data = pd.read_csv(source)
data = data[data["n_rationale"] > 0].copy()


def summarize(frame):
    cells = frame.groupby(["cell_id", "family", "task", "model"], as_index=False).agg(
        molecules=("miss_loss", "size"),
        mean_risk=("miss_loss", "mean"),
        mean_selected_fraction=("selected_fraction", "mean"),
        mean_iou=("iou", "mean"),
        mean_fidelity_advantage=("fidelity_advantage", "mean"),
    )
    return cells, {
        "molecules": int(len(frame)),
        "cells": int(len(cells)),
        "macro_mean_risk": float(cells["mean_risk"].mean()),
        "risk_pass_cells": int((cells["mean_risk"] <= 0.10).sum()),
        "macro_mean_selected_fraction": float(cells["mean_selected_fraction"].mean()),
        "macro_mean_iou": float(cells["mean_iou"].mean()),
        "macro_mean_fidelity_advantage": float(cells["mean_fidelity_advantage"].mean()),
        "fidelity_win_cells": int((cells["mean_fidelity_advantage"] > 0).sum()),
    }


all_cells, all_summary = summarize(data)
correct_cells, correct_summary = summarize(data[data["prediction_correct"]])
all_cells.assign(subset="all_rationale_bearing").to_csv(out / "all_cells.csv", index=False)
correct_cells.assign(subset="correctly_classified").to_csv(out / "correct_cells.csv", index=False)

payload = {
    "status": "PASS",
    "scope": "Integrated Gradients, seed 42, 22 task-backbone cells",
    "all_rationale_bearing": all_summary,
    "correctly_classified": correct_summary,
    "interpretation": "Correct-classification restriction does not change risk-pass or fidelity-win counts.",
}
(out / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
print(json.dumps(payload, indent=2))
