import json
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

from audit_graph_attribution import stratified_partitions
from run_gradient_grid import BXAIC_TASKS, GOOGLE_TASKS


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results" / "molecule_level_ig_seed42" / "molecules.csv.gz"
OUT = ROOT / "results" / "molecule_level_ig_seed42" / "scaffold_novelty"


def scaffold(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return "<INVALID>"
    mol = Chem.Mol(mol)
    Chem.RemoveStereochemistry(mol)
    value = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
    return value or "<ACYCLIC>"


maps = []
bxaic = pd.read_csv(ROOT / "data/raw/bxaic/data.csv")
bxaic_scaffolds = [scaffold(value) for value in bxaic["smiles"].astype(str)]
bxaic_seen = {bxaic_scaffolds[i] for i in bxaic.index[bxaic["split_0"] == "train"] if i != 997}
for task in BXAIC_TASKS:
    for i in bxaic.index[bxaic["split_0"] == "test"]:
        if i != 997:
            maps.append({"family": "bxaic", "task": task, "source_index": i, "scaffold": bxaic_scaffolds[i], "scaffold_seen": bxaic_scaffolds[i] in bxaic_seen})

for task in GOOGLE_TASKS:
    folder = ROOT / "reference/graph-attribution/data" / task
    frame = pd.read_csv(folder / f"{task}_smiles.csv")
    official = np.load(folder / f"{task}_traintest_indices.npz")
    labels = np.load(folder / "y_true.npz")["y"].reshape(-1).astype(int)
    fit, dev, _ = stratified_partitions(official["train_index"].astype(int), labels, frame["mol_id"].astype(str).to_numpy())
    scaffolds = [scaffold(value) for value in frame["smiles"].astype(str)]
    seen = {scaffolds[i] for i in np.concatenate([fit, dev])}
    for i in official["test_index"].astype(int):
        maps.append({"family": "google", "task": task, "source_index": i, "scaffold": scaffolds[i], "scaffold_seen": scaffolds[i] in seen})

data = pd.read_csv(SOURCE)
mapping = pd.DataFrame(maps)
merged = data.merge(mapping, on=["family", "task", "source_index"], how="left", validate="many_to_one")
if merged["scaffold_seen"].isna().any() or len(merged) != len(data):
    raise SystemExit("SCAFFOLD_MAPPING_FAILED")
merged["scaffold_novelty"] = np.where(merged["scaffold_seen"], "seen", "unseen")
nonnull = merged[merged["n_rationale"] > 0]

overall = nonnull.groupby("scaffold_novelty", as_index=False).agg(
    molecules=("source_index", "size"),
    mean_miss_loss=("miss_loss", "mean"),
    mean_iou=("iou", "mean"),
    mean_selected_fraction=("selected_fraction", "mean"),
    mean_fidelity_advantage=("fidelity_advantage", "mean"),
)
tasks = nonnull.groupby(["family", "task", "scaffold_novelty"], as_index=False).agg(
    molecules=("source_index", "size"),
    mean_miss_loss=("miss_loss", "mean"),
    mean_iou=("iou", "mean"),
    mean_selected_fraction=("selected_fraction", "mean"),
    mean_fidelity_advantage=("fidelity_advantage", "mean"),
)
rates = merged.groupby(["family", "task"], as_index=False).agg(
    molecules=("source_index", "size"),
    unseen_rate=("scaffold_seen", lambda values: 1 - values.mean()),
    unique_test_scaffolds=("scaffold", "nunique"),
)

OUT.mkdir(parents=True, exist_ok=True)
merged.to_csv(OUT / "molecules_with_scaffold.csv.gz", index=False, compression="gzip")
overall.to_csv(OUT / "overall.csv", index=False)
tasks.to_csv(OUT / "tasks.csv", index=False)
rates.to_csv(OUT / "rates.csv", index=False)
summary = {
    "status": "PASS",
    "molecule_rows": len(merged),
    "invalid_scaffold_rows": int((merged["scaffold"] == "<INVALID>").sum()),
    "definition": "unseen means absent from fit plus development; calibration is excluded from model exposure",
    "overall": json.loads(overall.to_json(orient="records")),
    "rates": json.loads(rates.to_json(orient="records")),
}
(OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps(summary, indent=2))
