import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from Levenshtein import distance
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds.MurckoScaffold import GetScaffoldForMol, MakeScaffoldGeneric


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonicalize(smiles):
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def standardize(path):
    raw = pd.read_csv(path)
    raw["canonical_smiles"] = raw["smiles"].map(canonicalize)
    raw["activity"] = pd.to_numeric(raw["y [pEC50/pKi]"], errors="coerce")
    raw["exp_nm"] = pd.to_numeric(raw["exp_mean [nM]"], errors="coerce")
    valid = raw.dropna(subset=["canonical_smiles", "activity", "exp_nm"]).copy()
    ranges = valid.groupby("canonical_smiles")["activity"].agg(lambda x: x.max() - x.min())
    conflicts = set(ranges[ranges > 0.2].index)
    usable = valid[~valid["canonical_smiles"].isin(conflicts)].copy()
    overlap = set(usable.loc[usable["split"] == "train", "canonical_smiles"]) & set(
        usable.loc[usable["split"] == "test", "canonical_smiles"]
    )
    rows = []
    for canonical, group in usable.groupby("canonical_smiles", sort=True):
        splits = sorted(group["split"].unique())
        if len(splits) != 1:
            continue
        rows.append(
            {
                "source_smiles": group.iloc[0]["smiles"],
                "canonical_smiles": canonical,
                "activity": float(group["activity"].median()),
                "exp_nm": float(group["exp_nm"].median()),
                "official_split": splits[0],
                "source_rows": int(len(group)),
            }
        )
    clean = pd.DataFrame(rows)
    summary = {
        "source_rows": int(len(raw)),
        "valid_rows": int(len(valid)),
        "valid_fraction": float(len(valid) / len(raw)),
        "conflicting_canonical_groups": int(len(conflicts)),
        "canonical_train_test_overlap": int(len(overlap)),
        "unique_molecules": int(len(clean)),
    }
    return clean, summary


def fingerprints(smiles):
    mols = [Chem.MolFromSmiles(s) for s in smiles]
    morgan = [AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=1024) for m in mols]
    scaffold = []
    for mol in mols:
        try:
            core = MakeScaffoldGeneric(mol)
        except Exception:
            core = GetScaffoldForMol(mol)
        scaffold.append(AllChem.GetMorganFingerprintAsBitVect(core, 2, nBits=1024))
    return morgan, scaffold


def cliff_pairs(frame):
    smiles = frame["canonical_smiles"].tolist()
    exp_nm = frame["exp_nm"].to_numpy(float)
    morgan, scaffold = fingerprints(smiles)
    pairs = []
    for i in range(len(frame) - 1):
        tani = DataStructs.BulkTanimotoSimilarity(morgan[i], morgan[i + 1 :])
        scaf = DataStructs.BulkTanimotoSimilarity(scaffold[i], scaffold[i + 1 :])
        for offset, (tani_value, scaf_value) in enumerate(zip(tani, scaf), start=1):
            j = i + offset
            lev = 1.0 - distance(smiles[i], smiles[j]) / max(len(smiles[i]), len(smiles[j]))
            similar = tani_value >= 0.9 or scaf_value >= 0.9 or lev >= 0.9
            fold = max(exp_nm[i], exp_nm[j]) / min(exp_nm[i], exp_nm[j])
            if similar and fold > 10.0:
                pairs.append((i, j))
    return pairs


def choose_datasets(qc, quantiles):
    candidates = qc[
        (qc["unique_molecules"] >= 1000)
        & (qc["valid_fraction"] >= 0.98)
        & (qc["canonical_train_test_overlap"] == 0)
        & (qc["test_cliff_pairs"] >= 30)
    ].copy()
    if len(candidates) < 3:
        raise RuntimeError(f"Only {len(candidates)} datasets meet the frozen eligibility rules")
    candidates["size_rank"] = candidates["unique_molecules"].rank(method="average")
    candidates["cliff_rank"] = candidates["test_cliff_pairs"].rank(method="average")
    candidates["composite_rank"] = (candidates["size_rank"] + candidates["cliff_rank"]) / 2
    candidates = candidates.sort_values(["composite_rank", "dataset"]).reset_index(drop=True)
    positions = [int(np.floor(q * (len(candidates) - 1) + 0.5)) for q in quantiles]
    selected = candidates.iloc[positions].copy()
    if selected["dataset"].nunique() != 3:
        raise RuntimeError("Frozen quantile rule did not produce three distinct datasets")
    return candidates, selected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    processed_dir = output_dir / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    frames = {}
    qc_rows = []
    for path in sorted(data_dir.glob("CHEMBL*.csv")):
        frame, summary = standardize(path)
        test_pairs = cliff_pairs(frame[frame["official_split"] == "test"].reset_index(drop=True))
        summary.update(
            {
                "dataset": path.stem,
                "test_molecules": int((frame["official_split"] == "test").sum()),
                "test_cliff_pairs": int(len(test_pairs)),
                "source_sha256": sha256(path),
            }
        )
        frames[path.stem] = frame
        qc_rows.append(summary)
        print(f"QC {path.stem}: n={summary['unique_molecules']} test_pairs={len(test_pairs)}", flush=True)
    qc = pd.DataFrame(qc_rows).sort_values("dataset")
    candidates, selected = choose_datasets(qc, config["dataset_selection_quantiles"])
    qc.to_csv(output_dir / "dataset_qc.csv", index=False)
    candidates.to_csv(output_dir / "dataset_candidates.csv", index=False)
    for dataset in selected["dataset"]:
        frames[dataset].to_csv(processed_dir / f"{dataset}.csv", index=False)
    selection = {
        "run_id": config["run_id"],
        "selection_rule": "eligible composite-rank quantiles 0.25, 0.50, 0.75",
        "selected": selected.to_dict(orient="records"),
        "moleculeace_commit": config["moleculeace_commit"],
    }
    (output_dir / "dataset_selection.json").write_text(json.dumps(selection, indent=2), encoding="utf-8")
    print("SELECTED " + ", ".join(selected["dataset"]), flush=True)


if __name__ == "__main__":
    main()
