import os
import csv
import hashlib
import io
import json
import pickle
import pickletools
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
from rdkit import Chem

CODE = Path(__file__).resolve().parent
PACKAGE = CODE.parents[1]
WORK = Path(os.environ.get('MOLXAI_WORK_ROOT', '../molxai-work')).resolve()
BASE = WORK / 'independent_reference/data/raw'
ALLOWED = {
    ("numpy.core.multiarray", "_reconstruct"): np.core.multiarray._reconstruct,
    ("numpy", "ndarray"): np.ndarray,
    ("numpy", "dtype"): np.dtype,
}


class ArrayReader(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) not in ALLOWED:
            raise pickle.UnpicklingError(f"Disallowed global {module}.{name}")
        return ALLOWED[module, name]


def read_masks():
    expected = {
        'Liver.csv': '255054c8030712c094de5f88484311d006a667ae6565a2879c417d5cd47d533f',
        'attributions.npz': 'd2abe69c7e5d1f7fb1f2bb77029ed6d01492ef63c744c14ca11141660672383c',
    }
    for name, checksum in expected.items():
        assert hashlib.sha256((BASE / name).read_bytes()).hexdigest() == checksum, f'Source hash mismatch: {name}'
    with zipfile.ZipFile(BASE / "attributions.npz") as archive:
        assert archive.namelist() == ["attributions.npy"]
        stream = io.BytesIO(archive.read("attributions.npy"))
    assert np.lib.format.read_magic(stream) == (1, 0)
    shape, fortran, dtype = np.lib.format.read_array_header_1_0(stream)
    assert shape == (587,) and not fortran and dtype == np.dtype("O")
    payload = stream.read()
    globals_seen = {tuple(arg.split(" ")) for op, arg, _ in pickletools.genops(payload) if op.name == "GLOBAL"}
    assert globals_seen <= set(ALLOWED)
    return ArrayReader(io.BytesIO(payload)).load()


if __name__ == "__main__":
    rows = list(csv.DictReader((BASE / "Liver.csv").open(encoding="utf-8-sig")))
    masks = read_masks()
    assert len(rows) == len(masks) == 587
    summary = Counter()
    problems = []
    for i, (row, record) in enumerate(zip(rows, masks)):
        mol = Chem.MolFromSmiles(row["SMILES"])
        values = np.asarray(record["node_atts"])
        summary[(row["SPLIT"], row["label"], bool(values.sum()))] += 1
        if mol is None or values.shape[0] != mol.GetNumAtoms():
            problems.append((i, None if mol is None else mol.GetNumAtoms(), values.shape))
    canonical = [Chem.MolToSmiles(Chem.MolFromSmiles(row["SMILES"])) for row in rows]
    result = {
        "rows": len(rows),
        "count_by_split_label_nonnull": {str(k): v for k, v in summary.items()},
        "mask_shape_problems": problems,
        "keys": sorted({k for record in masks for k in record}),
        "mask_value_counts": dict(Counter(float(v) for record in masks for v in np.asarray(record["node_atts"]).ravel())),
        "npz_csv_smiles_exact_agreement": sum(row["SMILES"] == record["SMILES"] for row, record in zip(rows, masks)),
        "npz_original_class_agreement": sum(int(row["Hepatotoxicity Class"]) == record["label"] for row, record in zip(rows, masks)),
        "csv_shifted_label_agreement": sum(int(row["label"]) == record["label"] + 1 for row, record in zip(rows, masks)),
        "unique_isomeric_canonical_smiles": len(set(canonical)),
        "official_split_overlap": len({c for c, r in zip(canonical, rows) if r["SPLIT"] == "train"} & {c for c, r in zip(canonical, rows) if r["SPLIT"] == "test"}),
    }
    source = WORK
    comparisons = {"bxaic": source / "data/raw/bxaic/data.csv"}
    comparisons.update({"google_" + task: source / f"reference/graph-attribution/data/{task}/{task}_smiles.csv" for task in ["benzene", "logic7", "logic8", "logic10"]})
    result["primary_family_overlap"] = {}
    for name, path in comparisons.items():
        with path.open(encoding="utf-8-sig") as stream:
            other_rows = list(csv.DictReader(stream))
        smiles_key = next(k for k in other_rows[0] if k.lower() == "smiles")
        other = set()
        permissive_parse_count = 0
        for r in other_rows:
            mol = Chem.MolFromSmiles(r[smiles_key])
            if mol is None:
                mol = Chem.MolFromSmiles(r[smiles_key], sanitize=False)
                permissive_parse_count += 1
            assert mol is not None
            other.add(Chem.MolToSmiles(mol))
        indices = [i for i, c in enumerate(canonical) if c in other]
        result["primary_family_overlap"][name] = {"overlap_count": len(indices), "liver_source_indices": indices, "other_permissive_parse_count": permissive_parse_count}
    print(json.dumps(result, indent=2))
