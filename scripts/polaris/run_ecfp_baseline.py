import json
from pathlib import Path

import numpy as np
import pandas as pd
import polaris as po
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from sklearn.ensemble import RandomForestRegressor


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUT = ROOT / "ecfp_rf"
BENCHMARK_ID = "polaris/adme-fang-hclint-1"


def fingerprints(smiles: pd.Series) -> np.ndarray:
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    rows = []
    for value in smiles:
        molecule = Chem.MolFromSmiles(value)
        if molecule is None:
            raise ValueError(f"Invalid SMILES: {value}")
        rows.append(generator.GetFingerprintAsNumPy(molecule))
    return np.asarray(rows, dtype=np.uint8)


OUT.mkdir(parents=True, exist_ok=True)
train = pd.read_csv(DATA / "train.csv")
test = pd.read_csv(DATA / "test.csv")
X_train = fingerprints(train["smiles"])
X_test = fingerprints(test["smiles"])
model = RandomForestRegressor(
    n_estimators=500,
    min_samples_leaf=2,
    n_jobs=-1,
    random_state=42,
)
model.fit(X_train, train["LOG_HLM_CLint"])
predictions = model.predict(X_test)
predictions_csv = OUT / "predictions.csv"
pd.DataFrame(
    {"smiles": test["smiles"], "LOG_HLM_CLint": predictions}
).to_csv(predictions_csv, index=False)
result = po.load_benchmark(BENCHMARK_ID).evaluate(y_pred=predictions)
payload = {
    "model": "ECFP4-2048 RandomForestRegressor",
    "parameters": model.get_params(),
    "benchmark_id": BENCHMARK_ID,
    "results": result.model_dump(mode="json"),
}
(OUT / "results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
print(json.dumps(payload, indent=2))
