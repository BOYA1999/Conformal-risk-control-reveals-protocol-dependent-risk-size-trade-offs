import json
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import polaris as po


ROOT = Path(__file__).resolve().parent
seeds = [42, 123, 2026]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


frames = [pd.read_csv(ROOT / f"chemeleon_seed{seed}" / "predictions.csv") for seed in seeds]
seed_scores = {
    str(seed): json.loads(
        (ROOT / f"chemeleon_seed{seed}" / "polaris_metrics.json").read_text(
            encoding="utf-8"
        )
    )["results"][0]["scores"]
    for seed in seeds
}
if any(not frame["smiles"].equals(frames[0]["smiles"]) for frame in frames[1:]):
    raise ValueError("Seed predictions do not share the frozen test order")
matrix = np.column_stack([frame["LOG_HLM_CLint"] for frame in frames])
predictions = matrix.mean(axis=1)
out_dir = ROOT / "chemeleon_ensemble"
out_dir.mkdir(exist_ok=True)
pd.DataFrame(
    {"smiles": frames[0]["smiles"], "LOG_HLM_CLint": predictions}
).to_csv(out_dir / "predictions.csv", index=False)
ensemble_predictions = out_dir / "predictions.csv"
result = po.load_benchmark("polaris/adme-fang-hclint-1").evaluate(y_pred=predictions)
payload = {
    "status": "PASS",
    "seeds": seeds,
    "rule": "unweighted arithmetic mean fixed before evaluation",
    "seed_scores": seed_scores,
    "artifacts": {
        str(seed): {
            "model_sha256": sha256(ROOT / f"chemeleon_seed{seed}" / "model_0" / "best.pt"),
            "predictions_sha256": sha256(ROOT / f"chemeleon_seed{seed}" / "predictions.csv"),
            "config_sha256": sha256(ROOT / f"chemeleon_seed{seed}" / "config.toml"),
            "splits_sha256": sha256(ROOT / f"chemeleon_seed{seed}" / "splits.json"),
        }
        for seed in seeds
    },
    "seed_mean": {
        metric: float(np.mean([seed_scores[str(seed)][metric] for seed in seeds]))
        for metric in seed_scores[str(seeds[0])]
    },
    "seed_sample_sd": {
        metric: float(np.std([seed_scores[str(seed)][metric] for seed in seeds], ddof=1))
        for metric in seed_scores[str(seeds[0])]
    },
    "results": result.model_dump(mode="json"),
    "ensemble_predictions_sha256": sha256(ensemble_predictions),
    "test_targets_accessed": False,
}
(out_dir / "polaris_metrics.json").write_text(
    json.dumps(payload, indent=2), encoding="utf-8"
)
print(json.dumps(payload, indent=2))
