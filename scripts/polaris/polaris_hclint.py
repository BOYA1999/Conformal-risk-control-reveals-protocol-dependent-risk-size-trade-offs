import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import polaris as po


BENCHMARK_ID = "polaris/adme-fang-hclint-1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    benchmark = po.load_benchmark(BENCHMARK_ID)
    train, test = benchmark.get_train_test_split()
    train_csv = out_dir / "train.csv"
    test_csv = out_dir / "test.csv"
    train.as_dataframe().to_csv(train_csv, index=False)
    test.as_dataframe().to_csv(test_csv, index=False)
    metadata = {
        "benchmark_id": BENCHMARK_ID,
        "benchmark_md5sum": benchmark.md5sum,
        "dataset_artifact_id": benchmark.dataset.artifact_id,
        "dataset_license": str(benchmark.dataset.license),
        "dataset_source": str(benchmark.dataset.source),
        "polaris_version": po.__version__,
        "target": str(next(iter(benchmark.target_cols))),
        "main_metric": str(benchmark.main_metric),
        "metrics": sorted(str(metric.label) for metric in benchmark.metrics),
        "n_train": len(train),
        "n_test": len(test),
        "train_sha256": sha256(train_csv),
        "test_sha256": sha256(test_csv),
        "test_targets_accessed": False,
    }
    (out_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))


def evaluate(predictions_csv: Path, out_json: Path) -> None:
    frame = pd.read_csv(predictions_csv)
    expected = pd.read_csv(Path(__file__).resolve().parent / "data" / "test.csv")
    if "smiles" in frame and not frame["smiles"].equals(expected["smiles"]):
        raise ValueError("Prediction rows do not match the frozen Polaris test order")
    target = "LOG_HLM_CLint"
    if target not in frame:
        candidates = [column for column in frame if column != "smiles"]
        if len(candidates) != 1:
            raise ValueError(f"Cannot identify prediction column: {list(frame.columns)}")
        target = candidates[0]
    predictions = frame[target].to_numpy(dtype=float)
    if len(predictions) != 575 or not np.isfinite(predictions).all():
        raise ValueError("Expected 575 finite predictions")
    result = po.load_benchmark(BENCHMARK_ID).evaluate(y_pred=predictions)
    payload = result.model_dump(mode="json")
    payload["predictions_csv"] = str(predictions_csv.resolve())
    payload["predictions_sha256"] = sha256(predictions_csv)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--out-dir", type=Path, required=True)
    evaluate_parser = subparsers.add_parser("evaluate")
    evaluate_parser.add_argument("--predictions-csv", type=Path, required=True)
    evaluate_parser.add_argument("--out-json", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.out_dir)
    else:
        evaluate(args.predictions_csv, args.out_json)
