import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


FILES = [
    "pair_attributions.csv",
    "crc_results.csv",
    "crc_macro_summary.csv",
    "component_crc_feasibility.csv",
    "near_zero_direction_instability.csv",
    "summary.json",
    "attribution_environment.json",
]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-a", required=True)
    parser.add_argument("--run-b", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run_a = Path(args.run_a).resolve()
    run_b = Path(args.run_b).resolve()
    hashes_a = {name: sha256(run_a / name) for name in FILES}
    hashes_b = {name: sha256(run_b / name) for name in FILES}
    exact = {name: hashes_a[name] == hashes_b[name] for name in FILES}
    keys = ["dataset", "split", "pair_type", "pair_id", "seed", "method"]
    attribution_a = pd.read_csv(run_a / "pair_attributions.csv")
    attribution_b = pd.read_csv(run_b / "pair_attributions.csv")
    key_and_score_exact = attribution_a[keys + ["scores_a", "scores_b"]].equals(attribution_b[keys + ["scores_a", "scores_b"]])
    result_a = pd.read_csv(run_a / "crc_results.csv")
    result_b = pd.read_csv(run_b / "crc_results.csv")
    crc_exact = result_a.equals(result_b)
    status = "PASS_EXACT" if all(exact.values()) and key_and_score_exact and crc_exact else "FAIL"
    audit = {
        "status": status,
        "run_a": str(run_a),
        "run_b": str(run_b),
        "files": len(FILES),
        "attribution_rows": len(attribution_a),
        "crc_rows": len(result_a),
        "file_hashes_exact": exact,
        "run_a_sha256": hashes_a,
        "run_b_sha256": hashes_b,
        "attribution_keys_and_serialized_scores_exact": key_and_score_exact,
        "crc_table_exact": crc_exact,
        "boundary": "No tolerance or result-set substitution was used; every formal attribution/CRC artifact is byte-identical across independent CPU processes.",
    }
    Path(args.output).write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))
    if status != "PASS_EXACT":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
