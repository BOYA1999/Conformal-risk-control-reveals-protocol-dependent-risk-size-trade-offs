import os
import hashlib
import json
from pathlib import Path

CODE = Path(__file__).resolve().parent
PACKAGE = CODE.parents[1]
WORK = Path(os.environ.get('MOLXAI_WORK_ROOT', '../molxai-work')).resolve()
HERE = WORK / 'full_grid'
SCIENCE = WORK
CONTRACT = PACKAGE / 'contracts/full_grid/run_contract.json'



def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest().upper()


def main():
    expected = {path.stem for path in (SCIENCE / "artifacts/experiment/gradient_grid_main/cells").glob("*.json")}
    actual = {path.stem for path in (HERE / "cells").glob("*.json")}
    caches = {path.stem for path in (HERE / "scores").glob("*.npz")}
    assert expected == actual == caches and len(expected) == 66
    inputs = json.loads((HERE / "input_audit.json").read_text())
    for row in inputs["rows"]:
        assert sha(Path(row["path"])) == row["expected"]
    env = json.loads((HERE / "environment_full.json").read_text())
    assert sha(CODE / "run_full_grid.py") == env["runner_sha256"]
    assert sha(CONTRACT) == env["contract_sha256"]
    assert sha(CODE / "frozen_index_policy.py") == env["selector_sha256"]
    for filename, checksum in env["imported_sources"].items():
        assert sha(PACKAGE / "src" / filename) == checksum
    for name in ["validation.json", "score_replay_validation.json"]:
        assert json.loads((HERE / name).read_text())["status"] == "PASS"
    validation = json.loads((HERE / "validation.json").read_text())
    scores = json.loads((HERE / "score_replay_validation.json").read_text())
    assert validation["cells"] == scores["cell_count"] == 66
    assert validation["eligible_calibration_occurrences"] == 82866
    assert validation["eligible_test_occurrences"] == 81672
    assert scores["method_molecule_checks"] == 396
    assert len(scores["full_source_correspondence"]) == 132
    for path in (HERE / "cells").glob("*.json"):
        data = json.loads(path.read_text())
        assert data["cell_id"] == path.stem and data["status"] == "complete"
        assert data["runner_sha256"] == env["runner_sha256"]
        assert data["contract_sha256"] == env["contract_sha256"]
        assert data["score_cache_sha256"] == sha(HERE / "scores" / f"{path.stem}.npz")
    payload = {"status": "PASS", "exact_expected_cells": 66, "exact_expected_caches": 66, "inputs_rechecked": len(inputs["rows"]), "import_and_selector_hashes_unchanged": True, "all_metric_replays_pass": True, "all_sampled_score_replays_pass": True}
    (HERE / "seal.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    paths = sorted(path for path in HERE.rglob("*") if path.is_file() and "__pycache__" not in path.parts and path.name != "MANIFEST.sha256")
    assert not any(path.suffix == ".tmp" for path in paths)
    (HERE / "MANIFEST.sha256").write_text("".join(f"{sha(path)}  {path.relative_to(HERE).as_posix()}\n" for path in paths), encoding="utf-8")
    print(json.dumps(payload))


if __name__ == "__main__":
    main()
