import sys
sys.dont_write_bytecode = True
import gc
import hashlib
import json
import os
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import numpy as np
import pandas as pd
import torch_geometric

PACKAGE = Path(__file__).resolve().parents[3]
SCIENCE = Path(os.environ.get('MOLXAI_WORK_ROOT', '../molxai-work')).resolve()
HERE = Path(os.environ.get('MOLXAI_GRID_ROOT', SCIENCE / 'revision/grid')).resolve()
if HERE == PACKAGE or PACKAGE in HERE.parents:
    raise SystemExit('Use an external grid workspace.')
HERE.mkdir(parents=True, exist_ok=True)
P20 = SCIENCE / 'full_grid'
P01 = SCIENCE / "artifacts/experiment/gradient_grid_main"
sys.path.insert(0, str(PACKAGE / 'src'))
from run_gradient_grid import GraphClassifier, bxaic_partitions, google_partitions
from run_bxaic_probe import gradient_scores, set_seed

Q = np.arange(101, dtype=np.int64)
MACHINE = np.linspace(0, 1, 101)
ALPHAS = [0.05, 0.10, 0.20]
METHODS = ["ig", "gradinput", "atom_occlusion", "saliency"]
METRICS = ["risk", "mean_atom_fraction", "precision", "iou"]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest().upper()


def save_json(name, data):
    (HERE / name).write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8")


def log(value):
    line = f"{datetime.now(timezone.utc).isoformat()} {value}"
    print(line, flush=True)
    with (HERE / "run.log").open("a", encoding="utf-8") as stream:
        stream.write(line + "\n")


def count_matrix(lengths, grid):
    return np.ceil(lengths[:, None] * MACHINE).astype(np.int64) if grid == "machine" else (lengths[:, None] * Q + 99) // 100


def curves(scores, masks, counts):
    result = {key: np.empty(counts.shape, dtype=float) for key in METRICS}
    for i, (score, truth, selected) in enumerate(zip(scores, masks, counts)):
        order = np.lexsort((np.arange(len(score)), -score.astype(float)))
        ranks = np.flatnonzero(truth[order])
        hits = np.searchsorted(ranks, selected, side="left")
        result["risk"][i] = 1 - hits / len(ranks)
        result["mean_atom_fraction"][i] = selected / len(score)
        result["precision"][i] = np.divide(hits, selected, out=np.zeros(101), where=selected > 0)
        result["iou"][i] = hits / (selected + len(ranks) - hits)
    assert all(np.isfinite(a).all() for a in result.values())
    assert np.all(np.diff(result["risk"], axis=1) <= 1e-12)
    assert np.all(result["risk"][:, 0] == 1) and np.all(result["risk"][:, -1] == 0)
    return result


def score_rows(cache, split, method):
    offsets = cache[f"{split}__offsets"]
    scores = cache[f"{split}__{method}"]
    masks = cache[f"{split}__rationale"].astype(bool)
    return ([scores[a:b] for a, b in zip(offsets[:-1], offsets[1:])],
            [masks[a:b] for a, b in zip(offsets[:-1], offsets[1:])])


def load_gradient_cache(cell_id, parts, p20):
    cache_path = HERE / "scores" / f"{cell_id}.npz"
    if cache_path.exists():
        meta = json.loads(cache_path.with_suffix(".json").read_text())
        assert meta["sha256"] == sha(cache_path)
        return dict(np.load(cache_path, allow_pickle=False))
    identity = json.loads((P01 / "cells" / f"{cell_id}.json").read_text())
    set_seed(identity["seed"])
    model = GraphClassifier(identity["model"], parts["fit"][0].x.shape[1]).cuda()
    model.load_state_dict(torch.load(P01 / "checkpoints" / f"{cell_id}.pt", map_location="cuda", weights_only=True)["state_dict"])
    model.eval()
    cache, timings = {}, {}
    for split in ["calibration", "test"]:
        graphs = parts[split]
        ids = np.array([int(g.source_index) for g in graphs])
        assert np.array_equal(ids, p20[f"{split}__all_source_ids"])
        indices = [i for i, graph in enumerate(graphs) if bool(graph.rationale_mask.any())]
        assert np.array_equal(ids[indices], p20[f"{split}__source_ids"])
        for suffix in ["all_source_ids", "source_ids", "labels", "offsets", "rationale"]:
            cache[f"{split}__{suffix}"] = p20[f"{split}__{suffix}"]
        assert np.array_equal(np.concatenate([graphs[i].rationale_mask.numpy() for i in indices]), cache[f"{split}__rationale"])
        for method, batch in [("gradinput", 128), ("ig", 32)]:
            scores, masks, timing = gradient_scores(model, graphs, torch.device("cuda"), method, batch, 20)
            cache[f"{split}__{method}"] = np.concatenate([scores[i] for i in indices])
            assert np.isfinite(cache[f"{split}__{method}"]).all()
            timings[f"{split}__{method}"] = timing
            log(f"scores {cell_id} {split} {method} {timing['seconds']:.1f}s")
    np.savez_compressed(cache_path, **cache)
    cache_path.with_suffix(".json").write_text(json.dumps({"cell_id": cell_id, "sha256": sha(cache_path), "checkpoint_sha256": sha(P01 / "checkpoints" / f"{cell_id}.pt"), "timings": timings, "runner_sha256": sha(__file__), "score_origin": "P01_checkpoint_rerun"}, indent=2), encoding="utf-8")
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return cache


def analyze_cell(path, parts):
    start = time.perf_counter()
    p20_meta = json.loads(path.read_text())
    cell_id = path.stem
    identity = {k: p20_meta[k] for k in ["cell_id", "family", "task", "model", "seed"]}
    p20_path = P20 / "scores" / f"{cell_id}.npz"
    assert sha(p20_path) == p20_meta["score_cache_sha256"]
    assert sha(P01 / "checkpoints" / f"{cell_id}.pt") == p20_meta["checkpoint_sha256"]
    p20 = dict(np.load(p20_path, allow_pickle=False))
    gradients = load_gradient_cache(cell_id, parts, p20)
    historical = json.loads((P01 / "cells" / path.name).read_text())
    output = {key: [] for key in ["curves", "operating_points", "historical_points", "historical_replay", "grid_effect_cells", "cardinality_counts"]}
    counts = {}
    for split in ["calibration", "test"]:
        lengths = np.diff(p20[f"{split}__offsets"])
        for grid in ["machine", "exact_integer"]:
            counts[split, grid] = count_matrix(lengths, grid)
            assert np.all(np.diff(counts[split, grid], axis=1) >= 0)
            assert np.all(counts[split, grid][:, 0] == 0) and np.array_equal(counts[split, grid][:, -1], lengths)
        affected = counts[split, "machine"] != counts[split, "exact_integer"]
        output["cardinality_counts"].append({**identity, "split": split, "occurrences": len(lengths), "grid_pairs": int(affected.size), "affected_grid_pairs": int(affected.sum()), "affected_occurrences": int(affected.any(axis=1).sum())})
    for method in METHODS:
        origin = "P01_checkpoint_rerun" if method in ["ig", "gradinput"] else "P20_frozen_cache"
        ident = {**identity, "method": method, "score_origin": origin}
        cache = gradients if method in ["ig", "gradinput"] else p20
        all_values, means, selected = {}, {}, {}
        for split in ["calibration", "test"]:
            scores, masks = score_rows(cache, split, method)
            for grid in ["machine", "exact_integer"]:
                values = curves(scores, masks, counts[split, grid])
                all_values[split, grid] = values
                means[split, grid] = {key: value.mean(axis=0) for key, value in values.items()}
                for q in Q:
                    output["curves"].append({**ident, "grid": grid, "split": split, "q": int(q), "nominal_fraction": q / 100, "n": len(scores), **{key: float(value[q]) for key, value in means[split, grid].items()}})
        n = len(counts["calibration", "machine"])
        for alpha in ALPHAS:
            hist = historical["explainers"][method]["metrics"]["alpha"][f"{alpha:.2f}"]["crc"] if method in ["ig", "gradinput"] else p20_meta["methods"][method]["index_tiebreak_v1"]["alpha"][f"{alpha:.2f}"]["crc"]
            history_q = int(hist["index"])
            output["historical_points"].append({**identity, "method": method, "score_origin": "P01_historical" if method in ["ig", "gradinput"] else "P20_historical", "alpha": alpha, "q": history_q, **{key: hist[key] for key in METRICS}})
            for grid in ["machine", "exact_integer"]:
                corrected = (n * means["calibration", grid]["risk"] + 1) / (n + 1)
                valid = np.flatnonzero(corrected <= alpha)
                assert len(valid)
                q = int(valid[0])
                selected[grid, alpha] = q
                output["operating_points"].append({**ident, "grid": grid, "alpha": alpha, "q": q, "n_calibration": n, "n_test": len(counts["test", grid]), "calibration_risk": float(means["calibration", grid]["risk"][q]), "corrected_risk": float(corrected[q]), **{key: float(value[q]) for key, value in means["test", grid].items()}})
            machine_q, exact_q = selected["machine", alpha], selected["exact_integer", alpha]
            replay = {**ident, "alpha": alpha, "historical_q": history_q, "current_q": machine_q, "q_change": machine_q - history_q, **{f"delta_{key}": float(means["test", "machine"][key][machine_q] - hist[key]) for key in METRICS}}
            if origin == "P20_frozen_cache":
                assert all(abs(replay[f"delta_{key}"]) < 1e-10 for key in METRICS) and machine_q == history_q
            output["historical_replay"].append(replay)
            for comparison, qa, qb in [("fixed_historical_q", history_q, history_q), ("fixed_current_machine_q", machine_q, machine_q), ("recalibrated", machine_q, exact_q)]:
                row = {**ident, "alpha": alpha, "comparison": comparison, "machine_q": qa, "exact_q": qb, "q_change": qb - qa}
                for split in ["calibration", "test"]:
                    row[f"{split}_occurrences"] = len(counts[split, "machine"])
                    row[f"{split}_changed_sets"] = int(np.count_nonzero(counts[split, "machine"][:, qa] != counts[split, "exact_integer"][:, qb]))
                for key in METRICS:
                    row[f"machine_{key}"] = float(means["test", "machine"][key][qa])
                    row[f"exact_{key}"] = float(means["test", "exact_integer"][key][qb])
                    row[f"delta_{key}"] = row[f"exact_{key}"] - row[f"machine_{key}"]
                output["grid_effect_cells"].append(row)
    folder = HERE / "cells" / cell_id
    folder.mkdir(exist_ok=True)
    for name, rows in output.items():
        pd.DataFrame(rows).to_csv(folder / f"{name}.csv", index=False)
    save_json(f"cells/{cell_id}/complete.json", {"status": "complete", "cell_id": cell_id, "seconds": time.perf_counter() - start, "runner_sha256": sha(__file__), "p20_cache_sha256": sha(p20_path), "gradient_cache_sha256": sha(HERE / "scores" / f"{cell_id}.npz")})
    log(f"cell_complete {cell_id} {time.perf_counter()-start:.1f}s")


def aggregate():
    complete = sorted((HERE / "cells").glob("*/complete.json"))
    for name in ["curves", "operating_points", "historical_points", "historical_replay", "grid_effect_cells", "cardinality_counts"]:
        pd.concat([pd.read_csv(p.parent / f"{name}.csv") for p in complete], ignore_index=True).to_csv(HERE / f"{name}.csv", index=False)
    save_json("progress.json", {"status": "complete" if len(complete) == 66 else "partial", "cells": len(complete), "expected_cells": 66})


def main():
    if (HERE / 'environment.json').exists():
        assert json.loads((HERE / 'environment.json').read_text())['runner_sha256'] == sha(__file__), 'Use a fresh grid workspace for a different runner.'
    shutil.copy2(__file__, HERE / 'run_grid.py')
    (HERE / "scores").mkdir(exist_ok=True)
    (HERE / "cells").mkdir(exist_ok=True)
    torch.set_num_threads(1)
    torch.cuda.set_per_process_memory_fraction(0.8)
    sys.path.insert(0, str(PACKAGE / 'scripts/full_grid'))
    import run_full_grid as full_grid_audit
    full_grid_audit.HERE = HERE
    audit_rows = full_grid_audit.source_audit()
    audit = {'status': 'PASS', 'checked_files': len(audit_rows), 'rows': audit_rows}
    save_json("input_audit.json", audit)
    save_json("environment.json", {"python": sys.version, "executable": sys.executable, "torch": torch.__version__, "numpy": np.__version__, "pandas": pd.__version__, "pyg": torch_geometric.__version__, "gpu": torch.cuda.get_device_name(0), "memory_fraction": 0.8, "runner_sha256": sha(__file__), "plan_sha256": sha(PACKAGE / 'contracts/revision/grid_sensitivity.json')})
    paths = sorted((P20 / "cells").glob("*.json"))
    assert len(paths) == 66
    current_task, parts = None, None
    for path in paths:
        if (HERE / "cells" / path.stem / "complete.json").exists():
            continue
        family, task, _, _ = path.stem.split("__")
        if current_task != (family, task):
            log(f"load_task {family}/{task}")
            parts = bxaic_partitions(SCIENCE / "data/raw/bxaic/data.csv", SCIENCE / "data/raw/bxaic/explanations.sdf", task) if family == "bxaic" else google_partitions(SCIENCE / "reference/graph-attribution/data", task)
            current_task = family, task
        analyze_cell(path, parts)
        aggregate()
    log("full_complete")


if __name__ == "__main__":
    main()
