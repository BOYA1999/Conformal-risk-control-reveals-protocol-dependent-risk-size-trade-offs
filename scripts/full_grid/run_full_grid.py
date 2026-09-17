import os
import argparse
import gc
import hashlib
import importlib.util
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import numpy as np
import pandas as pd
import torch_geometric
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader

CODE = Path(__file__).resolve().parent
PACKAGE = CODE.parents[1]
WORK = Path(os.environ.get('MOLXAI_WORK_ROOT', '../molxai-work')).resolve()
HERE = WORK / 'full_grid'
SCIENCE = WORK
CONTRACT = PACKAGE / 'contracts/full_grid/run_contract.json'
ROOT = WORK
BASE = ROOT / 'artifacts/experiment/gradient_grid_main'
SUBSET = ROOT / 'experiments/established_subset'
sys.path.insert(0, str(PACKAGE / 'src'))
from run_gradient_grid import BXAIC_TASKS, GOOGLE_TASKS, GraphClassifier, bxaic_partitions, google_partitions
from run_bxaic_probe import hash_order
from frozen_index_policy import calibrate_crc, loss_table, top_fraction_set

FRACTIONS = np.linspace(0, 1, 101)
METHODS = ["atom_occlusion", "saliency"]
POLICIES = ["index_tiebreak_v1", "include_all_exact_ties_v2"]
DEVICE = torch.device("cuda")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest().upper()


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8")
    temp.replace(path)


def log(message):
    line = f"{datetime.now(timezone.utc).isoformat()} {message}"
    print(line, flush=True)
    with (HERE / "run.log").open("a", encoding="utf-8") as stream:
        stream.write(line + "\n")


def source_audit():
    paths = {
        ROOT / "data/raw/bxaic/data.csv": "14853568ECE75E5C5666C7190E6402CE8D24B3DD3DB171460E668C82C06344FC",
        ROOT / "data/raw/bxaic/explanations.sdf": "83C86A6366AFF1DB12A3E439CEAB9A4F275508FAF1BEB5E1B8ADB261465E70AA",
        PACKAGE / "src/run_gradient_grid.py": "5CE5AFA8E59CA6F7F8021B4D55FCCFE9A5DFA383349AE404BD196F195A5EB5E4",
        PACKAGE / "src/molxai_crc.py": "4038525E78D85A7CEDE507D91E342CFF7DC79A73213A339E6C4192BF331DE9E7",
    }
    google = json.loads((ROOT / "artifacts/intake/graph_attribution_audit.json").read_text())
    for task, payload in google["tasks"].items():
        for filename, digest in payload["hashes"].items():
            paths[ROOT / "reference/graph-attribution/data" / task / filename] = digest
    for path in sorted((SUBSET / "cells").glob("*.json")):
        payload = json.loads(path.read_text())
        paths[BASE / "checkpoints" / f"{payload['cell_id']}.pt"] = payload["checkpoint_sha256"]
    rows = [{"path": str(p), "expected": expected.upper(), "observed": sha(p)} for p, expected in paths.items()]
    assert all(row["expected"] == row["observed"] for row in rows), "input checksum mismatch"
    assert len(list((BASE / "checkpoints").glob("*.pt"))) == 66
    write_json(HERE / "input_audit.json", {"status": "PASS", "checked_files": len(rows), "rows": rows})
    return rows


def load_parts(family, task):
    if family == "bxaic":
        return bxaic_partitions(ROOT / "data/raw/bxaic/data.csv", ROOT / "data/raw/bxaic/explanations.sdf", task)
    return google_partitions(ROOT / "reference/graph-attribution/data", task)


def eligible(graphs):
    return [g for g in graphs if bool(g.rationale_mask.any())]


def load_model(cell_id, in_channels, model_kind):
    model = GraphClassifier(model_kind, in_channels).to(DEVICE)
    payload = torch.load(BASE / "checkpoints" / f"{cell_id}.pt", map_location=DEVICE, weights_only=True)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model


def saliency(model, graphs, batch_size=64):
    rows = []
    for batch in DataLoader(graphs, batch_size=batch_size, shuffle=False):
        batch = batch.to(DEVICE)
        x = batch.x.detach().clone().requires_grad_(True)
        logits = model(x, batch.edge_index, batch.batch)
        target = logits[torch.arange(batch.num_graphs, device=DEVICE), batch.y].sum()
        values = torch.autograd.grad(target, x)[0].abs().sum(-1).detach().cpu().numpy()
        ptr = batch.ptr.cpu().tolist()
        rows.extend(values[a:b].copy() for a, b in zip(ptr[:-1], ptr[1:]))
    return rows


@torch.no_grad()
def occlusion(model, graphs, solo=False, progress=""):
    rows = []
    for i, graph in enumerate(graphs):
        source = graph.clone().to(DEVICE)
        bi = torch.zeros(source.num_nodes, dtype=torch.long, device=DEVICE)
        target = int(source.y)
        base = model(source.x, source.edge_index, bi)[0, target]
        variants = []
        for atom in range(source.num_nodes):
            variant = source.clone()
            variant.x[atom] = 0
            variants.append(variant)
        if solo:
            perturbed = torch.stack([model(v.x, v.edge_index, bi)[0, target] for v in variants])
        else:
            batch = Batch.from_data_list(variants)
            perturbed = model(batch.x, batch.edge_index, batch.batch)[:, target]
        rows.append((base - perturbed).cpu().numpy())
        if progress and (i + 1) % 1000 == 0:
            log(f"{progress} {i + 1}/{len(graphs)}")
    return rows


def curves(score_rows, masks, policy):
    values = {key: [] for key in ["risk", "mean_atom_fraction", "precision", "iou", "tie_inflation"]}
    for scores, mask in zip(score_rows, masks):
        scores = np.asarray(scores, dtype=float)
        mask = np.asarray(mask, dtype=bool)
        assert scores.ndim == 1 and np.isfinite(scores).all() and len(scores) == len(mask) and mask.any()
        order = np.lexsort((np.arange(len(scores)), -scores))
        nominal = np.where(FRACTIONS == 0, 0, np.ceil(FRACTIONS * len(scores))).astype(int)
        counts = nominal.copy()
        if policy == POLICIES[1]:
            counts[1:] = np.searchsorted(-scores[order], -scores[order][nominal[1:] - 1], side="right")
        hits = np.r_[0, np.cumsum(mask[order])][counts]
        values["risk"].append(1 - hits / mask.sum())
        values["mean_atom_fraction"].append(counts / len(scores))
        values["precision"].append(np.divide(hits, counts, out=np.zeros(len(counts)), where=counts > 0))
        values["iou"].append(hits / (counts + mask.sum() - hits))
        values["tie_inflation"].append((counts - nominal) / len(scores))
    return {k: np.asarray(v) for k, v in values.items()}


def summarize_scores(cal_scores, test_scores, cal_masks, test_masks, policy):
    cal, test = curves(cal_scores, cal_masks, policy), curves(test_scores, test_masks, policy)
    mean = {key: values.mean(0) for key, values in test.items()}
    def point(index):
        return {**{key: float(values[index]) for key, values in mean.items()}, "median_atom_fraction": float(np.median(test["mean_atom_fraction"][:, index]))}
    result = {"n_calibration_rationale": len(cal_scores), "n_test_rationale": len(test_scores), "fixed": {f"{f:.2f}": point(round(f * 100)) for f in [.2, .5]}, "alpha": {}}
    for alpha in [.05, .1, .2]:
        chosen = calibrate_crc(cal["risk"], FRACTIONS, alpha)
        naive = int(np.flatnonzero(cal["risk"].mean(0) <= alpha)[0])
        result["alpha"][f"{alpha:.2f}"] = {
            "crc": {**chosen, **point(chosen["index"])},
            "naive_calibration": {"fraction": float(FRACTIONS[naive]), "calibration_risk": float(cal["risk"][:, naive].mean()), **point(naive)},
        }
    return result


def policy_check():
    spec = importlib.util.spec_from_file_location("legacy_selector", PACKAGE / "src/molxai_crc.py")
    legacy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(legacy)
    rng = np.random.default_rng(20260912)
    scores, masks = [], []
    for size in [1, 2, 7, 23, 51]:
        for _ in range(8):
            scores.append(rng.integers(-2, 3, size).astype(float))
            mask = rng.random(size) < .35
            mask[0] = True
            masks.append(mask)
    truths = [set(np.flatnonzero(m).tolist()) for m in masks]
    expected = legacy.loss_table(scores, truths, FRACTIONS)
    assert np.array_equal(expected, loss_table(scores, truths, FRACTIONS))
    assert np.array_equal(expected, curves(scores, masks, POLICIES[0])["risk"])
    for score in scores:
        for f in FRACTIONS:
            assert top_fraction_set(score, f) == legacy.top_fraction_set(score, f)
    for policy in POLICIES:
        table = curves(scores, masks, policy)
        assert (np.diff(table["risk"], axis=1) <= 1e-12).all()
        assert (table["risk"][:, 0] == 1).all() and (table["risk"][:, -1] == 0).all()
    return {"status": "PASS", "molecules": len(scores), "fractions": len(FRACTIONS), "legacy_loss_exact": True, "legacy_sets_exact": True}


def pilot():
    output, started = [], time.perf_counter()
    for family, task in [("bxaic", "B"), ("google", "benzene")]:
        parts = load_parts(family, task)
        graphs = sorted(eligible(parts["dev"]), key=lambda g: hash_order(int(g.source_index)))[:24]
        for model_kind in ["gin", "gcn"]:
            cell_id = f"{family}__{task}__{model_kind}__seed42"
            model = load_model(cell_id, graphs[0].x.shape[1], model_kind)
            torch.cuda.reset_peak_memory_stats(DEVICE)
            torch.cuda.synchronize()
            begin = time.perf_counter()
            scores = occlusion(model, graphs)
            torch.cuda.synchronize()
            seconds = time.perf_counter() - begin
            solo = occlusion(model, graphs[:3], solo=True)
            sal = saliency(model, graphs)
            sal_solo = saliency(model, graphs[:3], batch_size=1)
            occ_delta = max(float(np.max(np.abs(a - b))) for a, b in zip(scores[:3], solo))
            sal_delta = max(float(np.max(np.abs(a - b))) for a, b in zip(sal[:3], sal_solo))
            assert all(np.allclose(a, b, atol=1e-4, rtol=1e-5) for a, b in zip(scores[:3], solo)), f"occlusion parity failed {cell_id}"
            assert all(np.allclose(a, b, atol=1e-4, rtol=1e-5) for a, b in zip(sal[:3], sal_solo)), f"saliency parity failed {cell_id}"
            output.append({"cell_id": cell_id, "dev_source_ids": [int(g.source_index) for g in graphs], "molecules": len(graphs), "atoms": sum(g.num_nodes for g in graphs), "seconds": seconds, "seconds_per_molecule": seconds / len(graphs), "occlusion_solo_max_abs_delta": occ_delta, "saliency_solo_max_abs_delta": sal_delta, "peak_vram_mib": torch.cuda.max_memory_allocated(DEVICE) / 2**20})
            log(f"pilot_complete {cell_id} {seconds:.3f}s")
        del parts, graphs, model
        gc.collect()
        torch.cuda.empty_cache()
    projection = max(row["seconds_per_molecule"] for row in output) * 164538 * 1.5
    result = {"status": "PASS", "auxiliary_only": True, "rows": output, "elapsed_seconds": time.perf_counter() - started, "projection_seconds_with_1_5_safety": projection, "expected_eligible_occurrences": 164538, "projection_below_30_days": projection < 30 * 86400, "policy_check": policy_check()}
    write_json(HERE / "pilot/result.json", result)
    return result


def partition_metadata(parts, cell_id):
    baseline = json.loads((BASE / "cells" / f"{cell_id}.json").read_text())
    metrics = baseline["explainers"]["ig"]["metrics"]
    ids = {name: [int(g.source_index) for g in graphs] for name, graphs in parts.items()}
    assert len(set(sum(ids.values(), []))) == sum(map(len, ids.values()))
    result = {}
    for split in ["calibration", "test"]:
        full, selected = parts[split], eligible(parts[split])
        assert len(full) == baseline["explainers"]["ig"][f"{split}_timing"]["molecules"]
        assert len(selected) == metrics[f"n_{split}_rationale"]
        for g in full:
            assert g.num_nodes == g.x.shape[0] == g.rationale_mask.numel()
            assert g.edge_index.numel() == 0 or (int(g.edge_index.min()) >= 0 and int(g.edge_index.max()) < g.num_nodes)
        result[split] = {"all_count": len(full), "eligible_count": len(selected), "null_count": len(full) - len(selected), "all_id_sha256": hashlib.sha256(np.asarray(ids[split], dtype="<i8").tobytes()).hexdigest().upper()}
    return result


def run_cell(family, task, kind, seed, parts):
    cell_id = f"{family}__{task}__{kind}__seed{seed}"
    path = HERE / "cells" / f"{cell_id}.json"
    if path.exists():
        old = json.loads(path.read_text())
        if old["status"] == "complete":
            assert old["contract_sha256"] == sha(CONTRACT) and old["runner_sha256"] == sha(__file__)
            assert old["score_cache_sha256"] == sha(HERE / "scores" / f"{cell_id}.npz")
            log(f"skip_complete {cell_id}")
            return
    start = time.perf_counter()
    counts = partition_metadata(parts, cell_id)
    model = load_model(cell_id, parts["calibration"][0].x.shape[1], kind)
    cache, split_data, timings = {}, {}, {}
    torch.cuda.reset_peak_memory_stats(DEVICE)
    for split in ["calibration", "test"]:
        graphs = eligible(parts[split])
        masks = [g.rationale_mask.numpy() for g in graphs]
        cache[f"{split}__all_source_ids"] = np.asarray([int(g.source_index) for g in parts[split]], dtype=np.int64)
        cache[f"{split}__source_ids"] = np.asarray([int(g.source_index) for g in graphs], dtype=np.int64)
        cache[f"{split}__labels"] = np.asarray([int(g.y) for g in graphs], dtype=np.int64)
        cache[f"{split}__offsets"] = np.r_[0, np.cumsum([g.num_nodes for g in graphs])]
        cache[f"{split}__rationale"] = np.concatenate(masks)
        methods = {}
        for method, function in [("atom_occlusion", occlusion), ("saliency", saliency)]:
            torch.cuda.synchronize()
            begin = time.perf_counter()
            methods[method] = function(model, graphs, **({"progress": f"{cell_id} {split}"} if method == "atom_occlusion" else {}))
            torch.cuda.synchronize()
            timings[f"{split}__{method}"] = time.perf_counter() - begin
            assert all(np.isfinite(s).all() and len(s) == g.num_nodes for s, g in zip(methods[method], graphs))
            cache[f"{split}__{method}"] = np.concatenate(methods[method])
        split_data[split] = (methods, masks)
    (HERE / "scores").mkdir(exist_ok=True)
    cache_path = HERE / "scores" / f"{cell_id}.npz"
    np.savez_compressed(cache_path, **cache)
    results = {}
    for method in METHODS:
        results[method] = {}
        for policy in POLICIES:
            results[method][policy] = summarize_scores(split_data["calibration"][0][method], split_data["test"][0][method], split_data["calibration"][1], split_data["test"][1], policy)
    payload = {"status": "complete", "cell_id": cell_id, "family": family, "task": task, "model": kind, "seed": seed, "counts": counts, "methods": results, "timing_seconds": timings, "elapsed_seconds": time.perf_counter() - start, "peak_vram_mib": torch.cuda.max_memory_allocated(DEVICE) / 2**20, "checkpoint_sha256": sha(BASE / "checkpoints" / f"{cell_id}.pt"), "contract_sha256": sha(CONTRACT), "runner_sha256": sha(__file__), "score_cache_sha256": sha(cache_path)}
    write_json(path, payload)
    log(f"cell_complete {cell_id} elapsed={payload['elapsed_seconds']:.1f}s")
    del model, split_data, cache, results
    gc.collect()
    torch.cuda.empty_cache()


def aggregate():
    rows = []
    for path in sorted((HERE / "cells").glob("*.json")):
        p = json.loads(path.read_text())
        for method in METHODS:
            for policy in POLICIES:
                for alpha, result in p["methods"][method][policy]["alpha"].items():
                    rows.append({"cell_id": p["cell_id"], "family": p["family"], "task": p["task"], "model": p["model"], "seed": p["seed"], "method": method, "policy": policy, "alpha": float(alpha), "n_calibration": p["counts"]["calibration"]["eligible_count"], "n_test": p["counts"]["test"]["eligible_count"], **result["crc"]})
    frame = pd.DataFrame(rows)
    frame.to_csv(HERE / "cells.csv", index=False)
    summaries = []
    for (method, policy, alpha), group in frame.groupby(["method", "policy", "alpha"]):
        task = group.groupby(["family", "task"])[["risk", "mean_atom_fraction", "precision", "iou", "tie_inflation"]].mean()
        summaries.append({"method": method, "policy": policy, "alpha": alpha, "cells": len(group), **{f"macro_task_{key}": float(value) for key, value in task.mean().items()}, "risk_pass_cells": int((group.risk <= alpha).sum()), "efficiency_pass_cells": int((group.mean_atom_fraction < .8).sum()), "joint_pass_cells": int(((group.risk <= alpha) & (group.mean_atom_fraction < .8)).sum())})
    write_json(HERE / "summary.json", {"status": "complete" if frame.cell_id.nunique() == 66 else "partial", "cells": int(frame.cell_id.nunique()), "results": summaries, "claim_boundary": "descriptive retrospective full-partition extension using original checkpoints; historical gradient baseline results are not fresh same-run reproductions"})
    return frame


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["pilot", "full", "aggregate"], required=True)
    args = parser.parse_args()
    if WORK == PACKAGE or PACKAGE in WORK.parents:
        raise SystemExit("Set MOLXAI_WORK_ROOT outside this repository.")
    HERE.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    if args.mode == "aggregate":
        aggregate()
        return
    assert torch.cuda.is_available()
    source_audit()
    write_json(HERE / f"environment_{args.mode}.json", {"command": sys.argv, "timestamp_utc": datetime.now(timezone.utc).isoformat(), "python": sys.version, "executable": sys.executable, "platform": platform.platform(), "torch": torch.__version__, "pyg": torch_geometric.__version__, "numpy": np.__version__, "pandas": pd.__version__, "device": torch.cuda.get_device_name(0), "runner_sha256": sha(__file__), "contract_sha256": sha(CONTRACT), "selector_sha256": sha(CODE / "frozen_index_policy.py"), "imported_sources": {name: sha(PACKAGE / "src" / name) for name in ["run_gradient_grid.py", "run_bxaic_probe.py", "audit_graph_attribution.py"]}})
    log(f"start mode={args.mode}")
    if args.mode == "pilot":
        log(json.dumps(pilot()))
        return
    check = json.loads((HERE / "pilot/result.json").read_text())
    assert check["status"] == "PASS" and check["projection_below_30_days"]
    policy_check()
    for family, tasks in [("bxaic", BXAIC_TASKS), ("google", GOOGLE_TASKS)]:
        for task in tasks:
            log(f"load_task {family}/{task}")
            parts = load_parts(family, task)
            for kind in ["gin", "gcn"]:
                for seed in [42, 123, 2026]:
                    run_cell(family, task, kind, seed, parts)
            del parts
            gc.collect()
    aggregate()
    log("full_complete")


if __name__ == "__main__":
    main()
