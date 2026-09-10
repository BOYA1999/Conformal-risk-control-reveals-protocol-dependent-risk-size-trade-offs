import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import networkx
import numpy
import pandas
from rdkit import rdBase


ROOT = Path(__file__).resolve().parent
REPO = Path(r"<moleculeace-repository>")
DATA = REPO / "MoleculeACE" / "Data" / "benchmark_data"
SPLIT_DIR = Path(r"<split-semantics-directory>")
TARGETS = [
    "CHEMBL204_Ki",
    "CHEMBL214_Ki",
    "CHEMBL228_Ki",
    "CHEMBL233_Ki",
    "CHEMBL234_Ki",
    "CHEMBL235_EC50",
    "CHEMBL236_Ki",
    "CHEMBL237_Ki",
    "CHEMBL244_Ki",
    "CHEMBL264_Ki",
    "CHEMBL4792_Ki",
]
EXPECTED_COMMIT = "7e6de0bd2968c56589c580f2a397f01c531ede26"
EXPECTED_MANIFEST_HASH = "C71E0E2624AFFE872E368131517FA1A565B81520FED25F87CA4C64B53F83174F"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def git(*args):
    return subprocess.check_output(["git", "-C", str(REPO), *args], text=True).strip()


def main():
    selection_path = SPLIT_DIR / "dataset_selection.json"
    manifest_path = SPLIT_DIR / "split_manifest.csv"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    selected_hashes = {row["dataset"]: row["source_sha256"].upper() for row in selection["selected"]}
    commit = git("rev-parse", "HEAD")
    manifest_hash = sha256(manifest_path)
    if commit != EXPECTED_COMMIT:
        raise RuntimeError(f"MoleculeACE commit drift: {commit}")
    if manifest_hash != EXPECTED_MANIFEST_HASH:
        raise RuntimeError(f"split manifest drift: {manifest_hash}")
    sources = []
    for target in TARGETS:
        path = DATA / f"{target}.csv"
        observed = sha256(path)
        expected = selected_hashes[target]
        if observed != expected:
            raise RuntimeError(f"benchmark source drift for {target}: {observed}")
        frame = pandas.read_csv(path)
        sources.append(
            {
                "dataset": target,
                "path": str(path),
                "sha256": observed,
                "selection_json_sha256": expected,
                "rows": len(frame),
                "columns": list(frame.columns),
            }
        )
    method_files = [
        Path(r"<split-method-source>\prepare_data.py"),
        Path(r"<split-method-source>\prepare_splits.py"),
        REPO / "MoleculeACE" / "benchmark" / "cliffs.py",
    ]
    provenance = {
        "status": "PASS",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "code_source": {
            "repository": str(REPO),
            "remote": git("remote", "get-url", "origin"),
            "commit": commit,
            "commit_time": git("log", "-1", "--format=%cI"),
            "commit_subject": git("log", "-1", "--format=%s"),
            "license": "MIT",
            "license_scope": "MoleculeACE code only",
            "license_file": str(REPO / "LICENCE"),
            "license_sha256": sha256(REPO / "LICENCE"),
            "readme": str(REPO / "README.md"),
            "readme_sha256": sha256(REPO / "README.md"),
            "paper_doi": "10.1021/acs.jcim.2c01073",
        },
        "data_source": {
            "lineage": "MoleculeACE benchmark data derived from ChEMBL v29",
            "license": "CC BY-SA 3.0",
            "license_scope": "benchmark and derived data; not covered by the MoleculeACE MIT code license",
            "license_url": "https://chembl.gitbook.io/chembl-interface-documentation/frequently-asked-questions/general-questions",
            "publication_doi": "10.1021/acs.jcim.2c01073",
            "evidence_basis": "ChEMBL official licensing FAQ, MoleculeACE publication, and frozen source-file hashes",
            "benchmark_files": sources,
        },
        "split_source": {
            "manifest": str(manifest_path),
            "manifest_sha256": manifest_hash,
            "selection": str(selection_path),
            "selection_sha256": sha256(selection_path),
            "selection_commit": selection["moleculeace_commit"],
        },
        "method_sources": [{"path": str(path), "sha256": sha256(path)} for path in method_files],
        "claim_boundary": "Experimental SAR-linked transformation-site proxy; not causal, complete, mechanistic, expert, or ground truth.",
    }
    environment = {
        "python": sys.version,
        "platform": platform.platform(),
        "pandas": pandas.__version__,
        "numpy": numpy.__version__,
        "networkx": networkx.__version__,
        "rdkit": rdBase.rdkitVersion,
    }
    (ROOT / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    (ROOT / "pretrain_environment.json").write_text(json.dumps(environment, indent=2), encoding="utf-8")
    print(json.dumps({"status": "PASS", "targets": len(sources), "commit": commit, "manifest_sha256": manifest_hash}, indent=2))


if __name__ == "__main__":
    main()
