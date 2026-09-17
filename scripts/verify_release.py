import ast
import csv
import gzip
import io
import hashlib
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "MANIFEST_SHA256.txt"
failures = []


def require(condition, message):
    if not condition:
        failures.append(message)


required = [
    ".gitignore",
    "LICENSE",
    "README.md",
    "REPRODUCIBILITY.md",
    "DATA_SOURCES.md",
    "ENVIRONMENT.md",
    "SECURITY_AND_PRIVACY.md",
    "THIRD_PARTY_NOTICES.md",
    "requirements.txt",
    ".gitattributes",
    ".github/workflows/package-integrity.yml",
    "CHANGELOG.md",
    "scripts/build_manifest.py",
    "scripts/v8/verify_version_matched_aggregates.py",
    "scripts/v8/pharmacophore_oracle/audit_pharmacophore_oracle.py",
    "scripts/v8/rings_stratified/run_analysis.py",
    "scripts/v8/rings_stratified/validate_analysis.py",
    "results/v8/pharmacophore_oracle/reference_return_tie_summary.csv",
    "results/v8/rings_stratified/manuscript_ready_stratified_table.csv",
    "results/v8/version_provenance/analysis_version_result_provenance.csv",
    "contracts/v8/pharmacophore_oracle/analysis_contract.json",
    "contracts/v8/rings_stratified/run_contract.json",
    "scripts/verify_current_analyses.py",
    "scripts/full_grid/run_full_grid.py",
    "scripts/full_grid/seal_run.py",
    "scripts/independent_reference/run_liver.py",
    "scripts/independent_reference/fetch_inputs.py",
    "contracts/independent_reference/run_contract.json",
    "results/full_grid/seal.json",
    "results/full_grid/method_summary.csv",
    "results/independent_reference/README.md",
    "results/independent_reference/summary.csv",
    "metadata/reproducibility/checkpoints.csv",
    "metadata/reproducibility/analysis_dependencies.csv",
    "metadata/reproducibility/adapter_lineage.json",
    "metadata/reproducibility/recorded_reconstruction.json",
    "scripts/reproducibility/rebuild_recorded_outputs.py",
    "scripts/reproducibility/verify_reviewer_materials.py",
    "scripts/reproducibility/rebuild_fig4_fig5_sources.py",
    "metadata/reproducibility/figure45_source_reconstruction.json",
    "results/task_symmetry/figure_sources/fig4_task_source.csv",
    "results/protocol_figure/figure_sources/fig5_target_source.csv",
    "scripts/revision/grid_sensitivity/validate_and_summarize.py",
    "scripts/revision/grid_sensitivity/rebuild_figure_sources.py",
    "results/grid_sensitivity/figure_sources/fig3_curve_source.csv",
    "results/grid_sensitivity/figure_sources/fig3_point_source.csv",
    "scripts/revision/task_symmetry/analyze.py",
    "scripts/revision/liver_calibration/analyze.py",
    "results/grid_sensitivity/independent_QA.json",
    "results/grid_sensitivity/method_summary.csv",
    "results/task_symmetry/summary.csv",
    "results/liver_calibration/primary_table.csv",
    "results/liver_calibration/summary.csv",
]
for relative in required:
    require((ROOT / relative).is_file(), f"missing required file: {relative}")

provenance_path = ROOT / "results/v8/version_provenance/analysis_version_result_provenance.csv"
if provenance_path.is_file():
    with provenance_path.open(encoding="utf-8", newline="") as handle:
        provenance = list(csv.DictReader(handle))
    expected_ids = {f"P{number:02d}" for number in range(1, 25)}
    require(len(provenance) == 24, "provenance row count")
    require({row["provenance_id"] for row in provenance} == expected_ids, "provenance P01-P24 coverage")
    for row in provenance:
        location = row["public_aggregate_location"]
        require(bool(location) and (ROOT / location).exists(), f"missing public aggregate location: {row['provenance_id']} {location}")

readme = (ROOT / "README.md").read_text(encoding="utf-8")
license_text = (ROOT / "LICENSE").read_text(encoding="utf-8")
require("current release" in readme and "repository-authored code and documentation" in readme, "README scope")
require("MIT License" in license_text, "root license")

for path in ROOT.rglob("*.py"):
    if {"__pycache__", ".git"}.intersection(path.parts):
        continue
    try:
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except Exception as error:
        failures.append(f"Python syntax: {path.relative_to(ROOT)}: {error}")

for path in ROOT.rglob("*"):
    if not path.is_file() or path == MANIFEST or {"__pycache__", ".git"}.intersection(path.parts):
        continue
    if path.suffix.lower() in {"", ".py", ".json", ".csv", ".md", ".txt", ".yml", ".yaml", ".toml", ".ini", ".gz"}:
        text = gzip.decompress(path.read_bytes()).decode('utf-8') if path.suffix.lower() == '.gz' else path.read_text(encoding="utf-8", errors="ignore")
        drive_token = "[" + "A-Za-z" + "]:" + r"[\\/]"
        unix_home = "/" + "home/"
        unix_users = "/" + "Users/"
        local_path_pattern = rf"(?m)(?:^|[\s\"'`])(?:{drive_token}|{unix_home}|{unix_users})"
        require(not re.search(local_path_pattern, text), f"absolute path: {path.relative_to(ROOT)}")
        require(not re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", text), f"email address: {path.relative_to(ROOT)}")
        secret = r'(?:gh' + r'[pousr]_[A-Za-z0-9]{30,}|AK' + r'IA[A-Z0-9]{16}|-----BEGIN ' + r'(?:RSA |EC |OPENSSH )?PRIVATE KEY-----)'
        require(not re.search(secret, text), f"credential pattern: {path.relative_to(ROOT)}")
    if path.suffix.lower() == '.csv' or path.name.endswith('.csv.gz'):
        headers = next(csv.reader(io.StringIO(text)), [])
        blocked = {'smiles', 'canonical_smiles', 'inchi', 'inchikey', 'source_index', 'source_id', 'molecule_id', 'mol_id', 'identity_sha256', 'node_atts', 'atom_scores', 'rationale_mask'}
        require(not blocked.intersection(h.strip().lower() for h in headers), f"molecule-level schema: {path.relative_to(ROOT)}")

for path in ROOT.rglob("*"):
    if '.git' in path.parts:
        continue
    if path.is_file() and path.suffix.lower() in {".pt", ".pth", ".ckpt", ".bin", ".pkl", ".pickle", ".sdf", ".smi", ".parquet", ".npz", ".npy", ".mol", ".doc", ".docx", ".tex", ".pdf", ".zip", ".7z", ".rar", ".log", ".png", ".svg", ".tif", ".tiff"}:
        failures.append(f"blocked file type: {path.relative_to(ROOT)}")
    if path.is_dir() and path.name in {"__pycache__", ".pytest_cache"}:
        failures.append(f"blocked directory: {path.relative_to(ROOT)}")

if MANIFEST.is_file():
    listed = {}
    for line in MANIFEST.read_text(encoding="utf-8").splitlines():
        digest, relative = line.split("  ", 1)
        listed[relative] = digest
    actual = sorted(path.relative_to(ROOT).as_posix() for path in ROOT.rglob("*") if path.is_file() and path != MANIFEST and ".git" not in path.parts and "__pycache__" not in path.parts)
    require(sorted(listed) == actual, "manifest file list")
    for relative, expected in listed.items():
        path = ROOT / relative
        if path.is_file():
            value = hashlib.sha256(path.read_bytes()).hexdigest()
            require(value == expected, f"manifest hash: {relative}")
else:
    failures.append("missing MANIFEST_SHA256.txt")

if (ROOT / "scripts/v8/verify_version_matched_aggregates.py").is_file():
    check = subprocess.run([sys.executable, "-B", "scripts/v8/verify_version_matched_aggregates.py"], cwd=ROOT, capture_output=True, text=True)
    require(check.returncode == 0, "aggregate verification: " + check.stdout + check.stderr)

if (ROOT / 'scripts/verify_current_analyses.py').is_file():
    check = subprocess.run([sys.executable, '-B', 'scripts/verify_current_analyses.py'], cwd=ROOT, capture_output=True, text=True)
    require(check.returncode == 0, 'current analysis verification: ' + check.stdout + check.stderr)

if failures:
    raise SystemExit("FAIL\n" + "\n".join(failures))
print("PASS: current release package checks")
