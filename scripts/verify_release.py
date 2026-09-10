import ast
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
]
for relative in required:
    require((ROOT / relative).is_file(), f"missing required file: {relative}")

readme = (ROOT / "README.md").read_text(encoding="utf-8")
license_text = (ROOT / "LICENSE").read_text(encoding="utf-8")
require("current release" in readme and "repository-authored code and documentation" in readme, "README scope")
require("MIT License" in license_text, "root license")

for path in ROOT.rglob("*.py"):
    if "__pycache__" in path.parts:
        continue
    try:
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except Exception as error:
        failures.append(f"Python syntax: {path.relative_to(ROOT)}: {error}")

for path in ROOT.rglob("*"):
    if not path.is_file() or path == MANIFEST or "__pycache__" in path.parts:
        continue
    if path.suffix.lower() in {".py", ".json", ".csv", ".md", ".txt", ".yml", ".yaml", ".toml", ".ini"}:
        text = path.read_text(encoding="utf-8", errors="ignore")
        drive_token = "[" + "A-Za-z" + "]:" + r"[\\/]"
        unix_home = "/" + "home/"
        unix_users = "/" + "Users/"
        local_path_pattern = rf"(?m)(?:^|[\s\"'`])(?:{drive_token}|{unix_home}|{unix_users})"
        require(not re.search(local_path_pattern, text), f"absolute path: {path.relative_to(ROOT)}")
        require(not re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Za-z]{2,}", text), f"email address: {path.relative_to(ROOT)}")

for path in ROOT.rglob("*"):
    if path.is_file() and path.suffix.lower() in {".pt", ".pth", ".ckpt", ".bin", ".pkl", ".pickle", ".sdf", ".smi", ".parquet"}:
        failures.append(f"blocked file type: {path.relative_to(ROOT)}")
    if path.is_dir() and path.name in {".git", "__pycache__", ".pytest_cache"}:
        failures.append(f"blocked directory: {path.relative_to(ROOT)}")

if MANIFEST.is_file():
    listed = {}
    for line in MANIFEST.read_text(encoding="utf-8").splitlines():
        digest, relative = line.split("  ", 1)
        listed[relative] = digest
    actual = sorted(path.relative_to(ROOT).as_posix() for path in ROOT.rglob("*") if path.is_file() and path != MANIFEST and "__pycache__" not in path.parts)
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

if failures:
    raise SystemExit("FAIL\n" + "\n".join(failures))
print("PASS: current release package checks")
