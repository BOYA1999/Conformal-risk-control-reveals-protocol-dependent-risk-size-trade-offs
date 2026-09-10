import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def main():
    manifest = ROOT / "MANIFEST.sha256"
    recorded = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        digest, relative = line.split(" *", 1)
        recorded[relative] = digest
    files = {
        path.relative_to(ROOT).as_posix(): path
        for path in ROOT.rglob("*")
        if path.is_file() and path.name != "MANIFEST.sha256" and "__pycache__" not in path.parts and path.suffix != ".pyc"
    }
    missing = sorted(set(files) - set(recorded))
    extra = sorted(set(recorded) - set(files))
    mismatched = sorted(relative for relative in set(files) & set(recorded) if sha256(files[relative]) != recorded[relative])
    status = "PASS" if not missing and not extra and not mismatched else "FAIL"
    print(f"MANIFEST_VERIFY_{status} files={len(files)} missing={len(missing)} extra={len(extra)} mismatched={len(mismatched)}")
    if status != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
