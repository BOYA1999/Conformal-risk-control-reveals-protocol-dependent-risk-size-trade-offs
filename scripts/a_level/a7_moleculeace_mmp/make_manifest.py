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
    files = sorted(
        path
        for path in ROOT.rglob("*")
        if path.is_file()
        and path.name != "MANIFEST.sha256"
        and "__pycache__" not in path.parts
        and path.suffix != ".pyc"
    )
    lines = [f"{sha256(path)} *{path.relative_to(ROOT).as_posix()}" for path in files]
    (ROOT / "MANIFEST.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print(f"MANIFEST_PASS files={len(files)}")


if __name__ == "__main__":
    main()
