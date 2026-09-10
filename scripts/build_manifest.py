import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "MANIFEST_SHA256.txt"


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


paths = sorted(
    path for path in ROOT.rglob("*")
    if path.is_file() and path != MANIFEST and "__pycache__" not in path.parts
)
MANIFEST.write_text(
    "".join(f"{digest(path)}  {path.relative_to(ROOT).as_posix()}\n" for path in paths),
    encoding="utf-8",
    newline="\n",
)
print(f"WROTE {MANIFEST.name}: {len(paths)} files")
