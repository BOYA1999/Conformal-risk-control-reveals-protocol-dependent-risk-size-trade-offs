import csv
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def main():
    checkpoints = sorted((ROOT / "models").glob("*/seed_*.pt"))
    if len(checkpoints) != 33:
        raise RuntimeError(f"expected 33 checkpoints, found {len(checkpoints)}")
    with (ROOT / "model_checkpoint_hashes.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["relative_path", "sha256", "bytes"], lineterminator="\n")
        writer.writeheader()
        for path in checkpoints:
            writer.writerow({"relative_path": path.relative_to(ROOT).as_posix(), "sha256": sha256(path), "bytes": path.stat().st_size})
    print("CHECKPOINT_HASH_PASS files=33")


if __name__ == "__main__":
    main()
