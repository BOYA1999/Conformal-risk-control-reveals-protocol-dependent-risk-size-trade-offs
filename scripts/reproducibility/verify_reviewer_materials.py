import argparse
import csv
import hashlib
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--root', type=Path, required=True)
args = parser.parse_args()
root = args.root.resolve()
with (root / 'payload_manifest.csv').open(encoding='utf-8', newline='') as stream:
    rows = list(csv.DictReader(stream))
for row in rows:
    relative = Path(row['path'])
    assert not relative.is_absolute() and '..' not in relative.parts
    path = (root / relative).resolve()
    assert root in path.parents and path.is_file()
    with path.open('rb') as stream:
        observed = hashlib.file_digest(stream, 'sha256').hexdigest()
    assert path.stat().st_size == int(row['bytes']) and observed == row['sha256'], row['path']
expected = {row['path'] for row in rows}
actual = {path.relative_to(root).as_posix() for path in root.rglob('*') if path.is_file() and path.name != 'payload_manifest.csv' and '__pycache__' not in path.parts}
assert actual == expected, 'Manifest and file inventory differ.'
print(json.dumps({'status': 'PASS_FILE_INTEGRITY', 'files': len(rows), 'bytes': sum(int(row['bytes']) for row in rows), 'boundary': 'Integrity only; not model or statistic reconstruction and not permission to redistribute.'}))
