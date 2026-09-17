import hashlib
import os
import urllib.request
from pathlib import Path


PACKAGE = Path(__file__).resolve().parents[2]
WORK = Path(os.environ.get('MOLXAI_WORK_ROOT', '../molxai-work')).resolve()
FILES = {
    'Liver.csv': ('1rwzRokbkuE0brZ4LzucYBcSXkoJQ4wvw', '255054c8030712c094de5f88484311d006a667ae6565a2879c417d5cd47d533f'),
    'attributions.npz': ('1FxyMnDr2_oy494Ljh1lTokWkSEt3yrgp', 'd2abe69c7e5d1f7fb1f2bb77029ed6d01492ef63c744c14ca11141660672383c'),
}


def main():
    if WORK == PACKAGE or PACKAGE in WORK.parents:
        raise SystemExit('Set MOLXAI_WORK_ROOT to a directory outside this repository.')
    destination = WORK / 'independent_reference/data/raw'
    destination.mkdir(parents=True, exist_ok=True)
    for name, (file_id, expected) in FILES.items():
        path = destination / name
        data = path.read_bytes() if path.exists() else urllib.request.urlopen(
            'https://drive.google.com/uc?export=download&id=' + file_id, timeout=60
        ).read()
        if hashlib.sha256(data).hexdigest() != expected:
            raise SystemExit(f'Source hash mismatch: {name}; inspect the official MolRep data release.')
        if not path.exists():
            path.write_bytes(data)
        print(f'PASS {name}: verified source SHA-256')


if __name__ == '__main__':
    main()
