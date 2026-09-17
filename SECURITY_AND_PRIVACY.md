# Privacy and release scope

This package contains repository-authored code, source metadata and aggregate tables selected by an explicit allowlist. Manuscripts, publication figures, Word/TeX/PDF files, manuscript author metadata, email addresses, account names, local absolute paths, raw molecular records, molecule-level predictions or attributions, caches, logs, checkpoints and third-party repository copies remain outside its scope.

`scripts/verify_release.py` checks the manifest, denylisted file types and directories, common credential and email patterns, local path patterns, CSV molecule-level fields, required license notices and current aggregate invariants. Automated pattern checks do not establish legal permission or replace content inspection.

Analysis reruns use external working directories for excluded inputs and outputs. The Liver loader verifies both source hashes before restricted NumPy-array reconstruction; do not replace it with unrestricted pickle loading. Rebuild and rescan the package after any change. Citation records identify public sources; manuscript author metadata is outside this code-and-aggregate release.
