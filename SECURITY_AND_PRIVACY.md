# Privacy and release scope

This package was assembled from an explicit allowlist. It excludes manuscript and supplementary prose, publication figures, Word/TeX/PDF files, author identities, affiliations, email addresses, account names, local absolute paths, comments, tracked changes, raw molecular records, molecule-level predictions or attributions, caches, logs, checkpoints, model weights, failed-run artifacts, and bundled third-party repositories.

`scripts/verify_release.py` checks the manifest, denylisted file types and directories, common secret and identity patterns, local path patterns, structure-bearing schemas, and license boundaries. The final ZIP is extracted and checked again.

All scripts use external working directories for excluded inputs and outputs. Rebuild and rescan the package after any change. Author metadata, repository ownership, citation records, and manuscript material are outside this local code-and-aggregate release.
