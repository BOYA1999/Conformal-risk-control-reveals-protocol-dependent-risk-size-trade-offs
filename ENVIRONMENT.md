# Executed environment

The current release was checked with:

- Python 3.12.0
- PyTorch 2.11.0 with CUDA 12.8 runtime
- PyTorch Geometric 2.5.0
- NumPy 1.26.4
- pandas 2.3.3
- SciPy 1.17.1
- scikit-learn 1.8.0
- RDKit 2026.03.3
- Captum 0.7.0
- Chemprop 2.2.4
- Matplotlib 3.10.8
- Polaris 0.13.0
- NetworkX 3.6.1
- python-Levenshtein 0.27.4

A7 formal attribution used the frozen deterministic CPU path with one thread and 32 integrated-gradient steps. Model training remained the frozen 33-cell, 11-target by three-seed run.

Hardware details, host names, account names, and local paths are omitted because they are not needed to reproduce the documented checks.
