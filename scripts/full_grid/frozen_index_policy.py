import numpy as np


def top_fraction_set(scores, fraction):
    scores = np.asarray(scores, dtype=float)
    if not 0 <= fraction <= 1:
        raise ValueError("fraction must be in [0, 1]")
    k = 0 if fraction == 0 else int(np.ceil(fraction * len(scores)))
    order = np.lexsort((np.arange(len(scores)), -scores))
    return set(order[:k].tolist())


def missed_rationale_loss(selected, rationale):
    rationale = set(rationale)
    if not rationale:
        raise ValueError("primary loss requires a non-empty rationale")
    return 1.0 - len(set(selected) & rationale) / len(rationale)


def loss_table(score_rows, rationale_rows, fractions):
    fractions = np.asarray(fractions, dtype=float)
    table = np.empty((len(score_rows), len(fractions)), dtype=float)
    for i, (scores, rationale) in enumerate(zip(score_rows, rationale_rows)):
        scores = np.asarray(scores, dtype=float)
        rationale = np.asarray(sorted(set(rationale)), dtype=int)
        if not len(rationale):
            raise ValueError("primary loss requires a non-empty rationale")
        order = np.lexsort((np.arange(len(scores)), -scores))
        ranks = np.empty(len(scores), dtype=int)
        ranks[order] = np.arange(len(scores))
        rationale_ranks = np.sort(ranks[rationale])
        retained = np.where(fractions == 0, 0, np.ceil(fractions * len(scores))).astype(int)
        recovered = np.searchsorted(rationale_ranks, retained, side="left")
        table[i] = 1.0 - recovered / len(rationale)
    if np.any(np.diff(table, axis=1) > 1e-12):
        raise ValueError("loss must be non-increasing as the atom set grows")
    return table


def calibrate_crc(calibration_losses, fractions, alpha, bound=1.0):
    losses = np.asarray(calibration_losses, dtype=float)
    fractions = np.asarray(fractions, dtype=float)
    if losses.ndim != 2 or losses.shape[1] != len(fractions):
        raise ValueError("loss table and fraction grid disagree")
    if np.any(np.diff(fractions) <= 0) or np.any(np.diff(losses, axis=1) > 1e-12):
        raise ValueError("fractions must grow and losses must not grow")
    n = losses.shape[0]
    corrected_risk = (n * losses.mean(axis=0) + bound) / (n + 1)
    feasible = np.flatnonzero(corrected_risk <= alpha)
    if not len(feasible):
        raise ValueError("no feasible set; calibration sample is too small for alpha")
    index = int(feasible[0])
    return {
        "fraction": float(fractions[index]),
        "index": index,
        "empirical_risk": float(losses[:, index].mean()),
        "corrected_risk": float(corrected_risk[index]),
        "n_calibration": n,
        "alpha": float(alpha),
    }
