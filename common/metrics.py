"""Dependency-free evaluation metrics.

We compute ROC AUC by hand (Mann–Whitney U / rank statistic) rather than pulling
in scikit-learn, keeping the image small and the dependency set tight.
"""

import numpy as np


def _rankdata_average(a: np.ndarray) -> np.ndarray:
    """Ranks with ties resolved by averaging (equivalent to scipy's 'average')."""
    arr = np.asarray(a, dtype=float)
    sorter = np.argsort(arr, kind="mergesort")
    inv = np.empty(sorter.size, dtype=np.intp)
    inv[sorter] = np.arange(sorter.size, dtype=np.intp)
    arr = arr[sorter]
    obs = np.r_[True, arr[1:] != arr[:-1]]
    dense = obs.cumsum()[inv]
    count = np.r_[np.nonzero(obs)[0], len(obs)]
    return 0.5 * (count[dense] + count[dense - 1] + 1)


def roc_auc(y_true, scores) -> float:
    """ROC AUC for binary labels. Returns NaN if only one class is present."""
    y = np.asarray(y_true).astype(int)
    s = np.asarray(scores, dtype=float)
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = _rankdata_average(s)
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))
