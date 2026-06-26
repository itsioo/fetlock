from typing import Dict

import numpy as np
from numpy.typing import NDArray
from scipy.stats import rankdata


def roc_auc(scores: NDArray[np.float64], labels: NDArray[np.int64]) -> float:
    positives = int(labels.sum())
    negatives = int(len(labels) - positives)
    if positives == 0 or negatives == 0:
        raise ValueError("AUC needs both classes present")
    ranks = rankdata(scores)
    rank_sum = float(ranks[labels == 1].sum())
    return (rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def _counts(
    scores: NDArray[np.float64], labels: NDArray[np.int64], threshold: float
) -> NDArray[np.int64]:
    predicted = (scores >= threshold).astype(np.int64)
    tp = int(((predicted == 1) & (labels == 1)).sum())
    fp = int(((predicted == 1) & (labels == 0)).sum())
    tn = int(((predicted == 0) & (labels == 0)).sum())
    fn = int(((predicted == 0) & (labels == 1)).sum())
    return np.array([tp, fp, tn, fn], dtype=np.int64)


def sensitivity(
    scores: NDArray[np.float64], labels: NDArray[np.int64], threshold: float = 0.5
) -> float:
    tp, _, _, fn = _counts(scores, labels, threshold)
    return tp / (tp + fn) if (tp + fn) else 0.0


def specificity(
    scores: NDArray[np.float64], labels: NDArray[np.int64], threshold: float = 0.5
) -> float:
    _, fp, tn, _ = _counts(scores, labels, threshold)
    return tn / (tn + fp) if (tn + fp) else 0.0


def f1_score(
    scores: NDArray[np.float64], labels: NDArray[np.int64], threshold: float = 0.5
) -> float:
    tp, fp, _, fn = _counts(scores, labels, threshold)
    denominator = 2 * tp + fp + fn
    return (2 * tp) / denominator if denominator else 0.0


def accuracy(
    scores: NDArray[np.float64], labels: NDArray[np.int64], threshold: float = 0.5
) -> float:
    tp, fp, tn, fn = _counts(scores, labels, threshold)
    return (tp + tn) / (tp + fp + tn + fn)


def score_panel(
    scores: NDArray[np.float64], labels: NDArray[np.int64], threshold: float = 0.5
) -> Dict[str, float]:
    return {
        "auc": roc_auc(scores, labels),
        "f1": f1_score(scores, labels, threshold),
        "sensitivity": sensitivity(scores, labels, threshold),
        "specificity": specificity(scores, labels, threshold),
        "accuracy": accuracy(scores, labels, threshold),
    }
