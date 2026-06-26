from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.stats import norm

from fetlock.soundness.scores import roc_auc


@dataclass(frozen=True)
class DelongResult:
    auc_a: float
    auc_b: float
    diff: float
    z: float
    p_value: float


@dataclass(frozen=True)
class BootstrapResult:
    mean: float
    low: float
    high: float
    p_value: float


def _midrank(values: NDArray[np.float64]) -> NDArray[np.float64]:
    order = np.argsort(values, kind="mergesort")
    ordered = values[order]
    n = len(values)
    accumulator = np.empty(n, dtype=np.float64)
    index = 0
    while index < n:
        end = index
        while end < n and ordered[end] == ordered[index]:
            end += 1
        accumulator[index:end] = 0.5 * (index + end - 1) + 1
        index = end
    out = np.empty(n, dtype=np.float64)
    out[order] = accumulator
    return out


def delong_test(
    preds_a: NDArray[np.float64], preds_b: NDArray[np.float64], labels: NDArray[np.int64]
) -> DelongResult:
    positive = labels == 1
    negative = labels == 0
    m = int(positive.sum())
    n = int(negative.sum())
    if m == 0 or n == 0:
        raise ValueError("DeLong needs both classes present")
    stacked = np.vstack([preds_a.astype(np.float64), preds_b.astype(np.float64)])
    pos = stacked[:, positive]
    neg = stacked[:, negative]
    aucs = np.empty(2)
    v01 = np.empty((2, m))
    v10 = np.empty((2, n))
    for row in range(2):
        tx = _midrank(pos[row])
        ty = _midrank(neg[row])
        tz = _midrank(np.concatenate([pos[row], neg[row]]))
        aucs[row] = (tz[:m].sum() / m - (m + 1) / 2.0) / n
        v01[row] = (tz[:m] - tx) / n
        v10[row] = 1.0 - (tz[m:] - ty) / m
    cov = np.cov(v01) / m + np.cov(v10) / n
    variance = float(cov[0, 0] + cov[1, 1] - 2 * cov[0, 1])
    diff = float(aucs[0] - aucs[1])
    z = 0.0 if variance <= 0 else diff / float(np.sqrt(variance))
    return DelongResult(float(aucs[0]), float(aucs[1]), diff, z, float(2 * norm.sf(abs(z))))


def paired_bootstrap_auc_diff(
    preds_a: NDArray[np.float64],
    preds_b: NDArray[np.float64],
    labels: NDArray[np.int64],
    n_boot: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
) -> BootstrapResult:
    rng = np.random.default_rng(seed)
    n = len(labels)
    diffs: List[float] = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        drawn = labels[idx]
        if drawn.sum() == 0 or drawn.sum() == len(drawn):
            continue
        diffs.append(roc_auc(preds_a[idx], drawn) - roc_auc(preds_b[idx], drawn))
    sample = np.asarray(diffs, dtype=np.float64)
    low, high = np.percentile(sample, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    tail = 2.0 * min(float((sample <= 0).mean()), float((sample >= 0).mean()))
    return BootstrapResult(float(sample.mean()), float(low), float(high), min(1.0, tail))


def holm_bonferroni(
    pvalues: NDArray[np.float64], alpha: float = 0.05
) -> Tuple[NDArray[np.float64], NDArray[np.bool_]]:
    p = np.asarray(pvalues, dtype=np.float64)
    m = len(p)
    order = np.argsort(p)
    adjusted = np.empty(m, dtype=np.float64)
    running = 0.0
    for rank, idx in enumerate(order):
        running = min(1.0, max(running, p[idx] * (m - rank)))
        adjusted[idx] = running
    reject = np.zeros(m, dtype=np.bool_)
    keep = True
    for rank, idx in enumerate(order):
        keep = keep and (p[idx] * (m - rank) <= alpha)
        reject[idx] = keep
    return adjusted, reject


def benjamini_hochberg(
    pvalues: NDArray[np.float64], alpha: float = 0.05
) -> Tuple[NDArray[np.float64], NDArray[np.bool_]]:
    p = np.asarray(pvalues, dtype=np.float64)
    m = len(p)
    order = np.argsort(p)
    adjusted = np.empty(m, dtype=np.float64)
    prev = 1.0
    for rank in range(m - 1, -1, -1):
        idx = order[rank]
        prev = min(prev, p[idx] * m / (rank + 1))
        adjusted[idx] = prev
    return adjusted, adjusted <= alpha


def partial_eta_squared(
    values: NDArray[np.float64],
    factor_a: NDArray[np.int64],
    factor_b: NDArray[np.int64],
) -> float:
    grand = float(values.mean())
    ss_total = float(((values - grand) ** 2).sum())
    ss_a = 0.0
    for level in np.unique(factor_a):
        group = values[factor_a == level]
        ss_a += len(group) * (float(group.mean()) - grand) ** 2
    ss_b = 0.0
    for level in np.unique(factor_b):
        group = values[factor_b == level]
        ss_b += len(group) * (float(group.mean()) - grand) ** 2
    ss_cells = 0.0
    for la in np.unique(factor_a):
        for lb in np.unique(factor_b):
            cell = values[(factor_a == la) & (factor_b == lb)]
            if len(cell):
                ss_cells += len(cell) * (float(cell.mean()) - grand) ** 2
    ss_interaction = ss_cells - ss_a - ss_b
    ss_error = ss_total - ss_cells
    denominator = ss_interaction + ss_error
    return ss_interaction / denominator if denominator > 0 else 0.0


def cohens_kappa(rater_a: NDArray[np.int64], rater_b: NDArray[np.int64]) -> float:
    categories = np.unique(np.concatenate([rater_a, rater_b]))
    n = len(rater_a)
    confusion = np.zeros((len(categories), len(categories)), dtype=np.float64)
    lookup = {value: position for position, value in enumerate(categories)}
    for a, b in zip(rater_a, rater_b):
        confusion[lookup[int(a)], lookup[int(b)]] += 1
    observed = float(np.trace(confusion)) / n
    rows = confusion.sum(axis=1)
    cols = confusion.sum(axis=0)
    expected = float((rows * cols).sum()) / (n * n)
    return (observed - expected) / (1.0 - expected) if expected < 1.0 else 1.0


def permutation_test(
    sample_a: NDArray[np.float64],
    sample_b: NDArray[np.float64],
    n_perm: int = 10000,
    seed: int = 0,
) -> float:
    rng = np.random.default_rng(seed)
    observed = abs(float(sample_a.mean()) - float(sample_b.mean()))
    pool = np.concatenate([sample_a, sample_b])
    cut = len(sample_a)
    count = 0
    for _ in range(n_perm):
        shuffled = rng.permutation(pool)
        if abs(float(shuffled[:cut].mean()) - float(shuffled[cut:].mean())) >= observed:
            count += 1
    return (count + 1) / (n_perm + 1)


def expected_calibration_error(
    probs: NDArray[np.float64], labels: NDArray[np.int64], n_bins: int = 10
) -> float:
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = len(probs)
    error = 0.0
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (probs >= low) & (probs < high if high < 1.0 else probs <= high)
        if not mask.any():
            continue
        confidence = float(probs[mask].mean())
        observed = float(labels[mask].mean())
        error += (mask.sum() / total) * abs(confidence - observed)
    return error
