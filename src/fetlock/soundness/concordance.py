from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from numpy.typing import NDArray

from fetlock.soundness.stats import benjamini_hochberg


def pearson_r(x: NDArray[np.float64], y: NDArray[np.float64]) -> float:
    xc = x - x.mean()
    yc = y - y.mean()
    denominator = float(np.sqrt((xc * xc).sum() * (yc * yc).sum()))
    if denominator == 0.0:
        return 0.0
    return float((xc * yc).sum() / denominator)


@dataclass(frozen=True)
class ConcordanceCell:
    feature: str
    roi: str
    r: float
    p_value: float
    ci_low: float
    ci_high: float
    fdr_reject: bool


def _permutation_p(
    x: NDArray[np.float64], y: NDArray[np.float64], n_perm: int, rng: np.random.Generator
) -> float:
    observed = abs(pearson_r(x, y))
    count = 0
    for _ in range(n_perm):
        if abs(pearson_r(x, rng.permutation(y))) >= observed:
            count += 1
    return (count + 1) / (n_perm + 1)


def _bootstrap_ci(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    n_boot: int,
    rng: np.random.Generator,
    alpha: float,
) -> Tuple[float, float]:
    n = len(x)
    values = np.empty(n_boot, dtype=np.float64)
    for draw in range(n_boot):
        idx = rng.integers(0, n, n)
        values[draw] = pearson_r(x[idx], y[idx])
    low, high = np.percentile(values, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(low), float(high)


def fmri_concordance(
    wearable: Dict[str, NDArray[np.float64]],
    roi: Dict[str, NDArray[np.float64]],
    pairs: List[Tuple[str, str]],
    n_perm: int = 5000,
    n_boot: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
) -> List[ConcordanceCell]:
    rng = np.random.default_rng(seed)
    rows: List[Tuple[str, str, float, float, float, float]] = []
    pvalues = np.empty(len(pairs), dtype=np.float64)
    for position, (feature, region) in enumerate(pairs):
        x = wearable[feature]
        y = roi[region]
        r = pearson_r(x, y)
        p = _permutation_p(x, y, n_perm, rng)
        low, high = _bootstrap_ci(x, y, n_boot, rng, alpha)
        pvalues[position] = p
        rows.append((feature, region, r, p, low, high))
    _, reject = benjamini_hochberg(pvalues, alpha)
    return [
        ConcordanceCell(feature, region, r, p, low, high, bool(flag))
        for (feature, region, r, p, low, high), flag in zip(rows, reject)
    ]
