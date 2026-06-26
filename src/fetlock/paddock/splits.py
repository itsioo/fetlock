from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from fetlock.config import SplitSpec
from fetlock.paddock.cohort import Subject


@dataclass(frozen=True)
class StratifiedSplit:
    train: List[Subject]
    val: List[Subject]
    test: List[Subject]


def _stratum_key(subject: Subject, strata: Tuple[str, ...]) -> Tuple[int, ...]:
    fields = {
        "sex": subject.sex,
        "age_band": subject.age_band,
        "severity": int(subject.severity > 0),
        "label": subject.label,
    }
    return tuple(fields[name] for name in strata)


def stratified_subject_split(
    subjects: List[Subject],
    spec: SplitSpec,
    seed: int,
) -> StratifiedSplit:
    rng = np.random.default_rng(seed)
    buckets: Dict[Tuple[int, ...], List[Subject]] = defaultdict(list)
    for subject in subjects:
        buckets[_stratum_key(subject, spec.strata)].append(subject)

    train: List[Subject] = []
    val: List[Subject] = []
    test: List[Subject] = []
    for key in sorted(buckets):
        members = buckets[key]
        order = rng.permutation(len(members))
        n_train = round(len(members) * spec.train)
        n_val = round(len(members) * spec.val)
        for position, index in enumerate(order):
            subject = members[int(index)]
            if position < n_train:
                train.append(subject)
            elif position < n_train + n_val:
                val.append(subject)
            else:
                test.append(subject)
    return StratifiedSplit(train=train, val=val, test=test)
