from __future__ import annotations

import numpy as np

from rhis.stats.hypothesis.homogeneity import mann_whitney
from rhis.stats.hypothesis.independence import wald_wolfowitz
from rhis.stats.hypothesis.randomness import wallismoore
from rhis.stats.hypothesis.stationarity import mann_kendall


def calculate_rhis(ts: np.ndarray, alpha: float, *, min: bool=True) -> int | dict[float]:
    hypos = ['R', 'H', 'I', 'S']
    rhis_tests = [wallismoore, mann_whitney, wald_wolfowitz, mann_kendall]
    test_dict = dict(zip(hypos, rhis_tests))

    ps = []
    for hyp in hypos:
        ps.append(test_dict[hyp](ts, alpha).p_value)

    result = round(np.min(ps), 4) if min else ps

    return result


