from __future__ import annotations

import numpy as np

from rhis_ts.stats.hypo.mann_kendall import mann_kendall
from rhis_ts.stats.hypo.mann_whitney import mann_whitney
from rhis_ts.stats.hypo.runs import wallismoore
from rhis_ts.stats.hypo.wald_wolfowitz import wald_wolfowitz


def calculate_rhis(ts: np.ndarray, alpha: float, *, min: bool=True) -> int | dict[float]:
    hypos = ['R', 'H', 'I', 'S']
    rhis_tests = [wallismoore, mann_whitney, wald_wolfowitz, mann_kendall]
    test_dict = dict(zip(hypos, rhis_tests))

    ps = []
    for hyp in hypos:
        ps.append(test_dict[hyp](ts, alpha).p_value)

    result = round(np.min(ps), 4) if min else ps

    return result


