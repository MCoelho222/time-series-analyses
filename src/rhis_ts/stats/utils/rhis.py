from __future__ import annotations

import numpy as np

from rhis_ts.stats.hyp_testing.mann_kendall import mann_kendall
from rhis_ts.stats.hyp_testing.mann_whitney import mann_whitney
from rhis_ts.stats.hyp_testing.runs import wallismoore
from rhis_ts.stats.hyp_testing.wald_wolfowitz import wald_wolfowitz


def min_from_rhis(ts, alpha):
    hyps = ['R', 'H', 'I', 'S']
    rhis_tests = [wallismoore, mann_whitney, wald_wolfowitz, mann_kendall]
    test_dict = dict(zip(hyps, rhis_tests))

    ps = []
    for hyp in hyps:
        ps.append(test_dict[hyp](ts, alpha).p_value)

    return round(np.min(ps), 4)


