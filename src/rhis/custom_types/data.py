from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

TimeSeriesFlex = list[int | float] | NDArray[np.int64 | np.float64]
