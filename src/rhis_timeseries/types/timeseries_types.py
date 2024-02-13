from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


type TimeSeriesFlex = list[int | float] | np.ndarray[int | float]
