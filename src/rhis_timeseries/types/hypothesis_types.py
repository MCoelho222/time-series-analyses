"""Typing classes for hypothesis tests."""
from __future__ import annotations

from typing import NamedTuple


class TestResults(NamedTuple):
    """Types for hypothesis test results."""
    statistic: float
    p_value: float
    alternative: str
