"""Typing classes for hypothesis tests."""
from __future__ import annotations

from typing import NamedTuple


class TestResults(NamedTuple):
    """Types for hypothesis test results."""
    test_stat: float
    p_value: float
