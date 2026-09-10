"""Typing classes for hypothesis tests."""
from __future__ import annotations

from typing import Literal, NamedTuple, TypeAlias


class TestResults(NamedTuple):
    """Types for hypothesis test results."""
    statistic: float
    p_value: float
    alternative: str

RhisCode: TypeAlias = Literal["r", "h", "i", "s"]
RhisStat: TypeAlias = Literal["min", "median", "mean", "max"]


class TestDecisionNormal(NamedTuple):
    p_value: float
    alpha: float
    reject: bool
    alternative: str
