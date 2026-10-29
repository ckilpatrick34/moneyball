"""Odds conversion, de-vigging, and expected value.

All probabilities are in [0, 1]. American odds are ints/floats like -150 or +130.
"""

from __future__ import annotations

import math
from typing import Sequence


def american_to_decimal(american: float) -> float:
    a = float(american)
    if a == 0:
        raise ValueError("American odds cannot be 0")
    return 1.0 + (a / 100.0 if a > 0 else 100.0 / -a)


def decimal_to_american(decimal: float) -> float:
    d = float(decimal)
    if d <= 1.0:
        raise ValueError("Decimal odds must be > 1")
    return (d - 1.0) * 100.0 if d >= 2.0 else -100.0 / (d - 1.0)


def implied_prob(american: float) -> float:
    """Raw implied probability of a price, vig included."""
    return 1.0 / american_to_decimal(american)


def prob_to_american(p: float) -> float:
    return decimal_to_american(1.0 / p)


def devig_multiplicative(raw: Sequence[float]) -> list[float]:
    s = sum(raw)
    return [p / s for p in raw]


def devig_power(raw: Sequence[float], tol: float = 1e-10) -> list[float]:
    """Power method: find k with sum(p_i ** k) == 1.

    Handles favourite-longshot bias better than proportional normalisation,
    which matters for big MLB favourites and lopsided college games.
    """
    if abs(sum(raw) - 1.0) < tol:
        return list(raw)
    lo, hi = 0.5, 3.0
    for _ in range(200):
        k = (lo + hi) / 2
        s = sum(p ** k for p in raw)
        if s > 1.0:
            lo = k
        else:
            hi = k
        if hi - lo < tol:
            break
    k = (lo + hi) / 2
    out = [p ** k for p in raw]
    s = sum(out)
    return [p / s for p in out]


def fair_probs(prices: Sequence[float], method: str = "power") -> list[float]:
    """No-vig probabilities for a complete set of American prices on one market."""
    raw = [implied_prob(p) for p in prices]
    if method == "power":
        return devig_power(raw)
    if method == "multiplicative":
        return devig_multiplicative(raw)
    raise ValueError(f"unknown devig method {method}")


def expected_value(p: float, american: float) -> float:
    """EV per 1 unit staked at this price if the true win probability is p (no push)."""
    d = american_to_decimal(american)
    return p * (d - 1.0) - (1.0 - p)


def expected_value_with_push(p_win: float, p_push: float, american: float) -> float:
    d = american_to_decimal(american)
    p_lose = 1.0 - p_win - p_push
    return p_win * (d - 1.0) - p_lose


def clv(taken_american: float, close_fair_prob: float) -> float:
    """Closing line value: how much better the taken price is than the fair closing price.

    0.03 means the price you got pays 3% more than the no-vig closing line implies.
    Sustained positive CLV is the best early evidence of a real edge.
    """
    return american_to_decimal(taken_american) * close_fair_prob - 1.0


def logit(p: float) -> float:
    p = min(max(p, 1e-9), 1 - 1e-9)
    return math.log(p / (1 - p))


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))
