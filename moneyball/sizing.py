"""Bet sizing: fractional Kelly with hard caps."""

from __future__ import annotations

from dataclasses import dataclass

from .odds import american_to_decimal


@dataclass
class SizingConfig:
    kelly_fraction: float = 0.25   # quarter Kelly: model error makes full Kelly ruinous
    max_bet_pct: float = 0.02      # never more than 2% of bankroll on one bet
    min_bet: float = 1.0           # skip dust bets


def kelly(p: float, american: float, p_push: float = 0.0) -> float:
    """Full-Kelly fraction of bankroll for a single bet. 0 when there is no edge."""
    b = american_to_decimal(american) - 1.0
    p_lose = 1.0 - p - p_push
    if b <= 0:
        return 0.0
    f = (b * p - p_lose) / b
    return max(0.0, f)


def stake(p: float, american: float, bankroll: float, cfg: SizingConfig | None = None,
          p_push: float = 0.0) -> float:
    cfg = cfg or SizingConfig()
    f = kelly(p, american, p_push) * cfg.kelly_fraction
    amt = bankroll * min(f, cfg.max_bet_pct)
    return round(amt, 2) if amt >= cfg.min_bet else 0.0
