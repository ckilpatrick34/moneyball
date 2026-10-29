"""Turn expected runs into moneyline, run line, and total probabilities.

Runs are modelled as negative binomial (MLB run scoring is over-dispersed:
variance is roughly twice the mean, so Poisson understates blowouts and
shutouts). Probabilities are computed exactly on a grid instead of simulated.

Regulation ties go to extra innings. The old script counted every tie as an
away win, which shaved ~5 points off every home team's win probability.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

MAX_RUNS = 35

# Runs added to the total when a game goes to extras (ghost-runner era, rough).
EXTRA_RUNS_DIST = {1: 0.45, 2: 0.30, 3: 0.15, 4: 0.10}


def run_pmf(mu: float, dispersion: float | None = 4.0, max_runs: int = MAX_RUNS) -> np.ndarray:
    """P(runs = k) for k = 0..max_runs. dispersion=None gives Poisson."""
    mu = max(mu, 0.05)
    k = np.arange(max_runs + 1)
    if dispersion is None:
        logp = k * math.log(mu) - mu - np.array([math.lgamma(x + 1) for x in k])
    else:
        r = dispersion
        p = r / (r + mu)
        logp = (np.array([math.lgamma(x + r) - math.lgamma(r) - math.lgamma(x + 1) for x in k])
                + r * math.log(p) + k * math.log(1 - p))
    pmf = np.exp(logp)
    return pmf / pmf.sum()


@dataclass
class GameProbs:
    home_win: float
    away_win: float
    home_runs: float
    away_runs: float
    joint: np.ndarray          # joint[h, a] over regulation
    p_tie_reg: float
    extra_home_win: float

    @property
    def total_mean(self) -> float:
        return self.home_runs + self.away_runs

    def total_pmf(self) -> np.ndarray:
        n = self.joint.shape[0]
        tot = np.zeros(2 * n + 6)
        for h in range(n):
            for a in range(n):
                if h == a:
                    for extra, w in EXTRA_RUNS_DIST.items():
                        tot[h + a + extra] += self.joint[h, a] * w
                else:
                    tot[h + a] += self.joint[h, a]
        return tot

    def total_probs(self, line: float) -> tuple[float, float, float]:
        """(p_over, p_under, p_push) for a total line like 8.5 or 9."""
        pmf = self.total_pmf()
        k = np.arange(len(pmf))
        over = float(pmf[k > line].sum())
        under = float(pmf[k < line].sum())
        push = float(pmf[k == line].sum())
        return over, under, push

    def home_cover_prob(self, spread: float) -> tuple[float, float, float]:
        """(p_home_covers, p_away_covers, p_push) for a home spread like -1.5 or +1.5.

        Extra-inning games are treated as decided by exactly one run.
        """
        n = self.joint.shape[0]
        h_idx, a_idx = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
        margin = h_idx - a_idx
        reg = self.joint * (margin != 0)
        adj = margin + spread
        home = float(reg[adj > 0].sum())
        away = float(reg[adj < 0].sum())
        push = float(reg[adj == 0].sum())
        # extras: home wins by 1 with extra_home_win, loses by 1 otherwise
        for m, w in ((1, self.extra_home_win), (-1, 1 - self.extra_home_win)):
            v = m + spread
            mass = self.p_tie_reg * w
            if v > 0:
                home += mass
            elif v < 0:
                away += mass
            else:
                push += mass
        return home, away, push


def game_probs(mu_home: float, mu_away: float, dispersion: float | None = 4.0,
               extra_home_win: float = 0.52) -> GameProbs:
    ph = run_pmf(mu_home, dispersion)
    pa = run_pmf(mu_away, dispersion)
    joint = np.outer(ph, pa)
    home_reg = float(np.tril(joint, -1).sum())   # h > a
    away_reg = float(np.triu(joint, 1).sum())    # a > h
    tie = float(np.trace(joint))
    home = home_reg + tie * extra_home_win
    return GameProbs(
        home_win=home,
        away_win=1.0 - home,
        home_runs=float((ph * np.arange(len(ph))).sum()),
        away_runs=float((pa * np.arange(len(pa))).sum()),
        joint=joint,
        p_tie_reg=tie,
        extra_home_win=extra_home_win,
    )
