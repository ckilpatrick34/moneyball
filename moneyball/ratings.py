"""Point-in-time team and starting-pitcher run ratings.

Every team has an offense and a run-prevention rating on a log scale
(0 = league average). MLB starting pitchers get their own rating, which
covers the share of innings they usually throw. Ratings only ever see games
that finished before the game being predicted, so a walk-forward backtest
has no look-ahead leakage by construction.

Expected home runs = L * exp(hfa + off[home] - prevent(away, away_sp)).
After each game the ratings move along the gradient of the
negative-binomial/Poisson log-likelihood, which is a run-based Elo.

The same model runs college baseball (no pitcher data, more teams, more
regression) by changing the config.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field

from .runs import GameProbs, game_probs


@dataclass
class RatingConfig:
    league_runs: float = 4.5        # starting runs/team/game, then tracked online
    league_runs_lr: float = 0.002
    hfa: float = 0.035              # log-runs bonus for the home offense
    lr_off: float = 0.012
    lr_def: float = 0.012
    lr_sp: float = 0.025
    sp_share: float = 0.55          # share of a game's run prevention owed to the starter
    sp_min_ip_for_full_update: float = 5.0
    season_carryover: float = 0.6   # fraction of a rating kept into the next season
    sp_carryover: float = 0.75
    dispersion: float | None = 4.0
    extra_home_win: float = 0.52
    rating_clip: float = 0.6

    @classmethod
    def mlb(cls) -> "RatingConfig":
        return cls()

    @classmethod
    def college(cls) -> "RatingConfig":
        # Bigger talent gaps and home edge, aluminium bats. Tuned on 2022-2026 D1
        # results by walk-forward log loss (research/ncaa_ratings.py).
        return cls(league_runs=6.5, hfa=0.12, lr_off=0.05, lr_def=0.05, lr_sp=0.0,
                   sp_share=0.0, season_carryover=0.85, dispersion=3.0,
                   extra_home_win=0.53, rating_clip=1.2)


@dataclass
class RatingModel:
    cfg: RatingConfig = field(default_factory=RatingConfig)
    off: dict = field(default_factory=lambda: defaultdict(float))
    prevent: dict = field(default_factory=lambda: defaultdict(float))
    sp: dict = field(default_factory=lambda: defaultdict(float))
    games_seen: dict = field(default_factory=lambda: defaultdict(int))
    season: int | None = None

    def __post_init__(self):
        self.L = self.cfg.league_runs

    # ---- season handling -------------------------------------------------
    def start_season(self, season: int) -> None:
        if self.season is not None and season != self.season:
            c, cs = self.cfg.season_carryover, self.cfg.sp_carryover
            for d in (self.off, self.prevent):
                for k in d:
                    d[k] *= c
            for k in self.sp:
                self.sp[k] *= cs
            for k in self.games_seen:
                self.games_seen[k] = int(self.games_seen[k] * c)
        self.season = season

    # ---- prediction ------------------------------------------------------
    def _prevention(self, team: str, sp_id) -> float:
        # Team rating covers the staff overall; the starter adds his deviation
        # from that, weighted by the share of the game he usually pitches.
        if sp_id is None or self.cfg.sp_share <= 0:
            return self.prevent[team]
        return self.prevent[team] + self.cfg.sp_share * self.sp[sp_id]

    def expected_runs(self, home: str, away: str, home_sp=None, away_sp=None,
                      neutral: bool = False) -> tuple[float, float]:
        hfa = 0.0 if neutral else self.cfg.hfa
        mu_h = self.L * math.exp(hfa + self.off[home] - self._prevention(away, away_sp))
        mu_a = self.L * math.exp(self.off[away] - self._prevention(home, home_sp))
        return mu_h, mu_a

    def predict(self, home: str, away: str, home_sp=None, away_sp=None,
                neutral: bool = False) -> GameProbs:
        mu_h, mu_a = self.expected_runs(home, away, home_sp, away_sp, neutral)
        return game_probs(mu_h, mu_a, self.cfg.dispersion,
                          0.5 if neutral else self.cfg.extra_home_win)

    # ---- update ----------------------------------------------------------
    def _clip(self, x: float) -> float:
        c = self.cfg.rating_clip
        return max(-c, min(c, x))

    def update(self, home: str, away: str, home_score: float, away_score: float,
               home_sp=None, away_sp=None, neutral: bool = False,
               home_sp_ip: float | None = None, away_sp_ip: float | None = None) -> None:
        cfg = self.cfg
        mu_h, mu_a = self.expected_runs(home, away, home_sp, away_sp, neutral)
        # Scaled score residuals; dividing by L keeps learning rates unit-free.
        g_h = (home_score - mu_h) / self.L
        g_a = (away_score - mu_a) / self.L

        for att, dfn, sp_id, sp_ip, g in ((home, away, away_sp, away_sp_ip, g_h),
                                          (away, home, home_sp, home_sp_ip, g_a)):
            self.off[att] = self._clip(self.off[att] + cfg.lr_off * g)
            s = cfg.sp_share if sp_id is not None and cfg.sp_share > 0 else 0.0
            self.prevent[dfn] = self._clip(self.prevent[dfn] - cfg.lr_def * g)
            if s:
                # Credit the starter in proportion to how long he actually pitched.
                w = 1.0 if sp_ip is None else min(1.0, sp_ip / cfg.sp_min_ip_for_full_update)
                self.sp[sp_id] = self._clip(self.sp[sp_id] - cfg.lr_sp * s * w * g)

        self.L += cfg.league_runs_lr * ((home_score + away_score) / 2 - self.L)
        self.games_seen[home] += 1
        self.games_seen[away] += 1
