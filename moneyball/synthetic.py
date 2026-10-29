"""Synthetic league with known true strengths, for tests and sanity checks.

The "market" sees the true probability plus noise and charges vig, so we can
check the pipeline end to end: a model that learns the truth should show a
better log loss than a noisy market and positive CLV; a useless model should not.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .odds import prob_to_american
from .runs import game_probs


def _price_pair(p: float, vig: float = 0.045) -> tuple[int, int]:
    # spread the overround proportionally
    ph, pa = p * (1 + vig / 2), (1 - p) * (1 + vig / 2)
    return round(prob_to_american(min(ph, 0.97))), round(prob_to_american(min(pa, 0.97)))


def make_league(n_teams: int = 30, seasons=(2022, 2023, 2024), games_per_day: int = 15,
                days_per_season: int = 160, market_noise: float = 0.035, seed: int = 7,
                league_runs: float = 4.5, hfa: float = 0.035, team_sd: float = 0.10,
                sp_sd: float = 0.15, sp_per_team: int = 5) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    teams = [f"T{i:02d}" for i in range(n_teams)]
    off = dict(zip(teams, rng.normal(0, team_sd, n_teams)))
    dfn = dict(zip(teams, rng.normal(0, team_sd, n_teams)))
    sps = {t: [f"{t}-SP{j}" for j in range(sp_per_team)] for t in teams}
    sp_q = {s: rng.normal(0, sp_sd) for t in teams for s in sps[t]}
    rows, gid = [], 0
    for season in seasons:
        # strengths drift between seasons
        for t in teams:
            off[t] = 0.6 * off[t] + rng.normal(0, team_sd * 0.8)
            dfn[t] = 0.6 * dfn[t] + rng.normal(0, team_sd * 0.8)
        start = pd.Timestamp(f"{season}-04-01")
        for d in range(days_per_season):
            day = start + pd.Timedelta(days=d)
            order = rng.permutation(teams)
            for k in range(min(games_per_day, n_teams // 2)):
                h, a = order[2 * k], order[2 * k + 1]
                hsp, asp = sps[h][d % sp_per_team], sps[a][d % sp_per_team]
                mu_h = league_runs * np.exp(hfa + off[h] - dfn[a] - 0.55 * sp_q[asp])
                mu_a = league_runs * np.exp(off[a] - dfn[h] - 0.55 * sp_q[hsp])
                gp = game_probs(mu_h, mu_a)
                # negative binomial draws matching the model's dispersion
                r = 4.0
                hs = rng.negative_binomial(r, r / (r + mu_h))
                as_ = rng.negative_binomial(r, r / (r + mu_a))
                if hs == as_:
                    (hs := hs + 1) if rng.random() < 0.52 else (as_ := as_ + 1)
                p_mkt = float(np.clip(gp.home_win + rng.normal(0, market_noise), 0.08, 0.92))
                close = float(np.clip(gp.home_win + rng.normal(0, market_noise / 2), 0.08, 0.92))
                hml, aml = _price_pair(p_mkt)
                hc, ac = _price_pair(close)
                line = round(gp.total_mean * 2) / 2
                rows.append({"game_id": gid, "date": day.strftime("%Y-%m-%d"), "season": season,
                             "home": h, "away": a, "home_score": hs, "away_score": as_,
                             "home_sp": hsp, "away_sp": asp, "p_true_home": gp.home_win,
                             "home_ml": hml, "away_ml": aml, "home_ml_close": hc,
                             "away_ml_close": ac, "total_line": line, "over_price": -110,
                             "under_price": -110})
                gid += 1
    return pd.DataFrame(rows)
