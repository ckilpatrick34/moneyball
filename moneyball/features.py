"""Point-in-time MLB features from Retrosheet play-by-play.

Converts Retrosheet plate appearances into per-game records and pushes them
through `state.FeatureState` day by day: features for a day are read before
any of that day's results are added, so nothing from the game day leaks in
(except the weather fields, which stand in for a same-day forecast). The live
runner feeds the same class from MLB Stats API boxscores.
"""

from __future__ import annotations

import pandas as pd

from .ratings import RatingConfig
from .state import WOBA_W, FeatureConfig, FeatureState, weather_fields  # noqa: F401


def retro_records(games: pd.DataFrame, pa: pd.DataFrame, hands: dict) -> dict:
    """retro_id -> game record (see state.py) built from Retrosheet PAs."""
    p = pa[pa["outcome"] != "CI"].copy()
    p["pside"] = 1 - p["bat_side"]
    p["k"] = (p["outcome"] == "K").astype(int)
    p["bb"] = p["outcome"].isin(["BB", "HBP"]).astype(int)
    p["hr"] = (p["outcome"] == "HR").astype(int)
    p["w"] = p["outcome"].map(WOBA_W).fillna(0.0)
    p["throws"] = p["pitcher"].map(lambda x: hands.get(x, ("R", "R"))[1])
    pit = p.groupby(["retro_id", "pitcher", "pside"], sort=False).agg(
        pa=("k", "size"), k=("k", "sum"), bb=("bb", "sum"), hr=("hr", "sum"), pitches=("pitches", "sum")).reset_index()
    bat = p.groupby(["retro_id", "batter", "bat_side", "throws"], sort=False).agg(
        pa=("w", "size"), w=("w", "sum")).reset_index()
    pit_by = {k: list(v[["pitcher", "pside", "pa", "k", "bb", "hr", "pitches"]].itertuples(index=False, name=None))
              for k, v in pit.groupby("retro_id", sort=False)}
    bat_by = {k: list(v[["batter", "bat_side", "throws", "pa", "w"]].itertuples(index=False, name=None))
              for k, v in bat.groupby("retro_id", sort=False)}
    recs = {}
    for g in games.itertuples(index=False):
        recs[g.retro_id] = {"home": g.home, "away": g.away, "home_sp": g.home_sp, "away_sp": g.vis_sp,
                            "site": g.site, "home_runs": g.home_runs, "away_runs": g.vis_runs,
                            "pitchers": pit_by.get(g.retro_id, []), "batters": bat_by.get(g.retro_id, [])}
    return recs


def build(games: pd.DataFrame, pa: pd.DataFrame, lineups: pd.DataFrame,
          cfg: FeatureConfig | None = None, rating_cfg: RatingConfig | None = None,
          hands: dict | None = None, return_state: bool = False):
    """games needs: retro_id, date, home, away, home_sp, vis_sp, site, weather cols, runs."""
    hands = hands or {}
    st = FeatureState(cfg, rating_cfg, hands)
    recs = retro_records(games, pa, hands)
    lu = {(k, s): list(v.sort_values("bat_pos")["player"])
          for (k, s), v in lineups.groupby(["retro_id", "side"], sort=False)}
    out = []
    games = games.sort_values(["date", "game_num"]).reset_index(drop=True)
    for day, dg in games.groupby("date", sort=True):
        st.begin_day(day)
        for g in dg.itertuples(index=False):
            gd = g._asdict()
            wx = weather_fields(gd.get("temp"), gd.get("winddir"), gd.get("windspeed"), gd.get("sky"), gd.get("daynight"))
            lineup = {1: lu.get((g.retro_id, 1)), 0: lu.get((g.retro_id, 0))}
            f = st.game_features(g.home, g.away, g.home_sp, g.vis_sp, gd.get("site"), wx, lineup)
            f["retro_id"] = g.retro_id
            out.append(f)
        for g in dg.itertuples(index=False):
            st.update(recs[g.retro_id], day)
    df = pd.DataFrame(out)
    return (df, st) if return_state else df
