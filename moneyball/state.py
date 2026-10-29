"""The feature engine: one object that holds everything the model knows on a given day.

Both the historical backtest (fed from Retrosheet) and the live runner (fed
from MLB Stats API boxscores) push the same per-game records through this
class, so training features and live features are computed identically.

Usage, one day at a time:
    st.begin_day(day)
    feats = [st.game_features(g) for g in todays_games]   # before results
    for rec in todays_results: st.update(rec)             # after results

A game record (`update`) is a dict:
    home, away, home_sp, away_sp, site, home_runs, away_runs
    pitchers: list of (pitcher_id, pitching_side, pa, k, bb_hbp, hr, pitches)   side 1 = home
    batters:  list of (batter_id, batting_side, pitcher_throws, pa, woba_points)
`pitchers`/`batters` may be empty (no play data): then only runs/ratings update.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from functools import partial

import numpy as np
import pandas as pd

from .ratings import RatingConfig, RatingModel

WOBA_W = {"BB": 0.69, "HBP": 0.72, "1B": 0.88, "2B": 1.25, "3B": 1.58, "HR": 2.03, "ROE": 0.90}
PIT_KEYS = ("pa", "k", "bb", "hr")
BAT_KEYS = ("pa", "woba")
LG_KEYS = ("pa", "k", "bb", "hr", "woba", "g", "runs", "starts", "bf")


@dataclass
class FeatureConfig:
    half_life_days: float = 400.0
    prior_k: float = 70.0      # PA of league-average K% added to every pitcher
    prior_bb: float = 170.0
    prior_hr: float = 700.0
    prior_woba: float = 250.0
    prior_split: float = 300.0    # PA before a batter's own platoon split counts much
    prior_bf_start: float = 4.0   # starts of league-average depth
    prior_park_games: float = 160.0
    pen_days: int = 3


class Decayed:
    """Per-key accumulators that halve every `hl` days. Picklable."""

    def __init__(self, keys, hl):
        self.keys, self.hl = tuple(keys), hl
        self.v = defaultdict(partial(dict.fromkeys, self.keys, 0.0))
        self.t: dict = {}

    def get(self, who, day):
        acc = self.v[who]
        last = self.t.get(who)
        if last is not None and day != last:
            f = 0.5 ** ((day - last).days / self.hl)
            for k in self.keys:
                acc[k] *= f
        self.t[who] = day
        return acc

    def rename(self, mapping: dict, pos: int | None = None):
        """Re-key (e.g. Retrosheet ids -> MLBAM ids). pos: index inside tuple keys."""
        def m(k):
            if pos is None:
                return mapping.get(k, k)
            return tuple(mapping.get(x, x) if i == pos else x for i, x in enumerate(k))
        self.v = defaultdict(self.v.default_factory, {m(k): v for k, v in self.v.items()})
        self.t = {m(k): v for k, v in self.t.items()}


def weather_fields(temp=None, winddir="", windspeed=None, sky="", daynight=""):
    temp = pd.to_numeric(temp, errors="coerce")
    temp = np.nan if temp is None or pd.isna(temp) or temp <= 0 else float(temp)
    wd, ws = str(winddir or ""), pd.to_numeric(windspeed, errors="coerce")
    ws = 0.0 if ws is None or pd.isna(ws) or ws < 0 else float(ws)
    dome = str(sky or "") == "dome"
    return {"temp": temp, "wind_out": ws if wd.startswith("to") and not dome else 0.0,
            "wind_in": ws if wd.startswith("from") and not dome else 0.0, "dome": float(dome),
            "night": float(str(daynight) == "night")}


class FeatureState:
    def __init__(self, cfg: FeatureConfig | None = None, rating_cfg: RatingConfig | None = None,
                 hands: dict | None = None):
        self.cfg = cfg = cfg or FeatureConfig()
        hl = cfg.half_life_days
        self.pit = Decayed(PIT_KEYS, hl)
        self.bat = Decayed(BAT_KEYS, hl)
        self.pen = Decayed(PIT_KEYS, hl)          # per team, relievers only
        self.tbat = Decayed(BAT_KEYS, hl)         # per team batting
        self.bat_vs = Decayed(BAT_KEYS, hl)       # (batter, pitcher hand)
        self.tbat_vs = Decayed(BAT_KEYS, hl)      # (team, pitcher hand)
        self.lg_vs = Decayed(BAT_KEYS, hl)        # (bats, throws)
        self.start = Decayed(("n", "bf"), hl)     # per pitcher, as starter
        self.park = Decayed(("g", "runs"), hl * 2)
        self.lg = Decayed(LG_KEYS, hl)
        self.last_app: dict = {}
        self.pen_log = defaultdict(deque)         # team -> (date, pitches)
        self.ratings = RatingModel(rating_cfg or RatingConfig())
        self.hands = hands or {}
        self.day = None
        self.last_update = None

    # ---- league context for the day ------------------------------------------
    def begin_day(self, day: pd.Timestamp):
        day = pd.Timestamp(day)
        self.day = day
        L = self.lg.get("L", day)
        lpa = max(L["pa"], 1.0)
        has = L["pa"] > 0
        self.lk, self.lbb, self.lhr = (L["k"] / lpa, L["bb"] / lpa, L["hr"] / lpa) if has else (.225, .09, .032)
        self.lwoba = L["woba"] / lpa if has else 0.315
        self.lrpg = L["runs"] / L["g"] if L["g"] > 0 else 4.5
        self.lbf = L["bf"] / L["starts"] if L["starts"] > 0 else 22.0
        self.ratings.start_season(day.year)

    def _pitcher_rates(self, acc):
        c = self.cfg
        k = (acc["k"] + c.prior_k * self.lk) / (acc["pa"] + c.prior_k)
        bb = (acc["bb"] + c.prior_bb * self.lbb) / (acc["pa"] + c.prior_bb)
        hr = (acc["hr"] + c.prior_hr * self.lhr) / (acc["pa"] + c.prior_hr)
        fip = 13 * (hr - self.lhr) + 3 * (bb - self.lbb) - 2 * (k - self.lk)   # runs/PA above league
        return k, bb, hr, fip

    def _woba(self, acc):
        return (acc["woba"] + self.cfg.prior_woba * self.lwoba) / (acc["pa"] + self.cfg.prior_woba)

    def _lg_split(self, bats, throws):
        if bats == "B":
            bats = "L" if throws == "R" else "R"
        a = self.lg_vs.get((bats, throws), self.day)
        return (a["woba"] / a["pa"] - self.lwoba) if a["pa"] > 500 else 0.0

    def _split_woba(self, acc_all, acc_vs, base_delta):
        overall = self._woba(acc_all)
        own = acc_vs["woba"] / acc_vs["pa"] - overall if acc_vs["pa"] > 0 else 0.0
        w = acc_vs["pa"] / (acc_vs["pa"] + self.cfg.prior_split)
        return overall + w * own + (1 - w) * base_delta

    # ---- features for one upcoming game ---------------------------------------
    def game_features(self, home, away, home_sp, away_sp, site=None, weather: dict | None = None,
                      lineups: dict | None = None) -> dict:
        """lineups: {1: [home batter ids], 0: [away batter ids]} when known, else team-level."""
        day, cfg = self.day, self.cfg
        f = {"date": day, "season": day.year, "home": home, "away": away, "lg_rpg": self.lrpg}
        for side, sidx, team, sp, opp_sp in (("h", 1, home, home_sp, away_sp), ("a", 0, away, away_sp, home_sp)):
            acc = self.pit.get(sp, day)
            k, bb, hr, fip = self._pitcher_rates(acc)
            st = self.start.get(sp, day)
            f[f"{side}_sp_k"], f[f"{side}_sp_bb"], f[f"{side}_sp_hr"] = k, bb, hr
            f[f"{side}_sp_fip"] = fip
            f[f"{side}_sp_pa"] = acc["pa"]
            f[f"{side}_sp_depth"] = (st["bf"] + cfg.prior_bf_start * self.lbf) / (st["n"] + cfg.prior_bf_start)
            la = self.last_app.get(sp)
            f[f"{side}_sp_rest"] = min((day - la).days, 30) if la is not None else 30
            f[f"{side}_pen_fip"] = self._pitcher_rates(self.pen.get(team, day))[3]
            q = self.pen_log[team]
            while q and (day - q[0][0]).days > cfg.pen_days:
                q.popleft()
            f[f"{side}_pen_load"] = sum(p for d, p in q)
            f[f"{side}_pen_load1"] = sum(p for d, p in q if (day - d).days == 1)
            f[f"{side}_team_woba"] = self._woba(self.tbat.get(team, day))
            opp_throws = self.hands.get(opp_sp, ("R", "R"))[1]
            f[f"{side}_opp_sp_lefty"] = float(opp_throws == "L")
            f[f"{side}_team_woba_vs"] = self._split_woba(self.tbat.get(team, day),
                                                         self.tbat_vs.get((team, opp_throws), day), 0.0)
            players = (lineups or {}).get(sidx)
            if players:
                f[f"{side}_lineup_woba"] = float(np.mean([self._woba(self.bat.get(p, day)) for p in players]))
                f[f"{side}_lineup_woba_vs"] = float(np.mean([
                    self._split_woba(self.bat.get(p, day), self.bat_vs.get((p, opp_throws), day),
                                     self._lg_split(self.hands.get(p, ("R", "R"))[0], opp_throws))
                    for p in players]))
            else:
                f[f"{side}_lineup_woba"] = f[f"{side}_team_woba"]
                f[f"{side}_lineup_woba_vs"] = f[f"{side}_team_woba_vs"]
        pk = self.park.get(site, day)
        f["park_rpg"] = ((pk["runs"] + cfg.prior_park_games * self.lrpg * 2)
                         / (pk["g"] + cfg.prior_park_games) / (2 * self.lrpg))
        f.update(weather or weather_fields())
        gp = self.ratings.predict(home, away, home_sp, away_sp)
        f["rt_p_home"], f["rt_mu_h"], f["rt_mu_a"] = gp.home_win, gp.home_runs, gp.away_runs
        f["lg_woba"], f["lg_k"] = self.lwoba, self.lk
        return f

    # ---- learn from one finished game ----------------------------------------
    def update(self, rec: dict, day=None):
        day = pd.Timestamp(day) if day is not None else self.day
        teams = {1: rec["home"], 0: rec["away"]}
        sps = {1: rec["home_sp"], 0: rec["away_sp"]}
        pitchers, batters = rec.get("pitchers") or [], rec.get("batters") or []
        L = self.lg.get("L", day)
        if pitchers:
            bf = {0: 0, 1: 0}
            pen_pitches = {0: 0, 1: 0}
            for pid, pside, pa, k, bb, hr, pitches in pitchers:
                for acc in (self.pit.get(pid, day), L):
                    acc["pa"] += pa
                    acc["k"] += k
                    acc["bb"] += bb
                    acc["hr"] += hr
                if pid == sps[pside]:
                    bf[pside] += pa
                else:
                    a = self.pen.get(teams[pside], day)
                    a["pa"] += pa
                    a["k"] += k
                    a["bb"] += bb
                    a["hr"] += hr
                    pen_pitches[pside] += pitches
                if pa > 0:
                    self.last_app[pid] = day
            for bid, bside, throws, pa, w in batters:
                L["woba"] += w
                for acc in (self.bat.get(bid, day), self.tbat.get(teams[bside], day)):
                    acc["pa"] += pa
                    acc["woba"] += w
                bh = self.hands.get(bid, ("R", "R"))[0]
                if bh == "B":
                    bh = "L" if throws == "R" else "R"
                for acc in (self.bat_vs.get((bid, throws), day), self.tbat_vs.get((teams[bside], throws), day),
                            self.lg_vs.get((bh, throws), day)):
                    acc["pa"] += pa
                    acc["woba"] += w
            for s in (0, 1):
                st = self.start.get(sps[s], day)
                st["n"] += 1
                st["bf"] += bf[s]
                L["starts"] += 1
                L["bf"] += bf[s]
                self.pen_log[teams[s]].append((day, pen_pitches[s]))
        hr_, vr_ = rec.get("home_runs"), rec.get("away_runs")
        if hr_ is not None and vr_ is not None and not (pd.isna(hr_) or pd.isna(vr_)):
            L["g"] += 2
            L["runs"] += hr_ + vr_
            pk = self.park.get(rec.get("site"), day)
            pk["g"] += 1
            pk["runs"] += hr_ + vr_
            self.ratings.update(rec["home"], rec["away"], hr_, vr_, rec["home_sp"], rec["away_sp"])
        self.last_update = day

    # ---- id translation (Retrosheet -> MLBAM) ---------------------------------
    def rename_players(self, mapping: dict):
        for d in (self.pit, self.bat, self.start):
            d.rename(mapping)
        self.bat_vs.rename(mapping, pos=0)
        self.last_app = {mapping.get(k, k): v for k, v in self.last_app.items()}
        self.hands = {mapping.get(k, k): v for k, v in self.hands.items()}
        sp = self.ratings.sp
        self.ratings.sp = defaultdict(float, {mapping.get(k, k): v for k, v in sp.items()})
