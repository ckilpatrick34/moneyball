"""The betting models: feature derivation, training, saving and scoring.

`derive` turns raw FeatureState output (+ market columns) into model inputs;
the backtest (research/mlb_strategy.py) and the live runner both use it.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .odds import logit, sigmoid

# Known when opening lines post (no lineups).
ML_FEATS = ["sp_fip", "sp_k", "sp_bb", "sp_depth_x", "sp_new", "pen_fip", "pen_load", "pen_load1",
            "team_bat", "rt", "rest"]
# Live totals use forecast temperature only: wind relative to the park isn't forecastable well
# and added nothing once forecast noise was included (FINDINGS.md).
TOT_FEATS = ["sp_fip_sum", "sp_k_sum", "sp_bb_sum", "pen_fip_sum", "bat_sum", "pen_load_sum",
             "rt_tot", "park_rpg", "temp_c", "dome", "night", "depth_sum", "ln_line"]


def derive(d: pd.DataFrame) -> pd.DataFrame:
    d = d.copy()
    d["sp_fip"] = d.a_sp_fip - d.h_sp_fip
    d["sp_k"] = d.h_sp_k - d.a_sp_k
    d["sp_bb"] = d.a_sp_bb - d.h_sp_bb
    d["sp_depth_x"] = d.h_sp_depth * (-d.h_sp_fip) - d.a_sp_depth * (-d.a_sp_fip)
    d["sp_new"] = (d.h_sp_pa < 100).astype(int) - (d.a_sp_pa < 100).astype(int)
    d["pen_fip"] = d.a_pen_fip - d.h_pen_fip
    d["pen_load"] = d.a_pen_load - d.h_pen_load
    d["pen_load1"] = d.a_pen_load1 - d.h_pen_load1
    d["team_bat"] = d.h_team_woba - d.a_team_woba
    d["team_bat_vs"] = d.h_team_woba_vs - d.a_team_woba_vs
    d["rt"] = d.rt_p_home.map(logit)
    d["rest"] = d.h_sp_rest.clip(upper=10) - d.a_sp_rest.clip(upper=10)
    d["sp_fip_sum"] = d.h_sp_fip + d.a_sp_fip
    d["sp_k_sum"] = d.h_sp_k + d.a_sp_k
    d["sp_bb_sum"] = d.h_sp_bb + d.a_sp_bb
    d["pen_fip_sum"] = d.h_pen_fip + d.a_pen_fip
    d["bat_sum"] = d.h_team_woba + d.a_team_woba
    d["bat_vs_sum"] = d.h_team_woba_vs + d.a_team_woba_vs
    d["lefty_sp_sum"] = d.h_opp_sp_lefty + d.a_opp_sp_lefty
    d["pen_load_sum"] = d.h_pen_load + d.a_pen_load
    d["rt_tot"] = np.log(d.rt_mu_h + d.rt_mu_a)
    d["temp_c"] = d.temp.fillna(72).clip(40, 100)
    d["depth_sum"] = d.h_sp_depth + d.a_sp_depth
    if "total_open" in d:
        d["ln_line"] = np.log(d.total_open)
    return d


class LogitModel:
    """Standardised L2 logistic regression, optionally with a market-logit column first."""

    def __init__(self, feats, market_col=None, C=0.05):
        self.feats, self.market_col, self.C = list(feats), market_col, C

    def _X(self, d):
        X = (d[self.feats].to_numpy(float) - self.mean) / self.scale
        return np.column_stack([d[self.market_col].to_numpy(float), X]) if self.market_col else X

    def fit(self, d, y):
        from sklearn.linear_model import LogisticRegression
        self.mean = d[self.feats].to_numpy(float).mean(0)
        self.scale = d[self.feats].to_numpy(float).std(0)
        self.scale[self.scale == 0] = 1.0
        m = LogisticRegression(C=self.C).fit(self._X(d), y)
        self.coef, self.intercept = m.coef_[0], float(m.intercept_[0])
        return self

    def predict(self, d) -> np.ndarray:
        z = self._X(d) @ self.coef + self.intercept
        return 1 / (1 + np.exp(-z))

    def to_json(self) -> dict:
        return {"feats": self.feats, "market_col": self.market_col, "C": self.C, "mean": self.mean.tolist(),
                "scale": self.scale.tolist(), "coef": self.coef.tolist(), "intercept": self.intercept}

    @classmethod
    def from_json(cls, j) -> "LogitModel":
        m = cls(j["feats"], j["market_col"], j["C"])
        m.mean, m.scale = np.array(j["mean"]), np.array(j["scale"])
        m.coef, m.intercept = np.array(j["coef"]), j["intercept"]
        return m


def save_models(path: Path, ml: LogitModel, tot: LogitModel, meta: dict):
    Path(path).write_text(json.dumps({"ml": ml.to_json(), "totals": tot.to_json(), "meta": meta}, indent=1))


def load_models(path: Path):
    j = json.loads(Path(path).read_text())
    return LogitModel.from_json(j["ml"]), LogitModel.from_json(j["totals"]), j["meta"]


def blend(p_model, p_market, w_market=0.5):
    return sigmoid((1 - w_market) * logit(p_model) + w_market * logit(p_market))
