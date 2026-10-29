"""Walk-forward backtest against market prices.

Input is one row per game (see README for the column schema). Days are
processed in order: predict every game with ratings built only from earlier
days, decide bets at the recorded prices, settle, then update ratings with
that day's results.

The numbers that matter, in order:
  1. CLV: do our bets beat the closing line? (needs closing prices)
  2. Log loss vs the no-vig market: is the model at least as sharp as the book?
  3. ROI: noisy; thousands of bets are needed before it means much.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .odds import american_to_decimal, clv, expected_value_with_push, fair_probs, logit, sigmoid
from .ratings import RatingConfig, RatingModel
from .sizing import SizingConfig, kelly


@dataclass
class BetConfig:
    markets: tuple = ("ml", "total")
    min_ev: float = 0.025            # require 2.5% EV at the price offered
    market_weight: float = 0.5       # blend toward the no-vig market in logit space
    max_disagreement: float = 0.12   # bigger gaps usually mean bad data (wrong SP, injury)
    min_games_seen: int = 15         # per team, this season and carried over
    skip_first_season: bool = True   # ratings are cold in the first season of data
    bankroll: float = 1000.0
    sizing: SizingConfig = field(default_factory=SizingConfig)


def _num(row, col):
    v = row.get(col)
    if v is None or (isinstance(v, float) and math.isnan(v)) or v == "":
        return None
    return float(v)


def _opt(row, col):
    v = row.get(col)
    if v is None or (isinstance(v, float) and math.isnan(v)) or v == "":
        return None
    return v


def blend(p_model: float, p_market: float, market_weight: float) -> float:
    return sigmoid((1 - market_weight) * logit(p_model) + market_weight * logit(p_market))


@dataclass
class BacktestResult:
    predictions: pd.DataFrame
    bets: pd.DataFrame
    summary: dict

    def report(self) -> str:
        s = self.summary
        lines = ["Backtest summary", "-" * 40]
        for k, v in s.items():
            if isinstance(v, float):
                lines.append(f"{k:<28} {v:,.4f}")
            else:
                lines.append(f"{k:<28} {v}")
        if not self.bets.empty:
            lines.append("")
            lines.append("By season / market:")
            g = self.bets.groupby(["season", "market"]).agg(
                bets=("profit", "size"), staked=("stake", "sum"), profit=("profit", "sum"),
                avg_clv=("clv", "mean"))
            g["roi"] = g["profit"] / g["staked"]
            lines.append(g.to_string(float_format=lambda x: f"{x:,.3f}"))
        return "\n".join(lines)


def _consider(bets, bankroll_today, cfg, base, market, side, p_model, p_market, p_push,
              price, outcome, close_fair, line=None):
    """Append a bet to `bets` if it clears the thresholds. outcome: 1 win, 0 loss, None push."""
    if price is None or p_market is None:
        return
    if abs(p_model - p_market) > cfg.max_disagreement:
        return
    p = blend(p_model, p_market, cfg.market_weight)
    ev = expected_value_with_push(p, p_push, price)
    if ev < cfg.min_ev:
        return
    f = kelly(p, price, p_push) * cfg.sizing.kelly_fraction
    amt = bankroll_today * min(f, cfg.sizing.max_bet_pct)
    if amt < cfg.sizing.min_bet:
        return
    d = american_to_decimal(price)
    profit = 0.0 if outcome is None else (amt * (d - 1) if outcome == 1 else -amt)
    bets.append({**base, "market": market, "side": side, "line": line, "price": price,
                 "p_model": p_model, "p_market": p_market, "p_bet": p, "ev": ev,
                 "stake": round(amt, 2), "result": {1: "W", 0: "L", None: "P"}[outcome],
                 "profit": profit,
                 "clv": clv(price, close_fair) if close_fair is not None else np.nan})


def run_backtest(games: pd.DataFrame, rating_cfg: RatingConfig | None = None,
                 bet_cfg: BetConfig | None = None) -> BacktestResult:
    rating_cfg = rating_cfg or RatingConfig()
    cfg = bet_cfg or BetConfig()
    model = RatingModel(rating_cfg)

    g = games.copy()
    g["date"] = pd.to_datetime(g["date"])
    if "season" not in g:
        g["season"] = g["date"].dt.year
    sort_cols = ["date"] + (["game_id"] if "game_id" in g else [])
    g = g.sort_values(sort_cols).reset_index(drop=True)
    first_season = int(g["season"].min())

    preds, bets = [], []
    bankroll = cfg.bankroll
    peak, max_dd = bankroll, 0.0

    for date, day in g.groupby("date", sort=True):
        season = int(day["season"].iloc[0])
        model.start_season(season)
        bankroll_today = bankroll
        day_bets: list = []

        for _, r in day.iterrows():
            home, away = r["home"], r["away"]
            neutral = bool(_opt(r, "neutral") or False)
            hsp, asp = _opt(r, "home_sp"), _opt(r, "away_sp")
            gp = model.predict(home, away, hsp, asp, neutral)
            hs, as_ = float(r["home_score"]), float(r["away_score"])
            home_won = 1 if hs > as_ else 0
            base = {"date": date.date(), "season": season, "home": home, "away": away}

            rec = {**base, "p_home": gp.home_win, "mu_home": gp.home_runs,
                   "mu_away": gp.away_runs, "home_won": home_won, "total": hs + as_}

            hml, aml = _num(r, "home_ml"), _num(r, "away_ml")
            p_mkt_home = fair_probs([hml, aml])[0] if hml and aml else None
            rec["p_market_home"] = p_mkt_home
            preds.append(rec)

            eligible = (min(model.games_seen[home], model.games_seen[away]) >= cfg.min_games_seen
                        and not (cfg.skip_first_season and season == first_season))
            if not eligible:
                continue

            if "ml" in cfg.markets and p_mkt_home is not None:
                hc, ac = _num(r, "home_ml_close"), _num(r, "away_ml_close")
                close_home = fair_probs([hc, ac])[0] if hc and ac else None
                _consider(day_bets, bankroll_today, cfg, base, "ml", "home", gp.home_win,
                          p_mkt_home, 0.0, hml, home_won, close_home)
                _consider(day_bets, bankroll_today, cfg, base, "ml", "away", gp.away_win,
                          1 - p_mkt_home, 0.0, aml, 1 - home_won,
                          None if close_home is None else 1 - close_home)

            line, op, up = _num(r, "total_line"), _num(r, "over_price"), _num(r, "under_price")
            if "total" in cfg.markets and line is not None and op and up:
                p_over, p_under, p_push = gp.total_probs(line)
                mk_over, mk_under = fair_probs([op, up])
                # market probs exclude pushes; rescale ours to the same footing for comparison
                nonpush = max(1e-9, 1 - p_push)
                tot = hs + as_
                o_out = None if tot == line else int(tot > line)
                u_out = None if tot == line else int(tot < line)
                lc, oc, uc = (_num(r, "total_line_close"), _num(r, "over_close"),
                              _num(r, "under_close"))
                close_over = close_under = None
                if oc and uc and (lc is None or lc == line):
                    close_over, close_under = fair_probs([oc, uc])
                _consider(day_bets, bankroll_today, cfg, base, "total", "over", p_over / nonpush,
                          mk_over, 0.0, op, o_out, close_over, line)
                _consider(day_bets, bankroll_today, cfg, base, "total", "under",
                          p_under / nonpush, mk_under, 0.0, up, u_out, close_under, line)

        # settle, then learn from today's results
        for b in day_bets:
            bankroll += b["profit"]
        bets.extend(day_bets)
        peak = max(peak, bankroll)
        max_dd = max(max_dd, (peak - bankroll) / peak if peak else 0.0)

        for _, r in day.iterrows():
            model.update(r["home"], r["away"], float(r["home_score"]), float(r["away_score"]),
                         _opt(r, "home_sp"), _opt(r, "away_sp"), bool(_opt(r, "neutral") or False),
                         _num(r, "home_sp_ip"), _num(r, "away_sp_ip"))

    pred_df = pd.DataFrame(preds)
    bet_df = pd.DataFrame(bets)
    summary = _summarise(pred_df, bet_df, cfg, bankroll, max_dd, first_season)
    return BacktestResult(pred_df, bet_df, summary)


def _log_loss(p, y):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    y = np.asarray(y, float)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def _summarise(pred, bets, cfg, bankroll, max_dd, first_season) -> dict:
    s: dict = {}
    scored = pred[pred["season"] > first_season] if cfg.skip_first_season else pred
    s["games_scored"] = len(scored)
    if len(scored):
        s["model_log_loss"] = _log_loss(scored["p_home"], scored["home_won"])
        s["model_brier"] = float(((scored["p_home"] - scored["home_won"]) ** 2).mean())
        s["home_win_rate"] = float(scored["home_won"].mean())
        s["model_avg_p_home"] = float(scored["p_home"].mean())
        with_mkt = scored.dropna(subset=["p_market_home"])
        if len(with_mkt):
            s["games_with_market"] = len(with_mkt)
            s["model_log_loss_vs_mkt"] = _log_loss(with_mkt["p_home"], with_mkt["home_won"])
            s["market_log_loss"] = _log_loss(with_mkt["p_market_home"], with_mkt["home_won"])
    s["bets"] = len(bets)
    if len(bets):
        s["staked"] = float(bets["stake"].sum())
        s["profit"] = float(bets["profit"].sum())
        s["roi"] = s["profit"] / s["staked"]
        s["win_rate"] = float((bets["result"] == "W").mean())
        s["avg_ev_at_bet"] = float(bets["ev"].mean())
        if bets["clv"].notna().any():
            s["avg_clv"] = float(bets["clv"].mean())
            s["pct_beat_close"] = float((bets["clv"].dropna() > 0).mean())
    s["final_bankroll"] = float(bankroll)
    s["max_drawdown"] = float(max_dd)
    return s
