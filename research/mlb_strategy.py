"""Reproducible MLB strategy backtest (2022-2025 out of sample).

    python research/mlb_strategy.py --retro <chadwick retrosheet>/seasons --odds mlb_odds_dataset.json

Steps: parse Retrosheet -> point-in-time features -> join SBR odds ->
walk-forward logistic models (train on all earlier seasons) -> bet at
opening prices -> quarter-Kelly bankroll, CLV vs the no-vig closing line.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from moneyball.data import retrosheet, sbr_odds  # noqa: E402
from moneyball.features import build  # noqa: E402
from moneyball.model import derive  # noqa: E402
from moneyball.odds import american_to_decimal as dec, fair_probs, logit, sigmoid  # noqa: E402

ML_FEATS = ["sp_fip", "sp_k", "sp_bb", "sp_depth_x", "sp_new", "pen_fip", "pen_load", "pen_load1",
            "team_bat", "rt", "rest"]          # all known when opening lines post
PLATOON_ML = ["team_bat_vs"]                   # team wOBA vs the opposing starter's hand
PLATOON_TOT = ["bat_vs_sum", "lefty_sp_sum"]
TOT_FEATS = ["sp_fip_sum", "sp_k_sum", "sp_bb_sum", "pen_fip_sum", "bat_sum", "pen_load_sum",
             "rt_tot", "park_rpg", "temp_c", "wind_out", "wind_in", "dome", "night", "depth_sum",
             "ln_line"]
TEST_SEASONS = [2022, 2023, 2024, 2025]


def load(retro_root: str, odds_path: str, cache: Path) -> pd.DataFrame:
    if cache.exists():
        return pd.read_pickle(cache)
    g, pa, lu = retrosheet.parse_seasons(Path(retro_root), range(2019, 2026))
    hands = retrosheet.parse_hands(Path(retro_root), range(2019, 2026))
    g = g[g.gametype.fillna("regular") != "allstar"].rename(columns={"hometeam": "home", "visteam": "away"})
    f = build(g, pa, lu, hands=hands).merge(g[["retro_id", "game_num", "gametype"]], on="retro_id")
    f = f[f.gametype.fillna("regular") == "regular"].sort_values(["date", "game_num"])
    sb = sbr_odds.load(Path(odds_path))
    f["k"] = f.groupby(["date", "home", "away"]).cumcount()
    sb["k"] = sb.groupby(["date", "home", "away"]).cumcount()
    d = sb.merge(f, on=["date", "home", "away", "k"], how="inner")
    d = d.dropna(subset=["home_ml_open", "away_ml_open", "home_ml_close", "away_ml_close"])
    d["p_open"] = [fair_probs([h, a])[0] for h, a in zip(d.home_ml_open, d.away_ml_open)]
    d["p_close"] = [fair_probs([h, a])[0] for h, a in zip(d.home_ml_close, d.away_ml_close)]
    d["lo_open"], d["lo_close"] = d.p_open.map(logit), d.p_close.map(logit)
    d["y"] = (d.home_score > d.away_score).astype(int)
    d = derive(d)
    d["temp_actual"], d["wind_out_actual"], d["wind_in_actual"] = d.temp_c, d.wind_out, d.wind_in
    d.to_pickle(cache)
    return d


def walk_forward(d, feats, target, market_logit, C=0.05):
    out = []
    for s in TEST_SEASONS:
        tr, te = d[d.season < s], d[d.season == s].copy()
        sc = StandardScaler().fit(tr[feats])
        m = LogisticRegression(C=C).fit(sc.transform(tr[feats]) if market_logit is None else
                                        np.column_stack([tr[market_logit], sc.transform(tr[feats])]), tr[target])
        X = sc.transform(te[feats]) if market_logit is None else np.column_stack([te[market_logit], sc.transform(te[feats])])
        te["p"] = m.predict_proba(X)[:, 1]
        out.append(te)
    return pd.concat(out)


def candidates_ml(t, w_mkt, price_tag):
    rows = []
    for r in t.itertuples():
        p = sigmoid((1 - w_mkt) * logit(r.p) + w_mkt * r.lo_open)
        for side, pp, price, pc, won in (("home", p, getattr(r, f"home_ml_{price_tag}"), r.p_close, r.y),
                                         ("away", 1 - p, getattr(r, f"away_ml_{price_tag}"), 1 - r.p_close, 1 - r.y)):
            if price is None or np.isnan(price):
                continue
            rows.append(dict(date=r.date, season=r.season, market="ML", game=f"{r.away}@{r.home}", side=side,
                             price=price, p=pp, won=won, push=False, clv=dec(price) * pc - 1))
    return pd.DataFrame(rows)


def candidates_tot(d, w_mkt, price_tag):
    d = d.dropna(subset=["total_open", "over_open", "under_open", f"over_{price_tag}", f"under_{price_tag}"]).copy()
    d["tot"] = d.home_score + d.away_score
    d["po"] = [fair_probs([o, u])[0] for o, u in zip(d.over_open, d.under_open)]
    d["lo_po"] = d.po.map(logit)
    d["ln_line"] = np.log(d.total_open)
    d["yo"] = (d.tot > d.total_open).astype(int)
    train = d[d.tot != d.total_open]
    t = walk_forward(train, TOT_FEATS, "yo", "lo_po")
    # score pushes too (they were excluded from training only)
    t = d.merge(t[["retro_id", "p"]], on="retro_id", how="inner")
    rows = []
    for r in t.itertuples():
        p = sigmoid((1 - w_mkt) * logit(r.p) + w_mkt * r.lo_po)
        same = r.total_close == r.total_open and not np.isnan(r.over_close)
        pc = fair_probs([r.over_close, r.under_close])[0] if same else np.nan
        push = r.tot == r.total_open
        for side, pp, price, won, pcs, mv in (("over", p, getattr(r, f"over_{price_tag}"), r.tot > r.total_open, pc, r.total_close - r.total_open),
                                             ("under", 1 - p, getattr(r, f"under_{price_tag}"), r.tot < r.total_open, 1 - pc, r.total_open - r.total_close)):
            rows.append(dict(date=r.date, season=r.season, market="TOT", game=f"{r.away}@{r.home} {r.total_open}",
                             side=side, price=price, p=pp, won=int(won), push=bool(push),
                             clv=dec(price) * pcs - 1 if same else np.nan, line_move=mv))
    return pd.DataFrame(rows)


def simulate(c, min_ev, bankroll=10_000, kelly_frac=0.25, cap=0.02):
    c = c.copy()
    c["ev"] = c.p * c.price.map(dec) - 1
    c = c[c.ev > min_ev].sort_values("date")
    bank, peak, mdd, rows = bankroll, bankroll, 0.0, []
    for day, bets in c.groupby("date"):
        start = bank
        for r in bets.itertuples():
            b = dec(r.price) - 1
            f = max(0.0, (b * r.p - (1 - r.p)) / b) * kelly_frac
            stake = start * min(f, cap)
            pnl = 0.0 if r.push else (stake * b if r.won else -stake)
            bank += pnl
            rows.append({**r._asdict(), "stake": stake, "pnl": pnl, "flat": 0.0 if r.push else (b if r.won else -1.0)})
        peak = max(peak, bank)
        mdd = max(mdd, 1 - bank / peak)
    b = pd.DataFrame(rows)
    return b, bank, mdd


def report(name, b, bank, mdd, start=10_000):
    if b.empty:
        return f"{name}: no bets"
    flat = b["flat"]
    t = flat.mean() / (flat.std() / np.sqrt(len(flat)))
    seas = b.groupby("season")["flat"].agg(["size", "mean"])
    s = (f"{name}\n  bets {len(b)}, flat ROI {flat.mean():+.2%} (t={t:+.2f}), avg CLV {b.clv.mean():+.2%}, "
         f"beat close {(b.clv.dropna() > 0).mean():.0%}\n  quarter-Kelly: $10,000 -> ${bank:,.0f}, "
         f"max drawdown {mdd:.0%}\n  by season: " + ", ".join(f"{k}: {int(v['size'])} bets {v['mean']:+.1%}" for k, v in seas.iterrows()))
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--retro", required=True)
    ap.add_argument("--odds", required=True)
    ap.add_argument("--cache", default="data/mlb_model_df.pkl")
    ap.add_argument("--out", default="data/strategy_bets.csv")
    a = ap.parse_args()
    d = load(a.retro, a.odds, Path(a.cache))
    ml = walk_forward(d, ML_FEATS, "y", None)
    lines, all_bets = [], []
    for price_tag in ("open_best", "open"):
        c_ml = candidates_ml(ml, 0.5, price_tag)
        c_tot = candidates_tot(d, 0.5, price_tag)
        for name, c, thr in (("Moneyline", c_ml, 0.03), ("Totals", c_tot, 0.03),
                             ("Combined", pd.concat([c_ml, c_tot]), 0.03)):
            b, bank, mdd = simulate(c, thr)
            label = f"{name} @ {'best opening price across books' if price_tag == 'open_best' else 'median opening price'} (EV > {thr:.0%})"
            lines.append(report(label, b, bank, mdd))
            if name == "Combined" and price_tag == "open_best":
                all_bets.append(b)
    print("\n\n".join(lines))
    pd.concat(all_bets).to_csv(a.out, index=False)


if __name__ == "__main__":
    main()
