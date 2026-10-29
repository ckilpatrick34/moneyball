"""Robustness: forecast-quality weather instead of actual, and platoon features.

No archived weather forecasts are reachable from the sandbox, so forecasts are
simulated from actual game weather with published day-ahead error levels:
temperature RMSE ~3.5 F, wind speed RMSE ~4 mph, and wind direction relative
to the park (blowing in vs out) wrong or unknown 35% of the time. Models are
trained on actual weather and scored on the noisy "forecast", which is the
conservative way round.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mlb_strategy as ms  # noqa: E402
from moneyball.odds import american_to_decimal as dec, fair_probs, logit, sigmoid  # noqa: E402


def forecast(d, seed, temp_sd=3.5, wind_sd=4.0, dir_err=0.35):
    rng = np.random.default_rng(seed)
    d = d.copy()
    d["temp_c"] = (d.temp_actual + rng.normal(0, temp_sd, len(d))).clip(40, 100)
    speed = (d.wind_out_actual + d.wind_in_actual + rng.normal(0, wind_sd, len(d))).clip(lower=0)
    direction = np.sign(d.wind_out_actual - d.wind_in_actual)       # +1 out, -1 in, 0 cross/none
    flip = rng.random(len(d)) < dir_err
    direction = np.where(flip, rng.choice([-1, 0, 1], len(d)), direction)
    speed = np.where(d.dome == 1, 0, speed)
    d["wind_out"] = np.where(direction > 0, speed, 0.0)
    d["wind_in"] = np.where(direction < 0, speed, 0.0)
    return d


def tot_walk(d, feats, test_df=None):
    """Like ms.candidates_tot but trains on `d` and scores `test_df` (same rows, other weather)."""
    def prep(x):
        x = x.dropna(subset=["total_open", "over_open", "under_open", "over_open_best", "under_open_best"]).copy()
        x["tot"] = x.home_score + x.away_score
        x["po"] = [fair_probs([o, u])[0] for o, u in zip(x.over_open, x.under_open)]
        x["lo_po"] = x.po.map(logit)
        x["ln_line"] = np.log(x.total_open)
        x["yo"] = (x.tot > x.total_open).astype(int)
        return x
    tr_all, te_all = prep(d), prep(test_df if test_df is not None else d)
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    rows = []
    for s in ms.TEST_SEASONS:
        tr = tr_all[(tr_all.season < s) & (tr_all.tot != tr_all.total_open)]
        te = te_all[te_all.season == s].copy()
        sc = StandardScaler().fit(tr[feats])
        m = LogisticRegression(C=0.05).fit(np.column_stack([tr.lo_po, sc.transform(tr[feats])]), tr.yo)
        te["p"] = m.predict_proba(np.column_stack([te.lo_po, sc.transform(te[feats])]))[:, 1]
        rows.append(te)
    t = pd.concat(rows)
    out = []
    for r in t.itertuples():
        p = sigmoid(0.5 * logit(r.p) + 0.5 * r.lo_po)
        push = r.tot == r.total_open
        for side, pp, price, won, mv in (("over", p, r.over_open_best, r.tot > r.total_open, r.total_close - r.total_open),
                                         ("under", 1 - p, r.under_open_best, r.tot < r.total_open, r.total_open - r.total_close)):
            out.append(dict(date=r.date, season=r.season, market="TOT", side=side, price=price, p=pp,
                            won=int(won), push=bool(push), clv=np.nan, line_move=mv))
    return pd.DataFrame(out)


def line(name, c, thr=0.03):
    b, bank, mdd = ms.simulate(c, thr, kelly_frac=0.125, cap=0.01)
    if b.empty:
        return f"{name}: no bets"
    f = b["flat"]
    t = f.mean() / (f.std() / np.sqrt(len(f)))
    seas = b.groupby("season")["flat"].mean()
    mv = (b.line_move > 0).mean() if "line_move" in b and b.line_move.notna().any() else np.nan
    mva = (b.line_move < 0).mean() if "line_move" in b and b.line_move.notna().any() else np.nan
    clv = b.clv.mean()
    return (f"{name:<48} bets {len(b):5d}  ROI {f.mean():+.2%} (t={t:+.2f})  "
            + (f"line moved for/against {mv:.0%}/{mva:.0%}  " if mv == mv else f"CLV {clv:+.2%}  ")
            + " ".join(f"{k}:{v:+.1%}" for k, v in seas.items()))


def main():
    d = ms.load(sys.argv[1], sys.argv[2], Path(sys.argv[3]))
    base = [x for x in ms.TOT_FEATS]
    no_wind = [x for x in base if x not in ("wind_out", "wind_in")]
    no_wx = [x for x in no_wind if x != "temp_c"]
    print("TOTALS (bet at best opening price, EV > 3%, 1/8 Kelly)")
    print(line("actual weather (old backtest)", tot_walk(d, base)))
    for seed in (1, 2, 3):
        print(line(f"forecast weather, seed {seed}", tot_walk(d, base, forecast(d, seed))))
    print(line("forecast temp only (no wind), seed 1", tot_walk(d, no_wind, forecast(d, 1))))
    print(line("no weather", tot_walk(d, no_wx)))
    print(line("forecast weather + platoon, seed 1", tot_walk(d, base + ms.PLATOON_TOT, forecast(d, 1))))

    print("\nMONEYLINE (bet at best opening price, EV > 3%, 1/8 Kelly)")
    for name, feats in (("baseline", ms.ML_FEATS), ("+ platoon (team wOBA vs SP hand)", ms.ML_FEATS + ms.PLATOON_ML)):
        ml = ms.walk_forward(d, feats, "y", None)
        edge = ml.p.map(logit) - ml.lo_open
        corr = np.corrcoef(edge, ml.lo_close - ml.lo_open)[0, 1]
        ll = float(-(ml.y * np.log(ml.p) + (1 - ml.y) * np.log(1 - ml.p)).mean())
        print(line(name, ms.candidates_ml(ml, 0.5, "open_best")) + f"  | logloss {ll:.5f} edge~move corr {corr:.3f}")


if __name__ == "__main__":
    main()
