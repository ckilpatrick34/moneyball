"""Tune and evaluate the college rating model on 2021-2026 D1 results (no odds available)."""
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from moneyball.backtest import run_backtest  # noqa: E402
from moneyball.ratings import RatingConfig  # noqa: E402

g = pd.read_pickle(sys.argv[1] if len(sys.argv) > 1 else "/home/user/data/proc/ncaa_games.pkl")
g = g[g.home_score != g.away_score]
res = []
for lr, hfa, co, disp in itertools.product((0.05, 0.07, 0.1), (0.10, 0.13), (0.7, 0.85), (3.0,)):
    cfg = RatingConfig.college()
    cfg.lr_off = cfg.lr_def = lr
    cfg.hfa, cfg.season_carryover, cfg.dispersion = hfa, co, disp
    r = run_backtest(g, cfg)
    p = r.predictions
    p = p[p.season >= 2022]
    pr = p.p_home.clip(1e-6, 1 - 1e-6)
    ll = float(-(p.home_won * np.log(pr) + (1 - p.home_won) * np.log(1 - pr)).mean())
    res.append(dict(lr=lr, hfa=hfa, carry=co, ll=ll, acc=float(((pr > .5) == p.home_won).mean())))
df = pd.DataFrame(res).sort_values("ll")
print(df.head(8).to_string(index=False))
print("baseline (always home-win rate) ll:",
      float(-np.mean(np.where(p.home_won, np.log(p.home_won.mean()), np.log(1 - p.home_won.mean())))))
