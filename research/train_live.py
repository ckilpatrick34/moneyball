"""Train the live models on all seasons with odds and build the starting feature state.

    python research/train_live.py --retro <retrosheet>/seasons --odds mlb_odds_dataset.json \
        --register <chadwick register>/data

Writes models/mlb_models.json and state/mlb_state.pkl (player ids converted to MLBAM).
"""
import argparse
import glob
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research"))
import mlb_strategy as ms  # noqa: E402
from moneyball.data import retrosheet  # noqa: E402
from moneyball.features import build  # noqa: E402
from moneyball.model import ML_FEATS, TOT_FEATS, LogitModel, save_models  # noqa: E402
from moneyball.odds import fair_probs, logit  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--retro", required=True)
    ap.add_argument("--odds", required=True)
    ap.add_argument("--register", required=True, help="folder with Chadwick register people-*.csv")
    ap.add_argument("--cache", default=str(ROOT / "data" / "mlb_model_df_v2.pkl"))
    a = ap.parse_args()

    d = ms.load(a.retro, a.odds, Path(a.cache))
    ml = LogitModel(ML_FEATS).fit(d, d.y)

    t = d.dropna(subset=["total_open", "over_open", "under_open"]).copy()
    t["tot"] = t.home_score + t.away_score
    t = t[t.tot != t.total_open]
    t["lo_po"] = [logit(fair_probs([o, u])[0]) for o, u in zip(t.over_open, t.under_open)]
    t["ln_line"] = np.log(t.total_open)
    tot = LogitModel(TOT_FEATS, market_col="lo_po").fit(t, (t.tot > t.total_open).astype(int))

    meta = {"trained_on": f"{int(d.season.min())}-{int(d.season.max())}", "ml_games": int(len(d)),
            "total_games": int(len(t))}
    (ROOT / "models").mkdir(exist_ok=True)
    save_models(ROOT / "models" / "mlb_models.json", ml, tot, meta)
    print("models:", meta)
    print("ML coefs:", dict(zip(ML_FEATS, np.round(ml.coef, 3))))
    print("Totals coefs:", dict(zip(["lo_po"] + TOT_FEATS, np.round(tot.coef, 3))))

    # starting state: every Retrosheet game 2019-2025, then ids -> MLBAM
    g, pa, lu = retrosheet.parse_seasons(Path(a.retro), range(2019, 2026))
    hands = retrosheet.parse_hands(Path(a.retro), range(2019, 2026))
    g = g[g.gametype.fillna("regular") != "allstar"].rename(columns={"hometeam": "home", "visteam": "away"})
    _, st = build(g, pa, lu, hands=hands, return_state=True)
    reg = pd.concat(pd.read_csv(f, usecols=["key_retro", "key_mlbam"], dtype=str)
                    for f in glob.glob(str(Path(a.register) / "people-*.csv"))).dropna()
    mapping = {r: int(m) for r, m in zip(reg.key_retro, reg.key_mlbam)}
    used = set(st.pit.v) | set(st.bat.v)
    print(f"id map covers {len(used & set(mapping))}/{len(used)} players seen 2019-2025")
    st.rename_players(mapping)
    (ROOT / "state").mkdir(exist_ok=True)
    (ROOT / "state" / "mlb_state.pkl").write_bytes(pickle.dumps(st))
    print("state through", st.last_update.date())


if __name__ == "__main__":
    main()
