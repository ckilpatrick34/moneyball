"""Roll the live state through the 2026 season from the MLB Stats API, scoring every
game before its result is added: an out-of-sample check of the live pipeline itself.

    python research/live_2026_check.py [state_in] [state_out] [box_cache]
"""
import pickle
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from moneyball.live import statsapi  # noqa: E402
from moneyball.live.parks import PARKS  # noqa: E402
from moneyball.model import derive, load_models  # noqa: E402

st_in = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "state" / "mlb_state.pkl"
st_out = Path(sys.argv[2]) if len(sys.argv) > 2 else st_in
cache = Path(sys.argv[3]) if len(sys.argv) > 3 else ROOT / "state" / "boxscores"
st = pickle.loads(st_in.read_bytes())
ml, tot, _ = load_models(ROOT / "models" / "mlb_models.json")

games = []
cur = (st.last_update + timedelta(days=1)).date()
end = date(2026, 11, 5)
while cur <= end:
    nxt = min(end, cur + timedelta(days=30))
    games += [g for g in statsapi.schedule(cur.isoformat(), nxt.isoformat(), "R,F,D,L,W")
              if g["state"] == "Final" and g["home_score"] is not None]
    cur = nxt + timedelta(days=1)
print("final games to add:", len(games))
rows = []
for i, (day, dg) in enumerate(pd.DataFrame(games).groupby("date", sort=True)):
    st.begin_day(pd.Timestamp(day))
    recs = []
    for g in dg.to_dict("records"):
        box = statsapi.boxscore(g["game_pk"], cache)
        rec = statsapi.record_from_boxscore(box, g, st.hands)
        recs.append(rec)
        if g["home"] in PARKS and rec["home_sp"] and rec["away_sp"]:
            f = st.game_features(g["home"], g["away"], rec["home_sp"], rec["away_sp"], rec["site"], None)
            rows.append({**f, "game_pk": g["game_pk"], "home_score": g["home_score"], "away_score": g["away_score"]})
    for rec in recs:
        st.update(rec, pd.Timestamp(day))
    if i % 30 == 0:
        print(day, len(rows))
st.last_update = pd.Timestamp(max(g["date"] for g in games))
st_out.write_bytes(pickle.dumps(st))
d = derive(pd.DataFrame(rows))
d["p"] = ml.predict(d)
d["y"] = (d.home_score > d.away_score).astype(int)
ll = float(-(d.y * np.log(d.p) + (1 - d.y) * np.log(1 - d.p)).mean())
base = float(-np.mean(np.where(d.y, np.log(d.y.mean()), np.log(1 - d.y.mean()))))
print(f"2026 games scored: {len(d)}  model log loss {ll:.4f}  (home-rate baseline {base:.4f})  "
      f"accuracy {((d.p > .5) == d.y).mean():.3f}  avg p_home {d.p.mean():.3f} vs actual {d.y.mean():.3f}")
d.drop(columns=["date"]).assign(date=d.date.astype(str)).to_csv(ROOT / "data" / "live_2026_scored.csv", index=False)
