"""Command line entry point.

    python -m moneyball.cli mlb-history --seasons 2022 2023 2024 2025
    python -m moneyball.cli ncaa-history --seasons 2024 2025
    python -m moneyball.cli backtest --games data/mlb_games_with_odds.csv --league mlb
    python -m moneyball.cli snapshot-odds --league mlb
    python -m moneyball.cli today --league mlb --bankroll 1000
"""

from __future__ import annotations

import argparse
import math
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from .backtest import BetConfig, blend, run_backtest
from .odds import expected_value, fair_probs, prob_to_american
from .ratings import RatingConfig, RatingModel
from .sizing import SizingConfig, stake

DATA = Path(__file__).resolve().parents[1] / "data"
LOCAL_TZ = "America/Chicago"


def _cfg(league: str) -> RatingConfig:
    return RatingConfig.college() if league == "ncaa" else RatingConfig.mlb()


def _val(v):
    return None if v is None or (isinstance(v, float) and math.isnan(v)) else v


def replay(games: pd.DataFrame, cfg: RatingConfig) -> RatingModel:
    """Build ratings from every completed game, in order."""
    m = RatingModel(cfg)
    g = games.dropna(subset=["home_score", "away_score"])
    g = g.assign(_d=pd.to_datetime(g["date"])).sort_values("_d", kind="stable")
    for _, r in g.iterrows():
        m.start_season(int(r["season"]))
        m.update(r["home"], r["away"], float(r["home_score"]), float(r["away_score"]),
                 _val(r.get("home_sp")), _val(r.get("away_sp")), bool(_val(r.get("neutral")) or False),
                 _val(r.get("home_sp_ip")), _val(r.get("away_sp_ip")))
    return m


def cmd_mlb_history(a):
    from .data import mlb
    DATA.mkdir(exist_ok=True)
    frames = [mlb.season_games(s, with_starters=not a.no_starters) for s in a.seasons]
    out = pd.concat(frames, ignore_index=True)
    path = Path(a.out or DATA / "mlb_games.csv")
    out.to_csv(path, index=False)
    print(f"Wrote {len(out)} games to {path}")


def cmd_ncaa_history(a):
    from .data import college
    DATA.mkdir(exist_ok=True)
    out = pd.concat([college.season_games(s) for s in a.seasons], ignore_index=True)
    path = Path(a.out or DATA / "ncaa_games.csv")
    out.to_csv(path, index=False)
    print(f"Wrote {len(out)} games to {path}")


def cmd_backtest(a):
    games = pd.read_csv(a.games)
    bet = BetConfig(min_ev=a.min_ev, market_weight=a.market_weight, bankroll=a.bankroll,
                    markets=tuple(a.markets.split(",")))
    res = run_backtest(games, _cfg(a.league), bet)
    print(res.report())
    if a.bets_out:
        res.bets.to_csv(a.bets_out, index=False)
        print(f"\nWrote bets to {a.bets_out}")


def cmd_snapshot_odds(a):
    from .data import odds_api
    df = odds_api.fetch_odds(a.league)
    odds_api.save_snapshot(df, a.league)
    print(f"Saved {len(df)} odds rows for {df['event_id'].nunique() if len(df) else 0} games")


def _today_schedule(league: str, day: date) -> pd.DataFrame:
    ds = day.strftime("%Y-%m-%d")
    if league == "mlb":
        from .data import mlb
        return pd.DataFrame([g for g in mlb.schedule(ds, ds) if not g["final"]])
    from .data import college
    return pd.DataFrame([g for g in college.day_games(ds) if not g["final"]])


def _season_to_date(league: str, day: date) -> pd.DataFrame:
    """Completed games this season through yesterday (probable starters as starters)."""
    yday = (day - timedelta(days=1)).strftime("%Y-%m-%d")
    if league == "mlb":
        from .data import mlb
        rows = [g for g in mlb.schedule(f"{day.year}-03-01", yday) if g["final"]]
    else:
        from .data import college
        return college.season_games(day.year, end=yday, verbose=False)
    df = pd.DataFrame(rows)
    if len(df):
        df["season"] = day.year
    return df


def cmd_today(a):
    from .data import odds_api
    from .data.names import match_name

    day = (datetime.strptime(a.date, "%Y-%m-%d").date() if a.date
           else datetime.now(ZoneInfo(LOCAL_TZ)).date())
    hist_path = Path(a.history or DATA / f"{a.league}_games.csv")
    hist = pd.read_csv(hist_path) if hist_path.exists() else pd.DataFrame()
    cur = _season_to_date(a.league, day)
    games = pd.concat([hist, cur], ignore_index=True)
    if "game_id" in games:
        games = games.drop_duplicates("game_id", keep="first")
    games = games[pd.to_datetime(games["date"]).dt.date < day]
    model = replay(games, _cfg(a.league))
    model.start_season(day.year)
    print(f"Ratings built from {len(games)} games")

    slate = _today_schedule(a.league, day)
    if slate.empty:
        print("No upcoming games.")
        return
    odds = odds_api.fetch_odds(a.league)
    odds_api.save_snapshot(odds, a.league)
    best = odds_api.best_prices(odds)
    fair = odds_api.consensus_fair(odds)
    events = odds.drop_duplicates("event_id")[["event_id", "home", "away", "commence_utc"]]

    sizing = SizingConfig(kelly_fraction=a.kelly, max_bet_pct=a.max_bet)
    recs, board = [], []
    for _, g in slate.iterrows():
        h = match_name(g["home"], list(events["home"]))
        aw = match_name(g["away"], list(events["away"]))
        ev = events[(events["home"] == h) & (events["away"] == aw)]
        gp = model.predict(g["home"], g["away"], _val(g.get("home_sp")), _val(g.get("away_sp")),
                           bool(g.get("neutral", False)))
        line = {"game": f"{g['away']} @ {g['home']}", "p_home": gp.home_win,
                "fair_home": prob_to_american(gp.home_win), "proj_total": gp.total_mean}
        board.append(line)
        if ev.empty:
            continue
        eid = ev["event_id"].iloc[0]
        b, f = best[best["event_id"] == eid], fair[fair["event_id"] == eid]

        def price(df, market, side, point=0.0):
            x = df[(df["market"] == market) & (df["side"] == side) & (df["point"] == point)]
            return None if x.empty else float(x["price"].iloc[0])

        cands = []
        fh, fa = price(f, "h2h", "home"), price(f, "h2h", "away")
        if fh and fa:
            mh, ma = fair_probs([fh, fa])
            cands += [("ML", g["home"], gp.home_win, mh, price(b, "h2h", "home")),
                      ("ML", g["away"], gp.away_win, ma, price(b, "h2h", "away"))]
        tot = f[(f["market"] == "totals")]
        if not tot.empty:
            pt = tot["point"].mode().iloc[0]
            fo, fu = price(f, "totals", "over", pt), price(f, "totals", "under", pt)
            if fo and fu:
                mo, mu = fair_probs([fo, fu])
                po, pu, pp = gp.total_probs(pt)
                nonpush = max(1e-9, 1 - pp)
                cands += [(f"O {pt}", "Over", po / nonpush, mo, price(b, "totals", "over", pt)),
                          (f"U {pt}", "Under", pu / nonpush, mu, price(b, "totals", "under", pt))]
        for market, side, pm, pmk, best_px in cands:
            if best_px is None or abs(pm - pmk) > a.max_disagreement:
                continue
            p = blend(pm, pmk, a.market_weight)
            evv = expected_value(p, best_px)
            if evv < a.min_ev:
                continue
            amt = stake(p, best_px, a.bankroll, sizing)
            if amt > 0:
                recs.append({"game": line["game"], "bet": f"{market} {side}", "price": int(best_px),
                             "model_p": round(pm, 3), "market_p": round(pmk, 3),
                             "bet_p": round(p, 3), "ev": round(evv, 3), "stake": amt})

    pd.set_option("display.width", 200)
    print(f"\n{a.league.upper()} board for {day}:")
    bd = pd.DataFrame(board)
    bd["fair_home"] = bd["fair_home"].round().astype(int)
    print(bd.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("\nRecommended bets:" if recs else "\nNo bets clear the thresholds today.")
    if recs:
        rdf = pd.DataFrame(recs).sort_values("ev", ascending=False)
        print(rdf.to_string(index=False))
        out = DATA / f"bets_{a.league}_{day}.csv"
        rdf.to_csv(out, index=False)
        print(f"\nSaved to {out}")


def main(argv=None):
    p = argparse.ArgumentParser(prog="moneyball")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("mlb-history", help="download MLB results + starters")
    s.add_argument("--seasons", type=int, nargs="+", required=True)
    s.add_argument("--no-starters", action="store_true", help="skip boxscores (faster, no SP ratings)")
    s.add_argument("--out")
    s.set_defaults(func=cmd_mlb_history)

    s = sub.add_parser("ncaa-history", help="download college results from ESPN")
    s.add_argument("--seasons", type=int, nargs="+", required=True)
    s.add_argument("--out")
    s.set_defaults(func=cmd_ncaa_history)

    s = sub.add_parser("backtest", help="walk-forward backtest against recorded odds")
    s.add_argument("--games", required=True)
    s.add_argument("--league", choices=["mlb", "ncaa"], default="mlb")
    s.add_argument("--min-ev", type=float, default=0.025)
    s.add_argument("--market-weight", type=float, default=0.5)
    s.add_argument("--bankroll", type=float, default=1000)
    s.add_argument("--markets", default="ml,total")
    s.add_argument("--bets-out")
    s.set_defaults(func=cmd_backtest)

    s = sub.add_parser("snapshot-odds", help="log current odds (run on a schedule for CLV)")
    s.add_argument("--league", choices=["mlb", "ncaa"], default="mlb")
    s.set_defaults(func=cmd_snapshot_odds)

    s = sub.add_parser("today", help="price today's slate and recommend bets")
    s.add_argument("--league", choices=["mlb", "ncaa"], default="mlb")
    s.add_argument("--date")
    s.add_argument("--history")
    s.add_argument("--bankroll", type=float, default=1000)
    s.add_argument("--kelly", type=float, default=0.25)
    s.add_argument("--max-bet", type=float, default=0.02)
    s.add_argument("--min-ev", type=float, default=0.025)
    s.add_argument("--market-weight", type=float, default=0.5)
    s.add_argument("--max-disagreement", type=float, default=0.12)
    s.set_defaults(func=cmd_today)

    a = p.parse_args(argv)
    a.func(a)


if __name__ == "__main__":
    main()
