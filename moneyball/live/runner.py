"""Daily MLB runner: catch up on results, price today's slate, log picks, settle and track CLV.

    python -m moneyball.live.runner picks   [--date YYYY-MM-DD] [--dry-run]
    python -m moneyball.live.runner snapshot          # odds snapshot only (for closing lines)
    python -m moneyball.live.runner settle            # grade bets, compute CLV, print record
    python -m moneyball.live.runner catchup [--date]  # roll the feature state forward (first run: whole 2026)
    python -m moneyball.live.runner oddscheck [--date] # fetch and print odds only, writes nothing

Files (under --home, default the repo root):
    config.json                 bankroll, books, thresholds
    models/mlb_models.json      trained models (research/train_live.py)
    state/mlb_state.pkl         FeatureState, rolled forward each day
    logs/bets.csv               every pick, settled in place
    logs/odds_snapshots.csv.gz  every odds pull (closing lines come from here)
    picks/<date>.md             human-readable card
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests

from ..model import blend, derive, load_models
from ..odds import american_to_decimal, fair_probs, logit
from . import action_network, statsapi, weather
from .parks import PARKS, std_abbr

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = {
    "bankroll_start": 1000.0,
    "kelly_fraction": 0.125,
    "max_bet_pct": 0.01,
    "min_ev": 0.03,
    "market_weight": 0.5,
    "books": ["draftkings", "fanduel", "betmgm", "williamhill_us", "betrivers", "espnbet"],
    "markets": ["ml", "total"],
    "timezone": "America/Chicago",
}
BET_COLS = ["date", "game_pk", "event_id", "commence_utc", "game", "market", "side", "line", "book",
            "price", "p_model", "p_market", "p_bet", "ev", "stake", "status", "result", "pnl",
            "close_price_fair", "clv"]


class Paths:
    def __init__(self, home: Path):
        self.home = Path(home)
        self.config = self.home / "config.json"
        self.models = self.home / "models" / "mlb_models.json"
        self.state = self.home / "state" / "mlb_state.pkl"
        self.bets = self.home / "logs" / "bets.csv"
        self.snaps = self.home / "logs" / "odds_snapshots.csv.gz"
        self.picks = self.home / "picks"
        self.box_cache = self.home / "state" / "boxscores"


def load_config(p: Paths) -> dict:
    cfg = dict(DEFAULT_CONFIG)
    if p.config.exists():
        cfg.update(json.loads(p.config.read_text()))
    return cfg


# ---------------------------------------------------------------- state upkeep
def catch_up(st, through: date, p: Paths, verbose=True):
    """Feed every completed game after the state's last update through `through` (inclusive)."""
    start = (st.last_update + timedelta(days=1)).date() if st.last_update is not None else date(through.year, 3, 1)
    if start > through:
        return 0
    n, cur = 0, start
    while cur <= through:
        end = min(through, cur + timedelta(days=30))
        games = [g for g in statsapi.schedule(cur.isoformat(), end.isoformat(), game_types="R,F,D,L,W")
                 if g["state"] == "Final" and g["home_score"] is not None]
        for day, dg in pd.DataFrame(games).groupby("date", sort=True) if games else []:
            st.begin_day(pd.Timestamp(day))
            for g in dg.to_dict("records"):
                box = statsapi.boxscore(g["game_pk"], p.box_cache)
                st.update(statsapi.record_from_boxscore(box, g, st.hands), pd.Timestamp(day))
                n += 1
        cur = end + timedelta(days=1)
    st.last_update = pd.Timestamp(through)
    if verbose:
        print(f"State caught up through {through}: {n} games added")
    return n


# ---------------------------------------------------------------- odds
def fetch_odds(books: list[str], day: date | None = None) -> pd.DataFrame:
    """The Odds API when ODDS_API_KEY is set, otherwise Action Network's keyless feed for `day`."""
    key = os.environ.get("ODDS_API_KEY")
    if not key:
        day = day or datetime.now(ZoneInfo(DEFAULT_CONFIG["timezone"])).date()
        odds = action_network.fetch(day, books)
        print(f"Odds from Action Network (no ODDS_API_KEY): {odds.event_id.nunique()} games, "
              f"books {sorted(odds.book.unique())}")
        return odds
    r = requests.get("https://api.the-odds-api.com/v4/sports/baseball_mlb/odds", timeout=30, params={
        "apiKey": key, "markets": "h2h,totals", "oddsFormat": "american", "bookmakers": ",".join(books)})
    r.raise_for_status()
    print(f"Odds API credits remaining: {r.headers.get('x-requests-remaining')}")
    return flatten_odds(r.json())


def flatten_odds(events: list[dict]) -> pd.DataFrame:
    now = datetime.now(timezone.utc).isoformat()
    rows = []
    for ev in events:
        home, away = std_abbr(ev["home_team"]), std_abbr(ev["away_team"])
        for bk in ev.get("bookmakers", []):
            for mk in bk.get("markets", []):
                for oc in mk.get("outcomes", []):
                    nm = oc["name"]
                    side = ("home" if std_abbr(nm) == home else "away" if std_abbr(nm) == away else nm.lower())
                    rows.append({"snapshot_utc": now, "event_id": ev["id"], "commence_utc": ev["commence_time"],
                                 "home": home, "away": away, "book": bk["key"], "market": mk["key"],
                                 "side": side, "price": float(oc["price"]), "point": oc.get("point")})
    return pd.DataFrame(rows)


def save_snapshot(odds: pd.DataFrame, p: Paths):
    p.snaps.parent.mkdir(parents=True, exist_ok=True)
    odds.to_csv(p.snaps, mode="a", header=not p.snaps.exists(), index=False, compression="gzip")


def market_view(odds_ev: pd.DataFrame) -> dict:
    """Consensus no-vig probabilities and best prices for one event."""
    out = {}
    h2h = odds_ev[odds_ev.market == "h2h"].pivot_table(index="book", columns="side", values="price", aggfunc="last")
    if {"home", "away"} <= set(h2h.columns):
        h2h = h2h.dropna(subset=["home", "away"])
        if len(h2h):
            fair = [fair_probs([h, a])[0] for h, a in zip(h2h.home, h2h.away)]
            out["p_home"] = float(np.median(fair))
            for s in ("home", "away"):
                book = max(h2h.index, key=lambda b: american_to_decimal(h2h.loc[b, s]))
                out[f"best_{s}"] = (float(h2h.loc[book, s]), book)
    tot = odds_ev[odds_ev.market == "totals"].dropna(subset=["point"])
    if len(tot):
        line = float(tot.point.mode().iloc[0])
        t = tot[tot.point == line].pivot_table(index="book", columns="side", values="price", aggfunc="last")
        if {"over", "under"} <= set(t.columns):
            t = t.dropna(subset=["over", "under"])
            if len(t):
                out["total_line"] = line
                out["p_over"] = float(np.median([fair_probs([o, u])[0] for o, u in zip(t.over, t.under)]))
                for s in ("over", "under"):
                    book = max(t.index, key=lambda b: american_to_decimal(t.loc[b, s]))
                    out[f"best_{s}"] = (float(t.loc[book, s]), book)
    return out


def match_event(g: dict, events: pd.DataFrame):
    ev = events[(events.home == g["home"]) & (events.away == g["away"])]
    if ev.empty:
        return None
    if len(ev) > 1 and g.get("start_utc"):   # double-header: nearest start time
        t = pd.Timestamp(g["start_utc"])
        ev = ev.assign(dt=(pd.to_datetime(ev.commence_utc) - t).abs()).sort_values("dt")
    return ev.iloc[0]


# ---------------------------------------------------------------- picks
def stake_for(p, price, bankroll, cfg):
    b = american_to_decimal(price) - 1
    f = max(0.0, (b * p - (1 - p)) / b) * cfg["kelly_fraction"]
    return round(bankroll * min(f, cfg["max_bet_pct"]), 2)


def read_bets(p: Paths) -> pd.DataFrame:
    b = pd.read_csv(p.bets, dtype={"result": str, "status": str, "event_id": str, "side": str, "book": str,
                                   "date": str})
    b["result"] = b["result"].fillna("")
    return b


def current_bankroll(p: Paths, cfg) -> float:
    if not p.bets.exists():
        return cfg["bankroll_start"]
    b = read_bets(p)
    return cfg["bankroll_start"] + float(b.loc[b.status == "settled", "pnl"].sum())


def make_picks(day: date, p: Paths, cfg: dict, odds: pd.DataFrame | None = None, dry_run=False,
               schedule_rows: list | None = None, weather_fn=None) -> pd.DataFrame:
    ml_model, tot_model, meta = load_models(p.models)
    st = pickle.loads(p.state.read_bytes())
    if schedule_rows is None:
        catch_up(st, day - timedelta(days=1), p)
        if not dry_run:
            p.state.parent.mkdir(parents=True, exist_ok=True)
            p.state.write_bytes(pickle.dumps(st))
        schedule_rows = statsapi.schedule(day.isoformat(), day.isoformat())
    games = [g for g in schedule_rows if g["state"] == "Preview" and g["home_sp"] and g["away_sp"]
             and g["home"] in PARKS]
    skipped = len(schedule_rows) - len(games)
    missing = {g[k] for g in games for k in ("home_sp", "away_sp") if g[k] not in st.hands}
    if missing and weather_fn is None:
        st.hands.update(statsapi.people_hands(missing))
    if odds is None:
        odds = fetch_odds(cfg["books"], day)
        if not dry_run:
            save_snapshot(odds, p)
    events = odds.drop_duplicates("event_id")[["event_id", "home", "away", "commence_utc"]]
    weather_fn = weather_fn or weather.game_weather

    st.begin_day(pd.Timestamp(day))
    rows = []
    for g in games:
        wx = weather_fn(g["home"], g["start_utc"])
        f = st.game_features(g["home"], g["away"], g["home_sp"], g["away_sp"], PARKS[g["home"]][0], wx)
        ev = match_event(g, events)
        mv = market_view(odds[odds.event_id == ev.event_id]) if ev is not None else {}
        rows.append({**f, **{k: v for k, v in g.items() if k not in f}, "event_id": None if ev is None else ev.event_id,
                     "commence_utc": None if ev is None else ev.commence_utc, "mv": mv,
                     "forecast_temp": wx.get("forecast_temp")})
    if not rows:
        print(f"No playable games on {day} ({skipped} skipped).")
        return pd.DataFrame(columns=BET_COLS)
    df = derive(pd.DataFrame(rows))
    df["p_ml"] = ml_model.predict(df)

    bankroll = current_bankroll(p, cfg)
    picks, board = [], []
    for r in df.to_dict("records"):
        mv = r["mv"]
        game = f"{r['away']} @ {r['home']}"
        line = {"game": game, "sp": f"{r.get('away_sp_name')} vs {r.get('home_sp_name')}",
                "model_home": r["p_ml"], "market_home": mv.get("p_home"), "temp": r.get("forecast_temp")}
        cands = []
        if "ml" in cfg["markets"] and "p_home" in mv:
            for side, pm, pmk in (("home", r["p_ml"], mv["p_home"]), ("away", 1 - r["p_ml"], 1 - mv["p_home"])):
                price, book = mv[f"best_{side}"]
                cands.append(("ML", r[side], None, pm, pmk, price, book))
        if "total" in cfg["markets"] and "p_over" in mv:
            one = pd.DataFrame([{**r, "ln_line": np.log(mv["total_line"]), "lo_po": logit(mv["p_over"])}])
            po = float(tot_model.predict(one)[0])
            line.update({"total": mv["total_line"], "model_over": po, "market_over": mv["p_over"]})
            for side, pm, pmk in (("over", po, mv["p_over"]), ("under", 1 - po, 1 - mv["p_over"])):
                price, book = mv[f"best_{side}"]
                cands.append(("TOTAL", side.capitalize(), mv["total_line"], pm, pmk, price, book))
        board.append(line)
        for market, side, ln, pm, pmk, price, book in cands:
            pb = float(blend(pm, pmk, cfg["market_weight"]))
            ev = pb * american_to_decimal(price) - 1
            if ev < cfg["min_ev"]:
                continue
            amt = stake_for(pb, price, bankroll, cfg)
            if amt <= 0:
                continue
            picks.append({"date": day.isoformat(), "game_pk": r["game_pk"], "event_id": r["event_id"],
                          "commence_utc": r["commence_utc"], "game": game, "market": market, "side": side,
                          "line": ln, "book": book, "price": price, "p_model": round(pm, 4),
                          "p_market": round(pmk, 4), "p_bet": round(pb, 4), "ev": round(ev, 4), "stake": amt,
                          "status": "open", "result": "", "pnl": 0.0, "close_price_fair": np.nan, "clv": np.nan})
    picks_df = pd.DataFrame(picks, columns=BET_COLS)
    card = render_card(day, picks_df, pd.DataFrame(board), bankroll, skipped)
    print(card)
    if not dry_run:
        p.picks.mkdir(parents=True, exist_ok=True)
        (p.picks / f"{day.isoformat()}.md").write_text(card)
        if len(picks_df):
            p.bets.parent.mkdir(parents=True, exist_ok=True)
            picks_df.to_csv(p.bets, mode="a", header=not p.bets.exists(), index=False)
        notify(card)
    return picks_df


def render_card(day, picks, board, bankroll, skipped) -> str:
    lines = [f"# MLB picks for {day}", "", f"Bankroll: ${bankroll:,.2f}", ""]
    if picks.empty:
        lines.append("No bets clear the 3% EV bar today.")
    else:
        lines += ["| Game | Bet | Price | Book | EV | Stake |", "|---|---|---|---|---|---|"]
        for r in picks.sort_values("ev", ascending=False).itertuples():
            bet = f"{r.side} ML" if r.market == "ML" else f"{r.side} {r.line:g}"
            lines.append(f"| {r.game} | {bet} | {int(r.price):+d} | {r.book} | {r.ev:.1%} | ${r.stake:,.2f} |")
        lines += ["", f"{len(picks)} bets, ${picks.stake.sum():,.2f} total."]
    if not board.empty:
        lines += ["", "<details><summary>Full board</summary>", "",
                  "| Game | Starters | Model home | Market home | Total | Model over | Market over | Temp |",
                  "|---|---|---|---|---|---|---|---|"]
        for r in board.itertuples():
            fmt = lambda x: "" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.3f}"  # noqa: E731
            tot = getattr(r, "total", None)
            lines.append(f"| {r.game} | {r.sp} | {fmt(r.model_home)} | {fmt(r.market_home)} | "
                         f"{'' if tot is None or tot != tot else f'{tot:g}'} | {fmt(getattr(r, 'model_over', None))} | "
                         f"{fmt(getattr(r, 'market_over', None))} | {'' if r.temp is None or r.temp != r.temp else round(r.temp)} |")
        lines += ["", "</details>"]
    if skipped:
        lines += ["", f"{skipped} games skipped (started, postponed, or no probable starter yet)."]
    return "\n".join(lines) + "\n"


def notify(text: str):
    url = os.environ.get("DISCORD_WEBHOOK_URL")
    if url:
        try:
            requests.post(url, json={"content": text[:1900]}, timeout=15)
        except Exception as e:  # noqa: BLE001
            print(f"notify failed: {e}")
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a") as f:
            f.write(text)


# ---------------------------------------------------------------- settle + CLV
def settle(p: Paths, results: dict | None = None) -> pd.DataFrame:
    if not p.bets.exists():
        print("No bets yet.")
        return pd.DataFrame()
    bets = read_bets(p)
    open_ = bets[bets.status == "open"]
    if open_.empty:
        return report(bets)
    if results is None:
        lo, hi = open_.date.min(), open_.date.max()
        results = {g["game_pk"]: g for g in statsapi.schedule(lo, hi)}
    snaps = pd.read_csv(p.snaps) if p.snaps.exists() else pd.DataFrame()
    for i, b in open_.iterrows():
        g = results.get(int(b.game_pk))
        if g is None:
            continue
        if g["state"] == "Final" and g["home_score"] is not None:
            hs, as_ = g["home_score"], g["away_score"]
            if b.market == "ML":
                won = (hs > as_) if b.side == g["home"] else (as_ > hs)
                push = False
            else:
                tot = hs + as_
                push = tot == b.line
                won = tot > b.line if b.side == "Over" else tot < b.line
            dec = american_to_decimal(b.price)
            bets.loc[i, "result"] = "P" if push else ("W" if won else "L")
            bets.loc[i, "pnl"] = 0.0 if push else (b.stake * (dec - 1) if won else -b.stake)
            bets.loc[i, "status"] = "settled"
        elif g["detailed"] in ("Postponed", "Cancelled"):
            bets.loc[i, "status"], bets.loc[i, "result"] = "settled", "void"
        cf = closing_fair(snaps, b)
        if cf is not None:
            bets.loc[i, "close_price_fair"] = cf
            bets.loc[i, "clv"] = american_to_decimal(b.price) * cf - 1
    bets.to_csv(p.bets, index=False)
    return report(bets)


def closing_fair(snaps: pd.DataFrame, b) -> float | None:
    """No-vig probability of the bet's side in the last snapshot before first pitch."""
    if snaps.empty or not isinstance(b.event_id, str):
        return None
    s = snaps[(snaps.event_id == b.event_id) & (snaps.snapshot_utc < snaps.commence_utc)]
    if s.empty:
        return None
    s = s[s.snapshot_utc == s.snapshot_utc.max()]
    mv = market_view(s)
    if b.market == "ML" and "p_home" in mv:
        return mv["p_home"] if b.side == s.home.iloc[0] else 1 - mv["p_home"]
    if b.market == "TOTAL" and mv.get("total_line") == b.line:
        return mv["p_over"] if b.side == "Over" else 1 - mv["p_over"]
    return None


def report(bets: pd.DataFrame) -> pd.DataFrame:
    s = bets[(bets.status == "settled") & (bets.result != "void")]
    if s.empty:
        print("No settled bets yet.")
        return bets
    roi = s.pnl.sum() / s.stake.sum()
    print(f"Settled {len(s)} bets: {(s.result == 'W').sum()}-{(s.result == 'L').sum()}-{(s.result == 'P').sum()}, "
          f"P&L ${s.pnl.sum():,.2f}, ROI {roi:+.2%}, avg CLV {s.clv.mean():+.2%} "
          f"(beat close {(s.clv.dropna() > 0).mean():.0%} of {s.clv.notna().sum()})")
    for m, x in s.groupby("market"):
        print(f"  {m}: {len(x)} bets, ROI {x.pnl.sum() / x.stake.sum():+.2%}, CLV {x.clv.mean():+.2%}")
    return bets


def main(argv=None):
    ap = argparse.ArgumentParser(prog="moneyball.live.runner")
    ap.add_argument("cmd", choices=["picks", "snapshot", "settle", "catchup", "oddscheck"])
    ap.add_argument("--date")
    ap.add_argument("--home", default=str(ROOT))
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    p = Paths(Path(a.home))
    cfg = load_config(p)
    if a.cmd == "picks":
        day = (datetime.strptime(a.date, "%Y-%m-%d").date() if a.date
               else datetime.now(ZoneInfo(cfg["timezone"])).date())
        settle(p)
        make_picks(day, p, cfg, dry_run=a.dry_run)
    elif a.cmd == "catchup":
        st = pickle.loads(p.state.read_bytes())
        through = (datetime.strptime(a.date, "%Y-%m-%d").date() if a.date
                   else datetime.now(ZoneInfo(cfg["timezone"])).date() - timedelta(days=1))
        catch_up(st, through, p)
        p.state.write_bytes(pickle.dumps(st))
    elif a.cmd == "oddscheck":
        day = (datetime.strptime(a.date, "%Y-%m-%d").date() if a.date
               else datetime.now(ZoneInfo(cfg["timezone"])).date())
        odds = fetch_odds(cfg["books"], day)
        for ev, o in odds.groupby("event_id", sort=False):
            mv = market_view(o)
            print(f"{o.away.iloc[0]} @ {o.home.iloc[0]}: p_home {mv.get('p_home', float('nan')):.3f}, "
                  f"total {mv.get('total_line')}, best home {mv.get('best_home')}, best away {mv.get('best_away')}")
    elif a.cmd == "snapshot":
        save_snapshot(fetch_odds(cfg["books"], datetime.now(ZoneInfo(cfg["timezone"])).date()), p)
    else:
        settle(p)


if __name__ == "__main__":
    main()
