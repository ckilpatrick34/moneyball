"""Keyless MLB odds from Action Network's public scoreboard feed.

Used when ODDS_API_KEY is not set. Returns the same long format as
runner.flatten_odds, so snapshots, picks and CLV work unchanged. Book prices
are the New Jersey markets; Action Network's consensus and opening lines are
left out so the median no-vig line is built from real books only.
"""

from __future__ import annotations

from datetime import date, datetime, timezone

import pandas as pd
import requests

from .parks import std_abbr

URL = "https://api.actionnetwork.com/web/v1/scoreboard/mlb"
BOOKS = {68: "draftkings", 69: "fanduel", 75: "betmgm", 123: "williamhill_us", 71: "betrivers",
         79: "bet365", 2988: "fanatics"}
HEADERS = {"User-Agent": "Mozilla/5.0 (moneyball-runner)"}


def fetch(day: date, books: list[str] | None = None) -> pd.DataFrame:
    ids = [i for i, k in BOOKS.items() if not books or k in books] or list(BOOKS)
    r = requests.get(URL, timeout=30, headers=HEADERS, params={
        "period": "game", "date": day.strftime("%Y%m%d"), "bookIds": ",".join(map(str, ids))})
    r.raise_for_status()
    return flatten(r.json(), set(ids))


def flatten(payload: dict, book_ids: set | None = None) -> pd.DataFrame:
    now = datetime.now(timezone.utc).isoformat()
    rows = []
    for g in payload.get("games", []):
        teams = {t["id"]: std_abbr(t["abbr"]) for t in g.get("teams", [])}
        home = teams.get(g["home_team_id"])
        away = teams.get(g["away_team_id"])
        if not home or not away:
            continue
        base = {"snapshot_utc": now, "event_id": f"an{g['id']}", "commence_utc": pd.Timestamp(g["start_time"]).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "home": home, "away": away}
        for o in g.get("odds") or []:
            bk = BOOKS.get(o.get("book_id"))
            if bk is None or (book_ids and o["book_id"] not in book_ids) or o.get("type", "game") != "game":
                continue
            for side, col in (("home", "ml_home"), ("away", "ml_away")):
                if o.get(col) is not None:
                    rows.append({**base, "book": bk, "market": "h2h", "side": side, "price": float(o[col]),
                                 "point": None})
            if o.get("total") is not None:
                for side in ("over", "under"):
                    if o.get(side) is not None:
                        rows.append({**base, "book": bk, "market": "totals", "side": side,
                                     "price": float(o[side]), "point": float(o["total"])})
    return pd.DataFrame(rows, columns=["snapshot_utc", "event_id", "commence_utc", "home", "away", "book",
                                       "market", "side", "price", "point"])
