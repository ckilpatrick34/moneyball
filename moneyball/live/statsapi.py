"""Live MLB Stats API adapters producing `state.FeatureState` records.

Boxscore lines are converted to the same per-game record the Retrosheet path
builds. Reached-on-error isn't in boxscores, so batters get the league ROE
rate (0.6% of PA) at its wOBA weight to keep the scale identical.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import requests

from ..state import WOBA_W
from .parks import PARKS, std_abbr

BASE = "https://statsapi.mlb.com/api/v1"
ROE_RATE = 0.006


def _get(url, params=None, retries=4):
    last = None
    for i in range(retries):
        try:
            r = requests.get(url, params=params, timeout=30, headers={"User-Agent": "moneyball/0.2"})
            r.raise_for_status()
            return r.json()
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(2 * (i + 1))
    raise RuntimeError(f"GET {url} failed: {last}")


def schedule(start: str, end: str, game_types: str = "R") -> list[dict]:
    """Games with probable starters (MLBAM ids). game_types e.g. "R" or "R,F,D,L,W" (with postseason)."""
    data = _get(f"{BASE}/schedule", {"sportId": 1, "startDate": start, "endDate": end, "gameType": game_types,
                                     "hydrate": "probablePitcher,team"})
    out = []
    for d in data.get("dates", []):
        for g in d.get("games", []):
            t = g["teams"]
            out.append({
                "game_pk": g["gamePk"], "date": d["date"], "start_utc": g.get("gameDate"),
                "state": g["status"]["abstractGameState"], "detailed": g["status"]["detailedState"],
                "home": std_abbr(t["home"]["team"].get("abbreviation") or t["home"]["team"]["name"]),
                "away": std_abbr(t["away"]["team"].get("abbreviation") or t["away"]["team"]["name"]),
                "home_score": t["home"].get("score"), "away_score": t["away"].get("score"),
                "home_sp": (t["home"].get("probablePitcher") or {}).get("id"),
                "away_sp": (t["away"].get("probablePitcher") or {}).get("id"),
                "home_sp_name": (t["home"].get("probablePitcher") or {}).get("fullName"),
                "away_sp_name": (t["away"].get("probablePitcher") or {}).get("fullName"),
                "venue": (g.get("venue") or {}).get("name"),
            })
    return out


def record_from_boxscore(box: dict, game: dict, hands: dict) -> dict:
    """Convert a /game/{pk}/boxscore payload into a FeatureState record."""
    sides = {"home": 1, "away": 0}
    sps, pitchers, batters = {}, [], []
    for side, sidx in sides.items():
        team = box["teams"][side]
        pids = team.get("pitchers") or []
        sps[sidx] = pids[0] if pids else None
        for pid in pids:
            s = team["players"].get(f"ID{pid}", {}).get("stats", {}).get("pitching", {})
            bf = int(s.get("battersFaced") or 0)
            pitchers.append((pid, sidx, bf, int(s.get("strikeOuts") or 0),
                             int(s.get("baseOnBalls") or 0) + int(s.get("hitByPitch") or 0),
                             int(s.get("homeRuns") or 0),
                             int(s.get("numberOfPitches") or s.get("pitchesThrown") or 0)))
    for side, sidx in sides.items():
        team = box["teams"][side]
        opp_sp = sps[1 - sidx]
        throws = hands.get(opp_sp, ("R", "R"))[1]
        for bid in team.get("batters") or []:
            s = team["players"].get(f"ID{bid}", {}).get("stats", {}).get("batting", {})
            pa = int(s.get("plateAppearances") or 0)
            if pa == 0:
                continue
            h, d2, d3, hr = (int(s.get(k) or 0) for k in ("hits", "doubles", "triples", "homeRuns"))
            bb, hbp = int(s.get("baseOnBalls") or 0), int(s.get("hitByPitch") or 0)
            w = (WOBA_W["BB"] * bb + WOBA_W["HBP"] * hbp + WOBA_W["1B"] * (h - d2 - d3 - hr)
                 + WOBA_W["2B"] * d2 + WOBA_W["3B"] * d3 + WOBA_W["HR"] * hr + WOBA_W["ROE"] * ROE_RATE * pa)
            batters.append((bid, sidx, throws, pa, w))
    home, away = game["home"], game["away"]
    return {"home": home, "away": away, "home_sp": sps[1], "away_sp": sps[0],
            "site": PARKS.get(home, (None,))[0],
            "home_runs": game.get("home_score"), "away_runs": game.get("away_score"),
            "pitchers": pitchers, "batters": batters}


def boxscore(game_pk: int, cache_dir: Path | None = None) -> dict:
    if cache_dir:
        f = Path(cache_dir) / f"{game_pk}.json"
        if f.exists():
            return json.loads(f.read_text())
    box = _get(f"{BASE}/game/{game_pk}/boxscore")
    if cache_dir:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        (Path(cache_dir) / f"{game_pk}.json").write_text(json.dumps(box))
    return box


def people_hands(ids) -> dict:
    """MLBAM id -> (bats, throws) for players missing from the state."""
    ids = [str(i) for i in ids if i]
    out = {}
    for i in range(0, len(ids), 100):
        data = _get(f"{BASE}/people", {"personIds": ",".join(ids[i:i + 100])})
        for p in data.get("people", []):
            out[p["id"]] = ((p.get("batSide") or {}).get("code", "R"), (p.get("pitchHand") or {}).get("code", "R"))
    return out
