"""End-to-end runner tests with recorded-shape API payloads (no network)."""
import json
import pickle
import shutil
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from moneyball.live import runner, statsapi
from moneyball.live.parks import roof_closed, std_abbr

ROOT = Path(__file__).resolve().parents[1]
HAVE_ARTIFACTS = (ROOT / "models" / "mlb_models.json").exists() and (ROOT / "state" / "mlb_state.pkl").exists()

SKUBAL, COLE, JUDGE, TORKELSON = 669373, 543037, 592450, 679529


def boxscore_payload():
    def pitcher(pid, bf, k, bb, hbp, hr, pitches):
        return {"stats": {"pitching": {"battersFaced": bf, "strikeOuts": k, "baseOnBalls": bb,
                                       "hitByPitch": hbp, "homeRuns": hr, "numberOfPitches": pitches}}}

    def batter(pa, h, d, t, hr, bb, hbp):
        return {"stats": {"batting": {"plateAppearances": pa, "hits": h, "doubles": d, "triples": t,
                                      "homeRuns": hr, "baseOnBalls": bb, "hitByPitch": hbp}}}
    return {"teams": {
        "home": {"pitchers": [COLE, 1], "batters": [JUDGE],
                 "players": {f"ID{COLE}": pitcher(COLE, 25, 8, 2, 0, 1, 98), "ID1": pitcher(1, 10, 3, 1, 0, 0, 40),
                             f"ID{JUDGE}": batter(5, 2, 1, 0, 1, 1, 0)}},
        "away": {"pitchers": [SKUBAL], "batters": [TORKELSON],
                 "players": {f"ID{SKUBAL}": pitcher(SKUBAL, 33, 11, 1, 1, 1, 104),
                             f"ID{TORKELSON}": batter(4, 0, 0, 0, 0, 0, 0)}}}}


def test_record_from_boxscore():
    g = {"home": "NYY", "away": "DET", "home_score": 3, "away_score": 2}
    rec = statsapi.record_from_boxscore(boxscore_payload(), g, {SKUBAL: ("L", "L")})
    assert rec["home_sp"] == COLE and rec["away_sp"] == SKUBAL and rec["site"] == "NYC21"
    cole = [p for p in rec["pitchers"] if p[0] == COLE][0]
    assert cole == (COLE, 1, 25, 8, 2, 1, 98)
    judge = [b for b in rec["batters"] if b[0] == JUDGE][0]
    assert judge[2] == "L"                              # faced a lefty starter
    assert judge[4] == pytest.approx(1.25 + 2.03 + 0.69 + 0.9 * 0.006 * 5)


def test_names_and_roofs():
    assert std_abbr("AZ") == "ARI" and std_abbr("Athletics") == "OAK" and std_abbr("New York Yankees") == "NYY"
    assert roof_closed("HOU", 75, 0) and not roof_closed("NYY", 40, 90)
    assert roof_closed("ARI", 104, 0) and not roof_closed("ARI", 75, 0)


def odds_payload(commence="2026-05-01T23:05:00Z", home_px=(-150, -145), away_px=(130, 135),
                 total=8.5, over=(-110, -105), under=(-110, -115)):
    books = []
    for i, key in enumerate(["draftkings", "fanduel"]):
        books.append({"key": key, "markets": [
            {"key": "h2h", "outcomes": [{"name": "New York Yankees", "price": home_px[i]},
                                        {"name": "Detroit Tigers", "price": away_px[i]}]},
            {"key": "totals", "outcomes": [{"name": "Over", "price": over[i], "point": total},
                                           {"name": "Under", "price": under[i], "point": total}]}]})
    return [{"id": "ev1", "commence_time": commence, "home_team": "New York Yankees",
             "away_team": "Detroit Tigers", "bookmakers": books}]


@pytest.mark.skipif(not HAVE_ARTIFACTS, reason="run research/train_live.py first")
def test_end_to_end_picks_settle_clv(tmp_path):
    for sub in ("models", "state"):
        shutil.copytree(ROOT / sub, tmp_path / sub)
    p = runner.Paths(tmp_path)
    cfg = dict(runner.DEFAULT_CONFIG, min_ev=-1.0)          # force picks so the whole path runs
    sched = [{"game_pk": 1, "date": "2026-05-01", "start_utc": "2026-05-01T23:05:00Z", "state": "Preview",
              "detailed": "Scheduled", "home": "NYY", "away": "DET", "home_score": None, "away_score": None,
              "home_sp": COLE, "away_sp": SKUBAL, "home_sp_name": "Gerrit Cole", "away_sp_name": "Tarik Skubal"}]
    odds = runner.flatten_odds(odds_payload())
    runner.save_snapshot(odds, p)
    wx = lambda team, t: {"temp": 80.0, "wind_out": 0.0, "wind_in": 0.0, "dome": 0.0, "night": 1.0,  # noqa: E731
                          "forecast_temp": 80.0, "precip_prob": 0}
    picks = runner.make_picks(date(2026, 5, 1), p, cfg, odds=odds, schedule_rows=sched, weather_fn=wx)
    assert len(picks) >= 1
    assert (tmp_path / "picks" / "2026-05-01.md").exists()
    assert picks.stake.max() <= runner.DEFAULT_CONFIG["bankroll_start"] * cfg["max_bet_pct"] + 1e-9
    # Skubal (elite) vs Cole on the road: the model should like DET more than the market's ~40%
    bets = pd.read_csv(p.bets)
    # closing snapshot moves toward DET
    close = runner.flatten_odds(odds_payload(home_px=(-120, -118), away_px=(105, 102)))
    close["snapshot_utc"] = "2026-05-01T22:59:00+00:00"
    runner.save_snapshot(close, p)
    done = runner.settle(p, results={1: {**sched[0], "state": "Final", "home_score": 2, "away_score": 5}})
    s = done[done.status == "settled"]
    assert len(s) == len(bets)
    assert s.clv.notna().any()
    det = s[(s.market == "ML") & (s.side == "DET")]
    if len(det):
        assert det.result.iloc[0] == "W" and det.clv.iloc[0] > 0


@pytest.mark.skipif(not HAVE_ARTIFACTS, reason="run research/train_live.py first")
def test_catch_up_feeds_boxscores(tmp_path, monkeypatch):
    st = pickle.loads((ROOT / "state" / "mlb_state.pkl").read_bytes())
    before = st.pit.get(COLE, pd.Timestamp("2026-11-05"))["pa"]
    games = [{"game_pk": 9, "date": "2026-11-05", "state": "Final", "detailed": "Final", "home": "NYY", "away": "DET",
              "home_score": 3, "away_score": 2, "start_utc": "2026-11-05T23:05:00Z"}]
    monkeypatch.setattr(statsapi, "schedule", lambda a, b, game_types="R": games if a <= "2026-11-05" <= b else [])
    monkeypatch.setattr(statsapi, "boxscore", lambda pk, cache=None: boxscore_payload())
    n = runner.catch_up(st, date(2026, 11, 10), runner.Paths(tmp_path), verbose=False)
    assert n == 1 and st.pit.v[COLE]["pa"] == pytest.approx(before + 25)
    assert st.last_update == pd.Timestamp("2026-11-10")


def test_action_network_flatten_matches_odds_api_format():
    from moneyball.live import action_network
    payload = {"games": [{"id": 1, "start_time": "2026-05-01T23:05:00.000Z", "home_team_id": 2, "away_team_id": 1,
                          "teams": [{"id": 1, "abbr": "ATH"}, {"id": 2, "abbr": "NYY"}],
                          "odds": [{"book_id": 68, "type": "game", "ml_home": -150, "ml_away": 130, "total": 8.5,
                                    "over": -110, "under": -110},
                                   {"book_id": 15, "type": "game", "ml_home": -148, "ml_away": 128}]}]}
    df = action_network.flatten(payload)
    assert set(df.book) == {"draftkings"}                       # consensus line dropped
    assert (df.away == "OAK").all() and (df.home == "NYY").all()
    assert df.commence_utc.iloc[0] == "2026-05-01T23:05:00Z"
    mv = runner.market_view(df)
    assert mv["total_line"] == 8.5 and 0.55 < mv["p_home"] < 0.6 and mv["best_home"] == (-150.0, "draftkings")
