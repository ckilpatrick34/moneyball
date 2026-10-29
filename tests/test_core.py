import math

import numpy as np
import pandas as pd
import pytest

from moneyball.backtest import run_backtest
from moneyball.odds import american_to_decimal, clv, expected_value, fair_probs, implied_prob
from moneyball.ratings import RatingModel
from moneyball.runs import game_probs
from moneyball.sizing import SizingConfig, kelly, stake
from moneyball.synthetic import make_league


def test_odds_conversions():
    assert american_to_decimal(-150) == pytest.approx(1 + 100 / 150)
    assert american_to_decimal(130) == pytest.approx(2.3)
    assert implied_prob(-110) == pytest.approx(0.5238, abs=1e-4)


def test_devig_sums_to_one_and_symmetric():
    ph, pa = fair_probs([-110, -110])
    assert ph == pytest.approx(0.5) and pa == pytest.approx(0.5)
    ph, pa = fair_probs([-200, 170])
    assert ph + pa == pytest.approx(1.0)
    assert ph > 0.62


def test_ev_and_kelly():
    assert expected_value(0.5, 100) == pytest.approx(0.0)
    assert expected_value(0.55, 100) == pytest.approx(0.10)
    assert kelly(0.55, 100) == pytest.approx(0.10)
    assert kelly(0.45, 100) == 0.0
    # capped at max_bet_pct
    assert stake(0.70, 100, 1000, SizingConfig(kelly_fraction=1.0, max_bet_pct=0.02)) == 20.0


def test_clv_sign():
    assert clv(+120, 0.5) > 0       # got +120 on a coin flip
    assert clv(-120, 0.5) < 0


def test_equal_teams_ties_go_to_extras():
    gp = game_probs(4.5, 4.5, extra_home_win=0.5)
    assert gp.home_win == pytest.approx(0.5, abs=1e-9)
    assert 0.08 < gp.p_tie_reg < 0.14  # ~10% of MLB games are tied after 9


def test_total_probs_partition():
    gp = game_probs(4.6, 4.2)
    o, u, p = gp.total_probs(9.0)
    assert o + u + p == pytest.approx(1.0, abs=1e-6)
    o, u, p = gp.total_probs(8.5)
    assert p == pytest.approx(0.0)


def test_run_line_partition():
    gp = game_probs(5.0, 4.0)
    h, a, p = gp.home_cover_prob(-1.5)
    assert h + a + p == pytest.approx(1.0, abs=1e-9)
    assert h < gp.home_win


def test_ratings_learn_strength():
    m = RatingModel()
    for _ in range(60):
        m.update("GOOD", "BAD", 7, 2)
        m.update("BAD", "GOOD", 2, 7)
    assert m.predict("GOOD", "BAD").home_win > 0.65
    assert m.predict("BAD", "GOOD").home_win < 0.35


def test_no_lookahead():
    """Changing results on a day must not change that day's predictions."""
    df = make_league(n_teams=10, seasons=(2022, 2023), days_per_season=40, games_per_day=5)
    base = run_backtest(df).predictions
    last_day = df["date"].max()
    flipped = df.copy()
    m = flipped["date"] == last_day
    flipped.loc[m, ["home_score", "away_score"]] = flipped.loc[m, ["away_score", "home_score"]].values
    again = run_backtest(flipped).predictions
    a = base[base["date"].astype(str) == last_day]["p_home"].values
    b = again[again["date"].astype(str) == last_day]["p_home"].values
    np.testing.assert_allclose(a, b)


def test_backtest_beats_noisy_market_on_synthetic():
    # Pipeline check only. A results-only rating does NOT beat a market with
    # 0.05 noise (see README); it needs a sloppy market to show an edge.
    df = make_league(seed=11, market_noise=0.08)
    res = run_backtest(df)
    s = res.summary
    assert s["bets"] > 100
    assert s["model_log_loss_vs_mkt"] < s["market_log_loss"]
    assert s["avg_clv"] > 0


def test_random_model_has_no_clv():
    """Sanity: if the model knows nothing, CLV should be ~0, not positive."""
    df = make_league(seed=3, market_noise=0.0)
    df["home"], df["away"] = df["home"].sample(frac=1, random_state=1).values, df["away"].values
    res = run_backtest(df)
    if res.summary["bets"]:
        assert res.summary["avg_clv"] < 0.01


def test_retrosheet_classify():
    from moneyball.data.retrosheet import classify
    assert classify("K") == "K"
    assert classify("K+WP.B-1") == "K"
    assert classify("W.1-2") == "BB"
    assert classify("IW") == "BB"
    assert classify("HP") == "HBP"
    assert classify("HR/F78") == "HR"
    assert classify("S8/G.2-H") == "1B"
    assert classify("D9/L9L") == "2B"
    assert classify("63/G") == "OUT"
    assert classify("E6/G") == "ROE"
    assert classify("SB2") is None
    assert classify("NP") is None
    assert classify("WP.1-2") is None
    assert classify("PO1(E1)") is None
