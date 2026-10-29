# Moneyball: MLB and college baseball betting engine

A rebuild of `mlb_predict_today.py` around the one question that decides
profit: **is our probability better than the price?**

## 1. Review of the current script

| # | Problem | Why it matters |
|---|---------|----------------|
| 1 | No odds anywhere. "Top 5 home sides by edge" ranks by win %, not by edge over a price. | A 70% favourite at -300 is a losing bet. Without prices there is no EV, no bet selection, no sizing. |
| 2 | `simulate_poisson` counts every tie as an away win (`h > a`). | About 10% of games are tied after nine, so every home team is understated by roughly 5 points. That error is 55% of the ensemble. |
| 3 | Team-season stats only. No starting pitchers, lineups, or bullpen. | The starter is the biggest single input to an MLB price, and the market already prices it. |
| 4 | `fetch_team_stats_for_season` pulls *current* season totals, even for past dates. | Running it on any past date uses future data, so anything trained or evaluated that way is leaky. How `mlb_model.pkl` was trained is unknown, which is a red flag for the same reason. |
| 5 | The "xFIP" and "SIERA" are the same number (team xwOBA allowed) rescaled twice, and `HOME_ADVAN` is a constant column. | The ML model gets two copies of one feature plus a useless one. |
| 6 | Fatigue = games in the last 3 days, up to +18% runs. | Almost every team plays 2 to 3 games in 3 days, so this is mostly noise with a large weight. |
| 7 | Poisson runs. | MLB scoring has roughly twice the Poisson variance, which skews totals and run lines. |
| 8 | No backtest, no closing-line tracking, no bet log. | There is no way to know whether it makes money before risking money. |
| 9 | No college support. | |

## 2. Architecture

```
data/                 MLB Stats API (results, actual starters), ESPN college scoreboard,
                      The Odds API (live odds + snapshot log in SQLite for closing lines)
ratings.py            Point-in-time run ratings: team offense, team run prevention,
                      MLB starter rating. Updated game by game, so no look-ahead.
runs.py               Expected runs -> exact win / run line / total probabilities.
                      Negative binomial, extra innings handled.
odds.py               American/decimal, de-vig (power method), EV, CLV.
sizing.py             Quarter Kelly with a 2% per-bet cap.
backtest.py           Walk-forward over history at recorded prices: log loss vs the
                      market, CLV, ROI, drawdown, by season and market.
cli.py                mlb-history, ncaa-history, backtest, snapshot-odds, today.
synthetic.py          Fake league with known truth, used by the tests.
```

Bet decision: model probability is blended with the no-vig market probability
(50/50 in logit space by default, `--market-weight`), then we bet only if EV at
the **best available price** is at least 2.5% and the model and market disagree
by less than 12 points (bigger gaps are almost always bad data, such as a
scratched starter).

## 3. Quick start (on your Mac)

```bash
cd moneyball
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m pytest -q                         # 11 tests, all offline

# History (boxscores are cached; first run of a season takes ~10 minutes)
python -m moneyball.cli mlb-history --seasons 2023 2024 2025
python -m moneyball.cli ncaa-history --seasons 2025 2026

# Live slate (needs a free key from the-odds-api.com)
export ODDS_API_KEY=...
python -m moneyball.cli today --league mlb --bankroll 1000
python -m moneyball.cli today --league ncaa --bankroll 1000

# Log odds every 30 min on game days to build your own closing lines
python -m moneyball.cli snapshot-odds --league mlb
```

## 3b. Live daily runner (GitHub Actions)

`moneyball/live/runner.py` runs the backtested strategy every morning in season:

1. **Catch up.** Yesterday's boxscores from the MLB Stats API roll the feature state forward
   (`state/mlb_state.pkl`). This is the same `FeatureState` the backtest used, so the features match.
2. **Price the slate.** It pulls the probable starters, a temperature forecast at first pitch (Open-Meteo,
   with a roof rule for retractable parks) and the odds from your books (The Odds API).
3. **Pick.** The models in `models/mlb_models.json` (trained on 2021–2025) are blended 50/50 with the
   no-vig consensus line. It bets the best price across your books when EV > 3%, sized at 1/8 Kelly with a
   1% bankroll cap.
4. **Log and grade.** Picks go to `picks/<date>.md` and `logs/bets.csv`. Bets are graded the next morning,
   and CLV is computed from the last odds snapshot before first pitch.

Setup, once:

1. Push this folder to a private GitHub repo.
2. Odds work with no key: without `ODDS_API_KEY` the runner pulls DraftKings, FanDuel, BetMGM, Caesars,
   BetRivers, bet365 and Fanatics prices from Action Network's public feed. Adding an `ODDS_API_KEY` secret
   (**Settings → Secrets and variables → Actions**) switches to The Odds API instead. Optionally add
   `DISCORD_WEBHOOK_URL` to get the card in Discord.
3. Edit `config.json`: set `bankroll_start` and list only the `books` you actually have accounts with
   (Odds API keys: draftkings, fanduel, betmgm, williamhill_us = Caesars, betrivers, espnbet, fanatics, ...).
4. The state in `state/mlb_state.pkl` is already current through the 2026 season (2,470 games, scored
   out of sample: log loss 0.681 vs 0.692 home-rate baseline, 56.0% accuracy). Each morning run catches up
   on its own; a manual `catchup` run is only needed after a long gap.

After that it runs on its own: picks at 10:00 CT, odds snapshots at 12:30 and 17:30 CT, March through
October. With an Odds API key that uses about 190 credits a month (the free tier has 500).

Rebuild the models and state from scratch with `research/train_live.py`.

## 4. Backtest input schema

One row per game, CSV:

- required: `date, season, home, away, home_score, away_score`
- optional: `game_id, neutral, home_sp, away_sp, home_sp_ip, away_sp_ip`
- prices you could have bet: `home_ml, away_ml, total_line, over_price, under_price`
- closing prices for CLV: `home_ml_close, away_ml_close, total_line_close, over_close, under_close`

`mlb-history` produces the first two groups. Odds have to be joined from a
historical source: The Odds API historical endpoint (paid plan), or a free
archive of past MLB lines if you have one. Join on date + home + away
(double-headers by start time).

## 5. What to expect, honestly

On a synthetic league the results-only ratings track the truth (correlation
~0.76) but do **not** beat a market whose prices are off by 5 points on
average; they only beat a sloppier one. Real MLB moneylines are sharper than
that. So this version is the measuring stick and the plumbing, not the edge.
The edge has to come from information the price underweights, tested one
piece at a time against the closing line:

1. **Starter quality from skill stats** (K-BB%, xFIP, Stuff+, velocity trend),
   not runs allowed, plus a projected innings count.
2. **Bullpen availability** from the last 3 days of reliever pitch counts
   (the right version of the old "fatigue" idea).
3. **Confirmed lineups** vs. projected, and platoon splits.
4. **Weather and park** for totals (wind at Wrigley, heat, roof status).
5. **College**: softer markets with low limits. Midweek games, weekend
   rotation spots (Friday ace vs Sunday starter), and early-season games
   before the market adjusts are where an edge is most plausible.
6. **Timing**: bet early lines where the model disagrees, and track CLV.
   Positive average CLV over 300+ bets is the signal to scale up; ROI over
   a few hundred bets is mostly luck.

## 6. Roadmap

See **FINDINGS.md** for the first real backtest (2022-2025 out of sample).
Reproduce it with `python research/mlb_strategy.py --retro <retrosheet>/seasons --odds mlb_odds_dataset.json`.

- [x] Core: odds math, sizing, run distributions, ratings, walk-forward backtest, tests
- [x] Data adapters: MLB Stats API, ESPN college, The Odds API + snapshot log
- [x] Retrosheet play-by-play parser + point-in-time features (`features.py`)
- [x] Historical odds (SBR dataset 2021-2025), first real backtest with CLV
- [x] College results 2021-2026 (SportsDataverse), ratings tuned
- [ ] Starter skill ratings from Statcast / FanGraphs
- [ ] Bullpen availability, lineups, weather
- [ ] College: team-name alias file, weekend rotation, conference strength
- [ ] Daily automation: snapshot odds on a schedule, bet log, CLV report
