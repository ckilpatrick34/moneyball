# Strategy research: first real backtest (2026-10-29)

## Data (all reachable from the cloud sandbox, no keys)

| Source | What | Coverage |
|---|---|---|
| Retrosheet event files (Chadwick Bureau GitHub mirror) | 1.19M plate appearances, starters, lineups, bullpen usage, game-time weather | MLB 2019–2025 |
| SportsBookReview odds dataset (ArnavSaraogi/mlb-odds-scraper release) | Opening and closing moneyline, run line, totals from 6 US books | 11,040 MLB games, 2021-04 to 2025-08-16 |
| SportsDataverse baseballr-data (stats.ncaa.org) | D1 results | 43,670 college games, 2021–2026 |

11,035 of 11,040 odds games matched to Retrosheet.

## Method

* Point-in-time features (`moneyball/features.py`): starter K%, BB%, HR% and FIP-per-PA, shrunk toward league with stat-specific priors and a 400-day half-life; starter depth (batters faced per start); rest; bullpen FIP and pitches thrown in the last 1 and 3 days; team and lineup wOBA; park run factor; temperature, wind in/out, dome; the run-based team ratings.
* Walk-forward: each season 2022–2025 is predicted by a model trained only on earlier seasons. Logistic regression (L2) for home win and for over at the opening total.
* Bets placed at **opening** prices, so lineup features are excluded from the moneyline model (lineups aren't known at the open). The model probability is blended 50/50 with the no-vig opening line, and we bet when EV is above 3%.
* CLV is measured against the median no-vig closing line.

## Results (out of sample, 2022–2025)

| Strategy | Bets | Flat ROI | t-stat | Avg CLV | Seasons positive |
|---|---|---|---|---|---|
| Moneyline, best opening price across books | 2,958 | +2.6% | 1.2 | +2.0% | 4 / 4 |
| Totals, best opening price across books | 2,536 | +3.8% | 2.0 | n/a* | 3 / 4 |
| **Combined, best opening price** | **5,494** | **+3.1%** | **2.2** | +1.1% | 3 / 4 (2022 flat) |
| Moneyline, median opening price (no shopping) | 801 | −0.5% | −0.1 | −2.6% | 2 / 4 |
| Totals, median opening price | 1,539 | +3.6% | 1.5 | n/a* | 3 / 4 |

\* Same-line CLV is biased for totals because games whose line moved drop out of it. Instead: on totals bets the line moved our way 34% of the time and against us 18%.

Bankroll, combined strategy, $10,000 start: at 1/8 Kelly with a 1% per-bet cap it ends at about $46k, with a 29% max drawdown. At 1/4 Kelly with a 2% cap the drawdown hit 53%, which is too much. Use 1/8 Kelly and a 1% cap.

## What is real and what isn't

1. **The moneyline model has genuine information.** Its disagreement with the opening line predicts the direction of the closing line in every test season (correlation 0.09, 0.15, 0.16, 0.16). The market moves toward us, which is the hardest evidence available without live betting.
2. **That information is only worth money with line shopping.** At the median book's price the vig eats it. At the best of 6 books it nets +2–3%. A placebo (no model, just best price vs consensus) also shows strong CLV. So a large part of the moneyline profit is plain line shopping, and the model decides which side of a shopped line to take.
3. **The totals edge is entirely weather.** Removing temperature and wind kills it (line-move correlation drops from 0.13 to −0.01). The backtest uses *actual* game-time weather, so live results will be somewhat worse unless we use a good forecast at bet time. Temperature forecasts a day out are accurate to a few degrees, so most of it should survive; the wind part less so.
4. **Statistical strength is modest.** A t-stat of 2.2 over 5,494 bets, after trying a handful of thresholds, is promising but not proof. 2022 was flat because it trained on one season only; it improved as training data grew.
5. **Opening lines are not always bettable in size.** Openers come with low limits and move fast. The realistic version is "bet within minutes of the open at whichever book is best", which needs accounts at several books and automation.

## Update: forecast weather and pitcher handedness (research/weather_platoon.py)

No archived weather forecasts are reachable from the sandbox (Open-Meteo, NWS and NOAA are blocked), so the
test simulates day-ahead forecasts from actual game weather. It uses published error levels: temperature
RMSE about 3.5 F, wind speed RMSE about 4 mph, and wind direction (in vs out) wrong or unknown 35% of the
time. Models are trained on actual weather and scored on the noisy "forecast", which is the
conservative way round.

| Totals variant (best opening price, EV > 3%) | Bets | Flat ROI | t | Line moved for / against |
|---|---|---|---|---|
| Actual weather (original backtest) | 2,653 | +3.6% | 2.0 | 34% / 18% |
| Forecast weather, 3 noise draws | 2,709–2,812 | +2.5% to +4.6% | 1.4–2.5 | 33% / 19% |
| Forecast temperature only, no wind | 2,539 | +3.6% | 1.9 | 32% / 19% |
| No weather at all | 1,310 | −2.9% | −1.1 | 24% / 26% |

**The totals edge survives realistic forecast error.** Temperature carries almost all of it, and wind adds
little once its forecast noise is included. Live, use a good temperature forecast and treat wind as optional.

**Pitcher handedness adds nothing.** Team wOBA against the opposing starter's hand (shrunk platoon splits
from Retrosheet rosters) left the moneyline log loss, closing-line-move correlation and ROI unchanged
(+2.5% vs +2.6%). Totals moved within noise. The market already prices platoon matchups, so it stays
out of the model for now. Lineup-level platoon features (`lineup_woba_vs`) are computed and kept for
bets placed after lineups post.

## College

43,670 D1 games, ratings tuned by walk-forward log loss: 66% pick accuracy, log loss 0.613 vs 0.674 for "home team always 60%". No free historical college odds exist that this sandbox can reach, so profitability **cannot be backtested yet**. The plan is to log The Odds API college lines from opening day (mid-February 2027) and judge it by CLV after a few hundred games.

## Next experiments, in order of expected value

1. ~~Weather forecasts~~ done (simulated, edge survives). Live: pull a temperature forecast per park at bet time.
2. ~~Pitcher handedness~~ done (no value).
3. First-5-innings markets: the starter features matter most there, but there's no F5 odds history in this dataset.
4. Better bullpen availability: which specific high-leverage relievers are unavailable.
5. College: weekend rotation (Friday/Saturday/Sunday starter), midweek games, conference strength.
