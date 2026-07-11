# World Cup 2026 Knockout Forecaster

Predicts the rest of the 2026 FIFA World Cup bracket by combining a from-scratch
Elo rating with a set of "knockout mentality" features (penalty-shootout record,
comeback rate, knockout-specific win rate) learned from 2006–2022 World Cup
history, then simulating the remaining bracket with Monte Carlo.

## Why this exists

Predicting a single knockout match is a genuinely noisy problem — one team, one
elimination, one bad bounce. Rather than pretend a model can call outright
winners, this project reports **calibrated win probabilities** and simulates the
bracket forward thousands of times, the same way FiveThirtyEight-style
tournament models work.

The specific question it asks: beyond raw team quality, does a team's *history
of performing under knockout pressure* add real signal? The answer, honestly,
is "some of it does, some of it doesn't" — see Limitations below.

## Result (captured at the Quarterfinal stage, July 11 2026)

| Team | Championship probability |
|---|---|
| France | 36.0% |
| Spain | 28.4% |
| Argentina | 26.0% |
| England | 8.8% |
| Norway | 0.6% |
| Switzerland | 0.2% |

Full reasoning, charts, and the model-vs-live-market comparison are in
`world_cup_2026_forecast.ipynb`.

## Method

1. **Elo rating** — computed match-by-match over 49k+ international matches
   (1872–2026), all competitions, with a competition-importance-weighted K-factor
   and a goal-difference multiplier. Fully chronological — a team's rating on any
   date only reflects matches played before it.
2. **Knockout mentality features** — career-to-date, leakage-free, computed per
   team before every match: knockout matches played, knockout win rate, average
   knockout goal margin, penalty-shootout win rate, comeback rate (win/draw rate
   after conceding the first goal of the match).
3. **Model** — logistic regression and gradient boosting (`HistGradientBoostingClassifier`),
   backtested with **leave-one-tournament-out cross-validation** (train on 4 World
   Cups, test on the held-out 5th, rotate through all 5) so no tournament ever
   leaks into its own test fold.
4. **Live application** — the trained model is pointed at the real, live 2026
   bracket, blended 50/50 with live market-implied odds for the two undecided
   Quarterfinals, then simulated forward 50,000 times.

### Backtest results (5-fold leave-one-tournament-out, 80 knockout matches total)

| Model | Accuracy | Log-loss | Brier |
|---|---|---|---|
| Gradient boosting | 71.3% | 0.573 | 0.183 |
| Logistic regression | 71.3% | 0.593 | 0.201 |
| Elo-only baseline | 71.3% | 0.898 | 0.261 |

All three pick the right team about as often, but the mentality-augmented models
are **much better calibrated** — the Elo-only baseline is a blunt 0/1 pick with no
real probability estimate behind it, which is what the much worse log-loss/Brier
reflects.

## Limitations (read this before trusting any number here)

- **Small sample.** Only 80 World Cup knockout matches exist in the 2006–2022
  window. Treat probabilities as directionally useful, not precise.
- **Shootout record is close to uninformative.** Most teams have played 5 or
  fewer career World Cup shootouts — not enough to reliably detect a "nerve"
  effect, even if one exists.
- **Teams with little World Cup knockout history get a near-blank mentality
  signal.** Norway has 4 career knockout matches and 0 shootouts in this dataset
  — the model effectively falls back to Elo for them.
- **No player-level data.** Injuries, suspensions, and current squad form aren't
  modeled. The clearest evidence of this gap: the model is notably more bullish
  on Argentina and France than live market odds are, likely because the model
  is leaning on historical pedigree the market has already priced past.

## Project layout

```
elo.py                 # chronological Elo rating engine
mentality.py            # knockout-stage tagging + "mentality" feature tracker
build_dataset.py        # builds the 2006-2022 training set (auto-downloads data)
train_backtest.py       # trains + leave-one-tournament-out backtest
forecast_2026.py        # applies the model to the live bracket + Monte Carlo sim
world_cup_2026_forecast.ipynb   # full narrative version with charts
data/                    # auto-downloaded on first run, gitignored
```

## Running it

```bash
pip install -r requirements.txt

# quick path: backtest metrics + coefficients
python train_backtest.py

# live forecast + Monte Carlo bracket simulation
python forecast_2026.py

# full narrative version with charts and write-up
jupyter notebook world_cup_2026_forecast.ipynb
```

No manual data download needed — `build_dataset.py` pulls the CSVs straight from
the source GitHub repo into a local `data/` folder the first time you run
anything. Because the source dataset is live-updated, re-running this after
future match results will pick up newer numbers automatically.

## Data source

[martj42/international_results](https://github.com/martj42/international_results)
— 49k+ international football results, 1872–present, CC0-licensed.
