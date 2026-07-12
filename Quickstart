<div align="center">

# 🏆 World Cup 2026 Knockout Forecaster

### Elo ratings + career knockout-mentality features, trained on 2006–2022, validated live against the 2026 bracket.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Built%20with-Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-Logit%20%2B%20GBM-F7931E?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![Data](https://img.shields.io/badge/Match%20history-1872--2026-8A2BE2)]()
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Last Commit](https://img.shields.io/github/last-commit/AkshPaliwal/FIFA-WORLD-CUP-PREDICTION-ORACLE-AI-)]()

</div>

<br/>

## ⚡ What this is

A live-updating knockout forecaster for the 2026 FIFA World Cup. Every matchup is scored two ways:

- **Elo**, computed from 150+ years of international match history (1872–2026)
- **Knockout mentality features** — a team's career penalty-shootout record, comeback rate, and knockout win rate specifically in World Cup knockout matches, on the theory that "big game temperament" is a real, measurable signal separate from overall strength

Both feed a logistic regression and a gradient-boosted model, ensembled together and cross-validated with a **leave-one-tournament-out backtest** across every World Cup from 2006–2022.

Predictions are frozen the moment they're made — and automatically checked against real results the moment matches finish. No manual updating.

<br/>

## 🎯 Live prediction record

The model doesn't just predict — it keeps score on itself.

| Match | Result | Model said (pre-match) | Market said | Call |
|---|---|---|---|---|
| 🏴󠁧󠁢󠁥󠁮󠁧󠁿 England vs Norway | **2–1** | England 62% | England 62% | ✅ Correct |
| 🇦🇷 Argentina vs Switzerland | **3–1** | Argentina 82% | Argentina 70% | ✅ Correct — model was *more* confident than the market, and right |

<sub>Updated automatically as later rounds finish — see <a href="#-live-result-tracking">Live Result Tracking</a> below.</sub>

<br/>

## 🧠 Model validation

Leave-one-tournament-out backtest, 2006–2022, 80 knockout matches:

| Model | Accuracy | Log Loss ↓ | Brier ↓ |
|---|---|---|---|
| Elo-only baseline | 0.712 | 0.898 | 0.261 |
| Gradient Boosting | 0.675 | 0.585 | 0.194 |
| **Logistic Regression** | **0.738** | 0.585 | 0.197 |

All three pick the winner about as often — but the mentality-augmented models are **substantially better calibrated** (lower log-loss/Brier) than Elo alone. Translation: when the augmented models say 70%, it actually happens close to 70% of the time.

<br/>

## 🗺️ How a prediction flows through the system

```mermaid
flowchart LR
    A[1872–2026 match history] --> B[Elo engine]
    C[WC knockout matches 2006–2026] --> D[Mentality tracker<br/>shootouts · comebacks · win rate]
    B --> E[Feature vector]
    D --> E
    E --> F[Logistic Regression]
    E --> G[Gradient Boosting]
    F --> H[Ensemble]
    G --> H
    H --> I[Prediction frozen<br/>SQLite log]
    J[SportAPI7<br/>live results] -->|match finishes| K{Team names<br/>match a frozen<br/>prediction?}
    I --> K
    K -->|yes| L[✅ Result card<br/>auto-rendered]
```

<br/>

## ✨ Features

- 🏟️ **Bracket & Odds** — Monte Carlo–simulated championship odds across the full remaining bracket, with model/market divergence flagged automatically
- 🔍 **Team Explorer** — Elo history back to 1872, knockout record, and shootout stats for any national team
- ⚔️ **Head-to-Head** — run any two teams against each other, neutral venue or not, regardless of whether they're in the live bracket
- 📊 **Model Validation** — full backtest transparency, per-tournament breakdown included
- 🔄 **Live Result Tracking** — see below

<br/>

## 🔄 Live result tracking

This is the part that makes the app self-maintaining:

1. **Prediction freezing** — the instant a fixture is upcoming, its model + market probability is written *once* to a local SQLite log. Reruns never overwrite it — that's the permanent record of "what we called beforehand."
2. **Automatic verification** — on every page load, the app checks [SportAPI7](https://rapidapi.com/rapidsportapi/api/sportapi7) for finished matches, matches them by team name against the frozen predictions, and swaps the live slider for a result card automatically.

No match ever needs its result typed in by hand.

<details>
<summary><strong>Setup: environment variables</strong></summary>

<br/>

```bash
export RAPIDAPI_KEY="your-rapidapi-key"
```

or, for Streamlit Cloud / persistent local use, create `.streamlit/secrets.toml`:

```toml
RAPIDAPI_KEY = "your-rapidapi-key"
```

</details>

<br/>

## 🚀 Quickstart

```bash
git clone https://github.com/AkshPaliwal/FIFA-WORLD-CUP-PREDICTION-ORACLE-AI-.git
cd FIFA-WORLD-CUP-PREDICTION-ORACLE-AI-

pip install -r requirements.txt
export RAPIDAPI_KEY="your-rapidapi-key"

streamlit run app.py
```

Then open **http://localhost:8501**.

<br/>

## 📁 Project structure

```
.
├── app.py                 # Streamlit dashboard — all 4 tabs
├── elo.py                 # Elo rating engine (1872–2026)
├── mentality.py            # Knockout-mentality feature tracker
├── build_dataset.py        # Feature matrix construction
├── train_backtest.py       # Model training + leave-one-tournament-out backtest
├── live_results.py         # SportAPI7 client + prediction-freezing SQLite log
├── requirements.txt
└── .streamlit/
    └── secrets.toml         # (gitignored) RAPIDAPI_KEY lives here
```

<br/>

<details>
<summary><strong>🛣️ Roadmap</strong></summary>

<br/>

- [ ] Auto-populate the Final once it's scheduled
- [ ] Deploy properly always-on (Streamlit Community Cloud) so results update even with no one watching the tab
- [ ] Historical accuracy tracker across the whole tournament, not just the two most recent rounds
- [ ] Confidence intervals on championship odds, not just point estimates

</details>

<br/>

## 📜 License

MIT — see [LICENSE](LICENSE).

<br/>

<div align="center">

Built by <a href="https://github.com/AkshPaliwal">Aksh Paliwal</a>

</div>
