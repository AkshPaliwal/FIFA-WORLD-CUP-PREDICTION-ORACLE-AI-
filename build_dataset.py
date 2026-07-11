import pandas as pd
import numpy as np
from elo import compute_elo_history, rating_as_of
from mentality import (
    tag_knockout_matches,
    match_shootout_winner,
    first_goal_scorer_per_match,
    MentalityTracker,
)

FEATURE_COLS = [
    "elo_diff",
    "ko_played_diff",
    "ko_win_rate_diff",
    "ko_avg_margin_diff",
    "shootout_win_rate_diff",
    "comeback_rate_diff",
    "neutral",
]


DATA_URL = "https://raw.githubusercontent.com/martj42/international_results/master/"
DATA_FILES = ["results.csv", "shootouts.csv", "goalscorers.csv"]


def ensure_data(data_dir="data"):
    import os
    import urllib.request

    os.makedirs(data_dir, exist_ok=True)
    for fname in DATA_FILES:
        path = os.path.join(data_dir, fname)
        if not os.path.exists(path):
            print(f"Downloading {fname}...")
            urllib.request.urlretrieve(DATA_URL + fname, path)


def load_raw():
    ensure_data()
    df = pd.read_csv("data/results.csv", parse_dates=["date"])
    so = pd.read_csv("data/shootouts.csv")
    gs = pd.read_csv("data/goalscorers.csv")
    return df, so, gs


def build_all(df, so, gs, years):
    # Elo uses the FULL match history (all competitions) up to each match date
    elo_timeline, _ = compute_elo_history(df)

    wc = df[df["tournament"] == "FIFA World Cup"].copy()
    wc = tag_knockout_matches(wc)
    wc_ko_all = wc[wc["stage"] == "knockout"].copy()
    wc_ko_all = match_shootout_winner(wc_ko_all, so)
    first_goal = first_goal_scorer_per_match(gs)

    tracker = MentalityTracker(wc_ko_all, first_goal)

    rows = []
    for row, fh, fa in tracker.run():
        year = row.date.year
        if year not in years:
            continue
        elo_h = rating_as_of(elo_timeline, row.home_team, row.date)
        elo_a = rating_as_of(elo_timeline, row.away_team, row.date)

        label = 1 if row.true_winner == row.home_team else 0

        rows.append(
            dict(
                date=row.date,
                year=year,
                home_team=row.home_team,
                away_team=row.away_team,
                neutral=1 if row.neutral else 0,
                elo_diff=elo_h - elo_a,
                ko_played_diff=fh["ko_played"] - fa["ko_played"],
                ko_win_rate_diff=fh["ko_win_rate"] - fa["ko_win_rate"],
                ko_avg_margin_diff=fh["ko_avg_margin"] - fa["ko_avg_margin"],
                shootout_win_rate_diff=fh["shootout_win_rate"] - fa["shootout_win_rate"],
                comeback_rate_diff=fh["comeback_rate"] - fa["comeback_rate"],
                label=label,
            )
        )
    out = pd.DataFrame(rows)
    # Missing career stats (team had zero prior knockout matches) -> 0 diff (neutral prior)
    for c in FEATURE_COLS:
        if c in out.columns:
            out[c] = out[c].fillna(0.0)
    return out, elo_timeline, tracker


if __name__ == "__main__":
    df, so, gs = load_raw()
    data, elo_timeline, tracker = build_all(df, so, gs, years=[2006, 2010, 2014, 2018, 2022])
    print(data.shape)
    print(data.head(10).to_string())
    print("\nLabel balance:", data["label"].mean())
    data.to_csv("data/training_set.csv", index=False)
