"""
Chronological Elo rating engine for international football.
Loosely follows the World Football Elo Ratings methodology:
- K varies by competition importance
- goal-difference multiplier rewards/punishes margin of victory
- small home-advantage bump when match is not on neutral ground
"""
import pandas as pd
import numpy as np
from collections import defaultdict

K_BY_TOURNAMENT = {
    "FIFA World Cup": 60,
    "FIFA World Cup qualification": 40,
}
DEFAULT_K = 30  # continental championships, other competitive fixtures
FRIENDLY_K = 20
HOME_ADV = 60  # rating-point bump applied to the home side's expected-score calc


def k_for(tournament: str) -> float:
    if tournament in K_BY_TOURNAMENT:
        return K_BY_TOURNAMENT[tournament]
    t = tournament.lower()
    if "friendly" in t:
        return FRIENDLY_K
    if "qualif" in t:
        return 40
    if "cup" in t or "championship" in t or "copa" in t or "euro" in t:
        return 45
    return DEFAULT_K


def goal_diff_multiplier(gd: int) -> float:
    gd = abs(gd)
    if gd <= 1:
        return 1.0
    elif gd == 2:
        return 1.5
    else:
        return (11 + gd) / 8


def compute_elo_history(results: pd.DataFrame, start_rating: float = 1500.0):
    """
    results: dataframe sorted chronologically with columns
      date, home_team, away_team, home_score, away_score, tournament, neutral
    Returns:
      rating_timeline: dict[team] -> list of (date, rating_after_this_match)
      final_ratings: dict[team] -> current rating
    """
    ratings = defaultdict(lambda: start_rating)
    timeline = defaultdict(list)

    df = results.sort_values("date").reset_index(drop=True)
    for row in df.itertuples(index=False):
        if pd.isna(row.home_score) or pd.isna(row.away_score):
            continue  # unplayed fixture
        h, a = row.home_team, row.away_team
        rh, ra = ratings[h], ratings[a]

        eff_rh = rh + (0 if row.neutral else HOME_ADV)
        exp_h = 1.0 / (1.0 + 10 ** ((ra - eff_rh) / 400))

        if row.home_score > row.away_score:
            score_h = 1.0
        elif row.home_score < row.away_score:
            score_h = 0.0
        else:
            score_h = 0.5

        k = k_for(row.tournament)
        gd = int(row.home_score - row.away_score)
        mult = goal_diff_multiplier(gd)

        delta = k * mult * (score_h - exp_h)
        ratings[h] = rh + delta
        ratings[a] = ra - delta

        timeline[h].append((row.date, ratings[h]))
        timeline[a].append((row.date, ratings[a]))

    return timeline, dict(ratings)


def rating_as_of(timeline, team, date, start_rating=1500.0):
    """Most recent rating strictly before `date`. Falls back to start_rating."""
    hist = timeline.get(team)
    if not hist:
        return start_rating
    r = start_rating
    for d, val in hist:
        if d < date:
            r = val
        else:
            break
    return r


if __name__ == "__main__":
    df = pd.read_csv("data/results.csv", parse_dates=["date"])
    timeline, final = compute_elo_history(df)
    top20 = sorted(final.items(), key=lambda x: -x[1])[:20]
    for team, r in top20:
        print(f"{team:20s} {r:.1f}")
