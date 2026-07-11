"""
Builds team-level 'knockout mentality' features, computed as cumulative
career-to-date stats strictly BEFORE a given cutoff date (no leakage).

Features per team as-of a date:
  ko_played        - # career WC knockout matches played
  ko_win_rate       - win rate in those (shootout-adjusted)
  ko_avg_margin     - avg (goals_for - goals_against) in WC knockout matches
  shootout_played   - # career WC penalty shootouts
  shootout_win_rate - win rate in those
  comeback_rate     - of WC matches where team conceded the FIRST goal,
                       fraction they still avoided losing (drew/won/won on pens)
"""
import pandas as pd
import numpy as np


def tag_knockout_matches(wc: pd.DataFrame) -> pd.DataFrame:
    """wc: FIFA World Cup rows only. Adds a 'stage' column.
    Uses the fact that 2006-2026 all use: first N matches of a tournament
    (chronologically) are group stage, last 16 are the knockout bracket
    (R16, QF, SF, 3rd, Final) -- true for the 32-team era (2006-2022).
    2026 is a 48-team format (100 matches) with an extra Round of 32; we
    tag its last 15 matches (R32 winners is separate) as knockout by round size below.
    """
    wc = wc.sort_values("date").reset_index(drop=True)
    wc["stage"] = "group"
    for year, grp in wc.groupby(wc["date"].dt.year):
        idx = grp.index
        n = len(idx)
        if n == 64:  # 2006-2022: 48 group + 16 knockout (R16,QF,SF,3rd,F)
            ko_idx = idx[48:]
        elif n >= 90:  # 2026: 48 teams, 12 groups of 4 -> 72 group games,
            # then R32(16) + R16(8) + QF(4) + SF(2) + 3rd(1) + Final(1) = 32 knockout
            ko_idx = idx[72:]
        else:
            ko_idx = idx[-16:] if n > 16 else idx
        wc.loc[ko_idx, "stage"] = "knockout"
    return wc


def match_shootout_winner(wc_ko: pd.DataFrame, shootouts: pd.DataFrame) -> pd.DataFrame:
    """For drawn knockout matches, attach the shootout winner as true_winner."""
    so = shootouts.copy()
    so["date"] = pd.to_datetime(so["date"])
    key_cols = ["date", "home_team", "away_team"]
    wc_ko = wc_ko.merge(so[key_cols + ["winner"]], on=key_cols, how="left")

    def resolve(row):
        if row.home_score > row.away_score:
            return row.home_team
        elif row.away_score > row.home_score:
            return row.away_team
        else:
            return row.winner  # drawn in regulation/ET -> shootout decides

    wc_ko["true_winner"] = wc_ko.apply(resolve, axis=1)
    return wc_ko


def first_goal_scorer_per_match(goalscorers: pd.DataFrame) -> pd.DataFrame:
    gs = goalscorers.dropna(subset=["minute"]).copy()
    gs["date"] = pd.to_datetime(gs["date"])
    gs = gs.sort_values(["date", "home_team", "away_team", "minute"])
    first = gs.groupby(["date", "home_team", "away_team"], as_index=False).first()
    return first[["date", "home_team", "away_team", "team"]].rename(
        columns={"team": "first_scorer"}
    )


class MentalityTracker:
    """Chronologically walk WC knockout matches, exposing pre-match career
    stats for each team at every step (so features never see the future)."""

    def __init__(self, wc_ko_resolved: pd.DataFrame, first_goal: pd.DataFrame):
        self.matches = wc_ko_resolved.sort_values("date").reset_index(drop=True)
        self.first_goal = first_goal.set_index(["date", "home_team", "away_team"])

        self.ko_played = {}
        self.ko_wins = {}
        self.ko_margin_sum = {}
        self.so_played = {}
        self.so_wins = {}
        self.conceded_first = {}
        self.avoided_loss_after_conceding_first = {}

    def _get(self, d, team):
        return dict(
            ko_played=self.ko_played.get(team, 0),
            ko_win_rate=(self.ko_wins.get(team, 0) / self.ko_played[team])
            if self.ko_played.get(team, 0) > 0
            else np.nan,
            ko_avg_margin=(self.ko_margin_sum.get(team, 0) / self.ko_played[team])
            if self.ko_played.get(team, 0) > 0
            else np.nan,
            shootout_played=self.so_played.get(team, 0),
            shootout_win_rate=(self.so_wins.get(team, 0) / self.so_played[team])
            if self.so_played.get(team, 0) > 0
            else np.nan,
            comeback_rate=(
                self.avoided_loss_after_conceding_first.get(team, 0)
                / self.conceded_first[team]
            )
            if self.conceded_first.get(team, 0) > 0
            else np.nan,
        )

    def features_before(self, date, home, away):
        return self._get(date, home), self._get(date, away)

    def _update(self, row):
        h, a = row.home_team, row.away_team
        winner = row.true_winner
        margin = row.home_score - row.away_score  # regulation/ET margin (0 if pens)

        for team, opp_margin in [(h, margin), (a, -margin)]:
            self.ko_played[team] = self.ko_played.get(team, 0) + 1
            self.ko_margin_sum[team] = self.ko_margin_sum.get(team, 0) + opp_margin
            if winner == team:
                self.ko_wins[team] = self.ko_wins.get(team, 0) + 1

        if pd.notna(getattr(row, "winner", np.nan)):  # a shootout occurred
            self.so_played[h] = self.so_played.get(h, 0) + 1
            self.so_played[a] = self.so_played.get(a, 0) + 1
            if row.winner == h:
                self.so_wins[h] = self.so_wins.get(h, 0) + 1
            elif row.winner == a:
                self.so_wins[a] = self.so_wins.get(a, 0) + 1

        key = (row.date, h, a)
        first_scorer = None
        if key in self.first_goal.index:
            first_scorer = self.first_goal.loc[key, "first_scorer"]
        if first_scorer in (h, a):
            trailer = a if first_scorer == h else h
            self.conceded_first[trailer] = self.conceded_first.get(trailer, 0) + 1
            if winner == trailer or winner is None:
                self.avoided_loss_after_conceding_first[trailer] = (
                    self.avoided_loss_after_conceding_first.get(trailer, 0) + 1
                )

    def run(self):
        """Yields (row, feats_home_before, feats_away_before) then updates state."""
        for row in self.matches.itertuples(index=False):
            fh, fa = self.features_before(row.date, row.home_team, row.away_team)
            yield row, fh, fa
            self._update(row)
