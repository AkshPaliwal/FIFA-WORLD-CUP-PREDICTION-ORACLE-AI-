import numpy as np
import pandas as pd

from elo import compute_elo_history, rating_as_of
from mentality import (
    tag_knockout_matches,
    match_shootout_winner,
    first_goal_scorer_per_match,
    MentalityTracker,
)
from build_dataset import load_raw, build_all, FEATURE_COLS
from train_backtest import fit_final_models, YEARS

NOW = pd.Timestamp("2026-12-31")  # "after everything played so far"

# Market-implied advance probabilities from live odds (draw split 50/50),
# captured July 11 2026, for cross-checking our model.
MARKET_ADVANCE = {
    ("Norway", "England"): (0.248 + 0.255 / 2, 0.497 + 0.255 / 2),
    ("Argentina", "Switzerland"): (0.569 + 0.266 / 2, 0.165 + 0.266 / 2),
    ("France", "Spain"): (0.403 + 0.29 / 2, 0.307 + 0.29 / 2),
}


def team_feature_vector(elo_timeline, tracker, team_a, team_b, neutral=1):
    """elo_diff / mentality diffs of team_a relative to team_b, evaluated now."""
    elo_a = rating_as_of(elo_timeline, team_a, NOW)
    elo_b = rating_as_of(elo_timeline, team_b, NOW)
    fa = tracker._get(NOW, team_a)
    fb = tracker._get(NOW, team_b)

    def d(key):
        va, vb = fa[key], fb[key]
        va = 0.0 if pd.isna(va) else va
        vb = 0.0 if pd.isna(vb) else vb
        return va - vb

    vec = dict(
        elo_diff=elo_a - elo_b,
        ko_played_diff=d("ko_played"),
        ko_win_rate_diff=d("ko_win_rate"),
        ko_avg_margin_diff=d("ko_avg_margin"),
        shootout_win_rate_diff=d("shootout_win_rate"),
        comeback_rate_diff=d("comeback_rate"),
        neutral=neutral,
    )
    return np.array([vec[c] for c in FEATURE_COLS]), elo_a, elo_b, fa, fb


def predict(logit, scaler, gbm, x):
    x = x.reshape(1, -1)
    p_logit = logit.predict_proba(scaler.transform(x))[0, 1]
    p_gbm = gbm.predict_proba(x)[0, 1]
    return p_logit, p_gbm, (p_logit + p_gbm) / 2


def main():
    df, so, gs = load_raw()
    train_data, elo_timeline_train, _ = build_all(df, so, gs, years=YEARS)
    logit, scaler, gbm = fit_final_models(train_data)

    # rebuild elo + mentality tracker over EVERYTHING played so far, incl. 2026
    elo_timeline, _ = compute_elo_history(df)
    wc = df[df["tournament"] == "FIFA World Cup"].copy()
    wc = tag_knockout_matches(wc)
    wc_ko_all = wc[wc["stage"] == "knockout"].copy()
    wc_ko_all = wc_ko_all.dropna(subset=["home_score", "away_score"])  # drop pending fixtures
    wc_ko_all = match_shootout_winner(wc_ko_all, so)
    first_goal = first_goal_scorer_per_match(gs)
    tracker = MentalityTracker(wc_ko_all, first_goal)
    for _ in tracker.run():
        pass  # drive the generator to walk through and update all state

    print("=== Current Elo (top context) ===")
    for team in ["France", "Spain", "Argentina", "England", "Norway", "Switzerland"]:
        print(f"{team:15s} {rating_as_of(elo_timeline, team, NOW):.1f}")

    print("\n=== Career WC knockout mentality snapshot ===")
    cols = ["ko_played", "ko_win_rate", "shootout_played", "shootout_win_rate", "comeback_rate"]
    for team in ["France", "Spain", "Argentina", "England", "Norway", "Switzerland"]:
        f = tracker._get(NOW, team)
        print(team, {k: (round(f[k], 2) if pd.notna(f[k]) else None) for k in cols})

    pending_qf = [("Norway", "England"), ("Argentina", "Switzerland")]
    print("\n=== Pending Quarterfinals: model vs market ===")
    qf_probs = {}
    for a, b in pending_qf:
        x, elo_a, elo_b, fa, fb = team_feature_vector(elo_timeline, tracker, a, b)
        p_logit, p_gbm, p_ens = predict(logit, scaler, gbm, x)
        m_a, m_b = MARKET_ADVANCE[(a, b)]
        print(
            f"{a} vs {b}: elo {elo_a:.0f}-{elo_b:.0f} | "
            f"model(logit={p_logit:.2f}, gbm={p_gbm:.2f}, ensemble={p_ens:.2f}) | "
            f"market(advance {a}={m_a:.2f}, {b}={m_b:.2f})"
        )
        qf_probs[(a, b)] = p_ens

    print("\n=== Semifinal 1 (locked): France vs Spain ===")
    x, elo_a, elo_b, fa, fb = team_feature_vector(elo_timeline, tracker, "France", "Spain")
    p_logit, p_gbm, p_ens = predict(logit, scaler, gbm, x)
    m_a, m_b = MARKET_ADVANCE[("France", "Spain")]
    print(
        f"France vs Spain: elo {elo_a:.0f}-{elo_b:.0f} | "
        f"model(logit={p_logit:.2f}, gbm={p_gbm:.2f}, ensemble={p_ens:.2f}) | "
        f"market(France={m_a:.2f}, Spain={m_b:.2f})"
    )
    sf1_prob_france = (p_ens + m_a) / 2  # blend model + market for the sim

    # Precompute win-prob for every possible SF2 pairing and every possible Final pairing
    sf2_candidates = [("Norway", "Argentina"), ("Norway", "Switzerland"),
                       ("England", "Argentina"), ("England", "Switzerland")]
    sf2_probs = {}
    for a, b in sf2_candidates:
        x, *_ = team_feature_vector(elo_timeline, tracker, a, b)
        _, _, p_ens_ab = predict(logit, scaler, gbm, x)
        sf2_probs[(a, b)] = p_ens_ab

    final_candidates = [(f, s) for f in ["France", "Spain"]
                         for s in ["Norway", "England", "Argentina", "Switzerland"]]
    final_probs = {}
    for a, b in final_candidates:
        x, *_ = team_feature_vector(elo_timeline, tracker, a, b)
        _, _, p_ens_ab = predict(logit, scaler, gbm, x)
        final_probs[(a, b)] = p_ens_ab

    # ---- Monte Carlo ----
    rng = np.random.default_rng(42)
    N = 50000
    titles = {t: 0 for t in ["France", "Spain", "Norway", "England", "Argentina", "Switzerland"]}
    # blend model+market for the two live QFs
    p_nor = (qf_probs[("Norway", "England")] + MARKET_ADVANCE[("Norway", "England")][0]) / 2
    p_arg = (qf_probs[("Argentina", "Switzerland")] + MARKET_ADVANCE[("Argentina", "Switzerland")][0]) / 2

    for _ in range(N):
        qf1_winner = "Norway" if rng.random() < p_nor else "England"
        qf2_winner = "Argentina" if rng.random() < p_arg else "Switzerland"

        sf1_winner = "France" if rng.random() < sf1_prob_france else "Spain"

        key = (qf1_winner, qf2_winner)
        p_sf2 = sf2_probs[key]
        sf2_winner = qf1_winner if rng.random() < p_sf2 else qf2_winner

        fkey = (sf1_winner, sf2_winner)
        p_final = final_probs[fkey]
        champion = sf1_winner if rng.random() < p_final else sf2_winner
        titles[champion] += 1

    print(f"\n=== Monte Carlo championship odds (N={N}) ===")
    for team, count in sorted(titles.items(), key=lambda x: -x[1]):
        print(f"{team:15s} {100*count/N:5.1f}%")


if __name__ == "__main__":
    main()
