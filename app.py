"""
World Cup 2026 Knockout Forecaster -- Streamlit dashboard

Run with:
    streamlit run app.py

Expects to live in the same folder as elo.py, mentality.py, build_dataset.py,
and train_backtest.py (i.e. drop this file straight into wc_repo/).
"""
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib as mpl

from elo import compute_elo_history, rating_as_of
from mentality import tag_knockout_matches, match_shootout_winner, first_goal_scorer_per_match, MentalityTracker
from build_dataset import load_raw, build_all, FEATURE_COLS
from train_backtest import fit_final_models, leave_one_tournament_out, YEARS

st.set_page_config(page_title="World Cup 2026 Forecaster", layout="wide",)

NOW = pd.Timestamp("2026-12-31")
QF_TEAMS = ["France", "Spain", "Argentina", "England", "Norway", "Switzerland"]

MARKET_ADVANCE = {
    ("Norway", "England"): (0.248 + 0.255 / 2, 0.497 + 0.255 / 2),
    ("Argentina", "Switzerland"): (0.569 + 0.266 / 2, 0.165 + 0.266 / 2),
    ("France", "Spain"): (0.403 + 0.29 / 2, 0.307 + 0.29 / 2),
}

# ---------------------------------------------------------------------------
# Theme: matte dark, minimal. One neutral surface, one accent (bone white)
# for whichever team is favored, one muted signal color reserved only for
# flagging model-vs-market disagreement. No decorative color otherwise.
# ---------------------------------------------------------------------------
BG = "#0A0A0A"
SURFACE = "#141414"
BORDER = "#232323"
INK = "#EDEDEC"
INK_DIM = "#6E6E6C"
INK_FAINT = "#48484A"
ACCENT = "#EDEDEC"       # favored team highlight -- same tone as ink, weight does the work
SIGNAL = "#C97A6D"       # muted terracotta-red, ONLY for model/market divergence flag

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

.stApp {{
    background: {BG};
    color: {INK};
}}
h1, h2, h3 {{
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
    letter-spacing: -0.01em;
    color: {INK} !important;
}}
p, div, span, label, li {{
    font-family: 'Inter', sans-serif;
}}
.mono {{
    font-family: 'IBM Plex Mono', monospace;
}}
[data-testid="stSidebar"] {{
    background: {SURFACE};
    border-right: 1px solid {BORDER};
}}
[data-testid="stMetric"] {{
    background: {SURFACE};
    border: 1px solid {BORDER};
    border-radius: 6px;
    padding: 12px 16px;
}}
.scorecard {{
    background: {SURFACE};
    border: 1px solid {BORDER};
    border-radius: 6px;
    padding: 20px 22px;
    margin-bottom: 12px;
}}
.scorecard .elo-line {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    color: {INK_FAINT};
    text-transform: uppercase;
    letter-spacing: 0.04em;
    margin-bottom: 14px;
}}
.scorecard .team-row {{
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    padding: 6px 0;
    border-bottom: 1px solid {BORDER};
}}
.scorecard .team-row:last-of-type {{ border-bottom: none; }}
.scorecard .team-name {{
    font-size: 0.95rem;
    font-weight: 500;
}}
.scorecard .team-name.favored {{ color: {INK}; font-weight: 700; }}
.scorecard .team-name.underdog {{ color: {INK_DIM}; }}
.scorecard .team-pct {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.4rem;
}}
.scorecard .team-pct.favored {{ color: {ACCENT}; font-weight: 600; }}
.scorecard .team-pct.underdog {{ color: {INK_FAINT}; font-weight: 400; }}
.scorecard .caption {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    color: {INK_FAINT};
    margin-top: 10px;
}}
.divergence-flag {{
    display: inline-block;
    background: rgba(201, 122, 109, 0.1);
    border: 1px solid {SIGNAL};
    color: {SIGNAL};
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.68rem;
    padding: 3px 9px;
    border-radius: 3px;
    margin-top: 10px;
    letter-spacing: 0.02em;
}}
hr {{ border-color: {BORDER} !important; }}
.stTabs [data-baseweb="tab-list"] {{
    gap: 28px;
    border-bottom: 1px solid {BORDER};
    padding-bottom: 0;
}}
.stTabs [data-baseweb="tab"] {{
    font-family: 'Inter', sans-serif;
    font-size: 0.88rem;
    font-weight: 500;
    color: {INK_DIM};
    padding: 10px 2px;
    letter-spacing: 0.01em;
    transition: color 0.15s ease;
}}
.stTabs [data-baseweb="tab"]:hover {{
    color: {INK};
}}
.stTabs [aria-selected="true"] {{
    color: {INK} !important;
    font-weight: 600;
}}
.stTabs [data-baseweb="tab-highlight"] {{
    background-color: {ACCENT} !important;
    height: 2px !important;
}}
.stTabs [data-baseweb="tab-border"] {{
    display: none;
}}
</style>
""", unsafe_allow_html=True)

mpl.rcParams.update({
    "figure.facecolor": BG,
    "axes.facecolor": BG,
    "axes.edgecolor": BORDER,
    "axes.labelcolor": INK_DIM,
    "xtick.color": INK_DIM,
    "ytick.color": INK_DIM,
    "text.color": INK,
    "grid.color": BORDER,
    "font.family": "monospace",
})


# ---------------------------------------------------------------------------
# Cached data / model layer
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner="Downloading match history (1872-2026)...")
def get_raw_data():
    return load_raw()


@st.cache_resource(show_spinner="Training model on 2006-2022 knockout matches...")
def get_trained_models(_df, _so, _gs):
    train_data, _, _ = build_all(_df, _so, _gs, years=YEARS)
    logit, scaler, gbm = fit_final_models(train_data)
    backtest = leave_one_tournament_out(train_data)
    return logit, scaler, gbm, backtest


@st.cache_resource(show_spinner="Computing live Elo ratings and mentality features...")
def get_live_state(_df, _so, _gs):
    elo_timeline, _ = compute_elo_history(_df)
    wc = tag_knockout_matches(_df[_df["tournament"] == "FIFA World Cup"].copy())
    wc_ko = wc[wc["stage"] == "knockout"].dropna(subset=["home_score", "away_score"])
    wc_ko = match_shootout_winner(wc_ko, _so)
    first_goal = first_goal_scorer_per_match(_gs)
    tracker = MentalityTracker(wc_ko, first_goal)
    for _ in tracker.run():
        pass
    return elo_timeline, tracker, wc_ko


def team_feature_vector(elo_timeline, tracker, team_a, team_b, neutral=1, at=NOW):
    elo_a = rating_as_of(elo_timeline, team_a, at)
    elo_b = rating_as_of(elo_timeline, team_b, at)
    fa, fb = tracker._get(at, team_a), tracker._get(at, team_b)

    def d(key):
        va = fa[key] if pd.notna(fa[key]) else 0.0
        vb = fb[key] if pd.notna(fb[key]) else 0.0
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
    return np.array([vec[c] for c in FEATURE_COLS]), elo_a, elo_b


def predict(logit, scaler, gbm, x):
    x = x.reshape(1, -1)
    p_logit = logit.predict_proba(scaler.transform(x))[0, 1]
    p_gbm = gbm.predict_proba(x)[0, 1]
    return p_logit, p_gbm, (p_logit + p_gbm) / 2


def scorecard(team_a, team_b, p_a, market_a=None, elo_a=None, elo_b=None):
    p_b = 1 - p_a
    a_favored = p_a >= p_b
    a_cls, b_cls = ("favored", "underdog") if a_favored else ("underdog", "favored")

    flag = ""
    if market_a is not None and abs(p_a - market_a) > 0.15:
        flag = f'<div class="divergence-flag">MODEL ({p_a:.0%}) VS MARKET ({market_a:.0%}) — {abs(p_a-market_a)*100:.0f}PT GAP ON {team_a.upper()}</div>'

    elo_line = f'ELO &nbsp; {team_a} {elo_a:.0f} &nbsp;·&nbsp; {team_b} {elo_b:.0f}' if elo_a is not None else ""
    market_line = f'<div class="caption">market-implied: {team_a} {market_a:.0%} / {team_b} {1-market_a:.0%}</div>' if market_a is not None else ""

    st.markdown(f"""
    <div class="scorecard">
        <div class="elo-line">{elo_line}</div>
        <div class="team-row">
            <span class="team-name {a_cls}">{team_a}</span>
            <span class="team-pct {a_cls}">{p_a:.0%}</span>
        </div>
        <div class="team-row">
            <span class="team-name {b_cls}">{team_b}</span>
            <span class="team-pct {b_cls}">{p_b:.0%}</span>
        </div>
        {market_line}
        {flag}
    </div>
    """, unsafe_allow_html=True)


def run_monte_carlo(p_nor, p_arg, p_fra_sf1, sf2_probs, final_probs, n=50000, seed=42):
    """Returns (championship %, semifinal-reach %, final-reach %) per team."""
    rng = np.random.default_rng(seed)
    titles = {t: 0 for t in QF_TEAMS}
    finalists = {t: 0 for t in QF_TEAMS}
    semifinalists = {t: 0 for t in QF_TEAMS}
    for _ in range(n):
        qf1 = "Norway" if rng.random() < p_nor else "England"
        qf2 = "Argentina" if rng.random() < p_arg else "Switzerland"
        semifinalists["France"] += 1
        semifinalists["Spain"] += 1
        semifinalists[qf1] += 1
        semifinalists[qf2] += 1

        sf1_w = "France" if rng.random() < p_fra_sf1 else "Spain"
        sf2_w = qf1 if rng.random() < sf2_probs[(qf1, qf2)] else qf2
        finalists[sf1_w] += 1
        finalists[sf2_w] += 1

        champion = sf1_w if rng.random() < final_probs[(sf1_w, sf2_w)] else sf2_w
        titles[champion] += 1

    to_pct = lambda d: (pd.Series(d) / n * 100)
    return to_pct(titles).sort_values(ascending=False), to_pct(finalists), to_pct(semifinalists)


# ---------------------------------------------------------------------------
# Load everything once
# ---------------------------------------------------------------------------

df, so, gs = get_raw_data()
logit, scaler, gbm, backtest = get_trained_models(df, so, gs)
elo_timeline, tracker, wc_ko_all = get_live_state(df, so, gs)

all_teams = sorted(set(df["home_team"]).union(df["away_team"]))

st.title("🏆 World Cup 2026 Knockout Forecaster")
st.caption(
    "Elo (1872-2026) + career knockout-mentality features, trained on 2006-2022, "
    "applied to the live bracket. Model blended with market odds where noted."
)

col_refresh, _ = st.columns([1, 5])
with col_refresh:
    if st.button("🔄 Refresh data"):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()

tab_bracket, tab_team, tab_h2h, tab_backtest = st.tabs(
    ["Bracket & Odds", "Team Explorer", "Head-to-Head", "Model Validation"]
)

# =============================================================================
# TAB 1: Bracket & Odds
# =============================================================================
with tab_bracket:
    left, right = st.columns([3, 2])

    with right:
        st.subheader("Pending Quarterfinals")
        st.caption("Slider defaults to the model+market blend. Drag to explore what-ifs.")

        qf_probs = {}
        qf_display = {}
        for a, b in [("Norway", "England"), ("Argentina", "Switzerland")]:
            x, elo_a, elo_b = team_feature_vector(elo_timeline, tracker, a, b)
            p_logit, p_gbm, p_ens = predict(logit, scaler, gbm, x)
            m_a, _ = MARKET_ADVANCE[(a, b)]
            default_blend = (p_ens + m_a) / 2
            qf_display[(a, b)] = (p_ens, m_a, elo_a, elo_b)

            p_user = st.slider(f"{a} win probability", 0.0, 1.0, float(round(default_blend, 2)),
                                key=f"slider_{a}_{b}")
            qf_probs[(a, b)] = p_user

        st.subheader("Semifinal 1 (locked)")
        x, elo_a, elo_b = team_feature_vector(elo_timeline, tracker, "France", "Spain")
        p_logit, p_gbm, p_ens = predict(logit, scaler, gbm, x)
        m_a, _ = MARKET_ADVANCE[("France", "Spain")]
        sf1_default = (p_ens + m_a) / 2
        p_fra_sf1 = st.slider("France win probability (SF1)", 0.0, 1.0, float(round(sf1_default, 2)))
        qf_display[("France", "Spain")] = (p_ens, m_a, elo_a, elo_b)

    sf2_candidates = [("Norway", "Argentina"), ("Norway", "Switzerland"),
                       ("England", "Argentina"), ("England", "Switzerland")]
    sf2_probs = {}
    for a, b in sf2_candidates:
        x, *_ = team_feature_vector(elo_timeline, tracker, a, b)
        sf2_probs[(a, b)] = predict(logit, scaler, gbm, x)[2]

    final_candidates = [(f, s) for f in ["France", "Spain"] for s in QF_TEAMS if s not in ("France", "Spain")]
    final_probs = {}
    for a, b in final_candidates:
        x, *_ = team_feature_vector(elo_timeline, tracker, a, b)
        final_probs[(a, b)] = predict(logit, scaler, gbm, x)[2]

    champ_odds, final_odds, semi_odds = run_monte_carlo(
        qf_probs[("Norway", "England")], qf_probs[("Argentina", "Switzerland")],
        p_fra_sf1, sf2_probs, final_probs,
    )

    with left:
        st.subheader("Round-by-round odds")
        roundtable = pd.DataFrame({
            "Reach Semifinal %": semi_odds.round(1),
            "Reach Final %": final_odds.round(1),
            "Win Championship %": champ_odds.round(1),
        }).loc[champ_odds.index]
        st.dataframe(roundtable, use_container_width=True)

        fig, ax = plt.subplots(figsize=(7, 3.6))
        colors = [ACCENT if i == 0 else INK_FAINT for i in range(len(champ_odds))]
        ax.bar(champ_odds.index, champ_odds.values, color=colors, width=0.55)
        ax.set_ylabel("Championship probability (%)", fontsize=9, family="sans-serif")
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(left=False)
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color(BORDER)
        for i, v in enumerate(champ_odds.values):
            ax.text(i, v + 0.8, f"{v:.1f}%", ha="center", color=INK, family="monospace", fontsize=10)
        plt.xticks(family="sans-serif", fontsize=9)
        st.pyplot(fig)

        st.subheader("Matchup scorecards")
        shown_probs = {**qf_probs, ("France", "Spain"): p_fra_sf1}
        for (a, b), (p_ens, m_a, ea, eb) in qf_display.items():
            scorecard(a, b, shown_probs[(a, b)], market_a=m_a, elo_a=ea, elo_b=eb)

# =============================================================================
# TAB 2: Team Explorer -- pick ANY team, see full history
# =============================================================================
with tab_team:
    st.subheader("Explore any national team's history")
    default_idx = all_teams.index("Argentina") if "Argentina" in all_teams else 0
    team = st.selectbox("Team", all_teams, index=default_idx)

    elo_hist = elo_timeline.get(team, [])
    if elo_hist:
        hist_df = pd.DataFrame(elo_hist, columns=["date", "elo"])
        current_elo = hist_df["elo"].iloc[-1]
    else:
        hist_df = pd.DataFrame(columns=["date", "elo"])
        current_elo = 1500.0

    snap = tracker._get(NOW, team)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Current Elo", f"{current_elo:.0f}")
    c2.metric("WC knockout matches", int(snap["ko_played"]))
    c3.metric("Knockout win rate", f"{snap['ko_win_rate']:.0%}" if pd.notna(snap["ko_win_rate"]) else "—")
    c4.metric("Shootout record", f"{snap['shootout_win_rate']:.0%}" if pd.notna(snap["shootout_win_rate"]) else "no shootouts")

    st.markdown(f"**Comeback rate** (avoided loss after conceding first, WC knockouts): "
                f"{snap['comeback_rate']:.0%}" if pd.notna(snap["comeback_rate"]) else "**Comeback rate:** n/a")

    if not hist_df.empty:
        st.subheader(f"{team} — Elo rating over time (1872–2026)")
        fig, ax = plt.subplots(figsize=(9, 3.0))
        ax.plot(hist_df["date"], hist_df["elo"], color=ACCENT, linewidth=1.1)
        ax.axhline(1500, color=BORDER, linewidth=0.8, linestyle="--")
        ax.spines[["top", "right"]].set_visible(False)
        for spine in ax.spines.values():
            spine.set_color(BORDER)
        ax.set_ylabel("Elo", fontsize=9)
        st.pyplot(fig)
    else:
        st.info("No match history found for this team in the dataset.")

    st.subheader(f"{team} — World Cup knockout match log")
    team_ko = wc_ko_all[(wc_ko_all["home_team"] == team) | (wc_ko_all["away_team"] == team)].copy()
    if not team_ko.empty:
        team_ko["opponent"] = np.where(team_ko["home_team"] == team, team_ko["away_team"], team_ko["home_team"])
        team_ko["result"] = np.where(team_ko["true_winner"] == team, "W",
                                      np.where(team_ko["true_winner"].isna(), "?", "L"))
        team_ko["score"] = team_ko["home_score"].astype(int).astype(str) + "-" + team_ko["away_score"].astype(int).astype(str)
        display_cols = team_ko[["date", "opponent", "score", "result"]].sort_values("date", ascending=False)
        display_cols["date"] = display_cols["date"].dt.strftime("%Y-%m-%d")
        st.dataframe(display_cols, use_container_width=True, hide_index=True)
    else:
        st.info(f"{team} has no World Cup knockout-stage appearances in this dataset (2006–2026 knockout rounds only).")

# =============================================================================
# TAB 3: Head-to-Head -- pick any two teams
# =============================================================================
with tab_h2h:
    st.subheader("Head-to-head predictor")
    st.caption("Model's win probability for any two teams, using each team's Elo and career "
               "knockout-mentality features as of today. Not limited to the live bracket.")

    hc1, hc2, hc3 = st.columns([2, 2, 1])
    team_a = hc1.selectbox("Team A", all_teams, index=all_teams.index("Brazil") if "Brazil" in all_teams else 0)
    team_b = hc2.selectbox("Team B", all_teams, index=all_teams.index("Germany") if "Germany" in all_teams else 1)
    neutral = hc3.toggle("Neutral venue", value=True)

    if team_a == team_b:
        st.warning("Pick two different teams.")
    else:
        x, elo_a, elo_b = team_feature_vector(elo_timeline, tracker, team_a, team_b, neutral=1 if neutral else 0)
        p_logit, p_gbm, p_ens = predict(logit, scaler, gbm, x)

        scorecard(team_a, team_b, p_ens, elo_a=elo_a, elo_b=elo_b)

        d1, d2, d3 = st.columns(3)
        d1.metric("Logistic regression", f"{p_logit:.0%}")
        d2.metric("Gradient boosting", f"{p_gbm:.0%}")
        d3.metric("Ensemble", f"{p_ens:.0%}")

        st.caption(
            "This uses the same model trained on 2006–2022 World Cup knockout matches. "
            "For teams with little/no WC knockout history, mentality features default to 0 "
            "(neutral prior) and the prediction leans almost entirely on Elo."
        )

# =============================================================================
# TAB 4: Model Validation
# =============================================================================
with tab_backtest:
    st.subheader("Leave-one-tournament-out backtest (2006-2022, 80 matches)")
    agg = backtest.groupby("model")[["accuracy", "log_loss", "brier"]].mean().round(3)
    st.dataframe(agg, use_container_width=True)
    st.caption(
        "All three models pick the right team about as often, but the mentality-augmented "
        "logit/gbm are much better calibrated (lower log-loss/Brier) than the Elo-only baseline. "
        "Small sample (80 matches) — treat as directionally useful, not precise."
    )
    with st.expander("Per-tournament detail"):
        st.dataframe(backtest, use_container_width=True)