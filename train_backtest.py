import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss

from build_dataset import load_raw, build_all, FEATURE_COLS

YEARS = [2006, 2010, 2014, 2018, 2022]


def naive_baseline_pred(row):
    """Always back the team with the better career knockout win rate + Elo edge;
    if pure Elo favorite, that's the pick. Returns predicted P(home wins) = 1/0."""
    return 1.0 if row["elo_diff"] > 0 else 0.0


def leave_one_tournament_out(data):
    results = []
    for test_year in YEARS:
        train = data[data["year"] != test_year]
        test = data[data["year"] == test_year]

        X_train, y_train = train[FEATURE_COLS].values, train["label"].values
        X_test, y_test = test[FEATURE_COLS].values, test["label"].values

        scaler = StandardScaler().fit(X_train)
        X_train_s, X_test_s = scaler.transform(X_train), scaler.transform(X_test)

        logit = LogisticRegression(max_iter=2000, C=1.0)
        logit.fit(X_train_s, y_train)
        p_logit = logit.predict_proba(X_test_s)[:, 1]

        gbm = HistGradientBoostingClassifier(
            max_depth=3, max_iter=100, learning_rate=0.05, random_state=0
        )
        gbm.fit(X_train, y_train)
        p_gbm = gbm.predict_proba(X_test)[:, 1]

        p_baseline = test.apply(naive_baseline_pred, axis=1).values
        p_baseline_soft = np.clip(p_baseline, 0.05, 0.95)  # avoid log(0)

        for name, p in [("logit", p_logit), ("gbm", p_gbm), ("elo_only_baseline", p_baseline_soft)]:
            pred = (p > 0.5).astype(int)
            results.append(
                dict(
                    test_year=test_year,
                    model=name,
                    n=len(y_test),
                    accuracy=accuracy_score(y_test, pred),
                    log_loss=log_loss(y_test, p, labels=[0, 1]),
                    brier=brier_score_loss(y_test, p),
                )
            )
    return pd.DataFrame(results)


def fit_final_models(data):
    """Fit on ALL 2006-2022 data (for forward use on the live 2026 bracket)."""
    X, y = data[FEATURE_COLS].values, data["label"].values
    scaler = StandardScaler().fit(X)
    logit = LogisticRegression(max_iter=2000, C=1.0).fit(scaler.transform(X), y)
    gbm = HistGradientBoostingClassifier(
        max_depth=3, max_iter=100, learning_rate=0.05, random_state=0
    ).fit(X, y)
    return logit, scaler, gbm


if __name__ == "__main__":
    df, so, gs = load_raw()
    data, elo_timeline, tracker = build_all(df, so, gs, years=YEARS)

    res = leave_one_tournament_out(data)
    pd.set_option("display.width", 120)
    print(res.to_string(index=False))
    print("\n--- Aggregated across all 5 held-out tournaments ---")
    print(res.groupby("model")[["accuracy", "log_loss", "brier"]].mean())

    # logistic coefficients for interpretability (fit on everything)
    X, y = data[FEATURE_COLS].values, data["label"].values
    scaler = StandardScaler().fit(X)
    logit_full = LogisticRegression(max_iter=2000).fit(scaler.transform(X), y)
    print("\n--- Logistic regression coefficients (standardized) ---")
    for f, c in zip(FEATURE_COLS, logit_full.coef_[0]):
        print(f"{f:28s} {c:+.3f}")
