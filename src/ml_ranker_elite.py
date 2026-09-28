
from __future__ import annotations
import numpy as np
import pandas as pd

try:
    from sklearn.ensemble import HistGradientBoostingRegressor
except Exception:
    HistGradientBoostingRegressor = None


DEFAULT_FEATURES = [
    "value_score",
    "quality_score",
    "momentum_skip_score",
    "short_reversal_score",
    "low_vol_score",
    "dividend_score",
    "sector_rotation_score",
    "crowding_score",
]


def _prepare_features(df, features):
    X = df.reindex(columns=features).apply(pd.to_numeric, errors="coerce")
    med = X.median()
    X = X.fillna(med).fillna(0.0)
    return X


def fit_walkforward_ranker(
    train: pd.DataFrame,
    features=None,
    target_col="forward_1m_return",
):
    """
    Conservative tree model:
    - no random train/test shuffle
    - no future dates in training
    - shallow-ish model and regularization
    """
    if HistGradientBoostingRegressor is None:
        raise ImportError("scikit-learn is required for ML ranker")

    features = features or DEFAULT_FEATURES
    t = train.dropna(subset=[target_col]).copy()
    raw = t.reindex(columns=features).apply(pd.to_numeric, errors="coerce")
    medians = raw.median().fillna(0.0)
    X = raw.fillna(medians)
    y = pd.to_numeric(t[target_col], errors="coerce").fillna(0.0)

    model = HistGradientBoostingRegressor(
        max_depth=3,
        learning_rate=0.05,
        max_iter=120,
        l2_regularization=1.0,
        min_samples_leaf=30,
        random_state=42,
        early_stopping=False,
    )
    model.fit(X, y)
    model.feature_medians_v8_ = medians
    return model, features


def predict_rank_score(model, features, df):
    if not hasattr(model, "feature_medians_v8_"):
        raise ValueError("模型缺少训练期填补参数，重新训练")
    X = df.reindex(columns=features).apply(pd.to_numeric, errors="coerce").fillna(model.feature_medians_v8_)
    pred = pd.Series(model.predict(X), index=df.index)
    sd = pred.std(ddof=0)
    if sd == 0 or not np.isfinite(sd):
        return pd.Series(0.0, index=df.index)
    return (pred-pred.mean())/sd


def strict_walkforward_ml_scores(
    panel_with_forward_returns: pd.DataFrame,
    train_months=24,
    min_train_rows=300,
    features=None,
):
    """
    Each signal date only trains on dates strictly earlier than the test date.
    """
    x = panel_with_forward_returns.copy()
    x["signal_date"] = pd.to_datetime(x["signal_date"])
    if "forward_end_trade_date" not in x:
        raise ValueError("需要 forward_end_trade_date 以剔除尚未形成的标签")
    x["forward_end_trade_date"] = pd.to_datetime(x["forward_end_trade_date"], errors="raise")
    dates = sorted(x["signal_date"].dropna().unique())
    rows = []

    for i, d in enumerate(dates):
        prior_dates = dates[max(0, i-int(train_months)):i]
        if not prior_dates:
            continue
        train = x[x["signal_date"].isin(prior_dates) & (x["forward_end_trade_date"] < d)].copy()
        test = x[x["signal_date"] == d].copy()

        if len(train.dropna(subset=["forward_1m_return"])) < int(min_train_rows):
            continue

        model, feats = fit_walkforward_ranker(train, features)
        test["ml_score"] = predict_rank_score(model, feats, test)
        rows.append(test)

    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
