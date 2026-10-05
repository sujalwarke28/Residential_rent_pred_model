"""Metrics, cross-validation reporting and diagnostic tables.

Every metric in this module is expressed in RUPEES, not in log units. Models are
trained on ``log1p(Rent)``, but because the estimator is wrapped in a
``TransformedTargetRegressor`` its ``predict`` already applies ``expm1``, so both
``y_true`` and ``y_pred`` arrive here on the rupee scale and ordinary scikit-learn
scorers are directly interpretable.

Reporting log-space errors instead would be much flattering and meaningless: an
RMSE of "0.34 log-rupees" tells a landlord nothing.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    root_mean_squared_error,
)
from sklearn.model_selection import KFold, cross_validate

from src import config as cfg

# Scorers used for every cross_validate call. Negated variants are sklearn's
# convention (higher is always better); `cv_report` flips the signs back.
CV_SCORING = {
    "r2": "r2",
    "mse": "neg_mean_squared_error",
    "rmse": "neg_root_mean_squared_error",
    "mae": "neg_mean_absolute_error",
}

METRIC_KEYS = ["r2", "mse", "rmse", "mae"]


def cv_splitter(n_splits: int = cfg.N_SPLITS) -> KFold:
    """The one shared CV splitter, so every model is scored on identical folds."""
    return KFold(n_splits=n_splits, shuffle=True, random_state=cfg.RANDOM_STATE)


# --------------------------------------------------------------------------- #
# Point metrics
# --------------------------------------------------------------------------- #
def compute_metrics(y_true, y_pred) -> dict[str, float]:
    """R2, MSE, RMSE, MAE and MAPE, all on the rupee scale."""
    y_true = np.asarray(y_true, dtype="float64")
    y_pred = np.asarray(y_pred, dtype="float64")

    # A model trained on raw rent (outlier-experiment setting 1) can emit
    # negative rents, which are nonsensical; clip before scoring so the
    # comparison is not distorted by impossible predictions.
    y_pred = np.clip(y_pred, 0.0, None)

    return {
        "r2": float(r2_score(y_true, y_pred)),
        "mse": float(mean_squared_error(y_true, y_pred)),
        "rmse": float(root_mean_squared_error(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        # Not requested by the spec, but the most intuitive answer to "how well
        # does it do on a new listing?" -- a 20% error on a Rs 15,000 flat and on
        # a Rs 3,00,000 penthouse are very different in rupees.
        "mape": float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100.0),
    }


# --------------------------------------------------------------------------- #
# Cross-validation
# --------------------------------------------------------------------------- #
def cv_report(estimator, X, y, n_splits: int = cfg.N_SPLITS,
              n_jobs: int = -1) -> dict[str, float]:
    """5-fold CV, returning mean and std of every metric across the folds.

    The estimator passed in must be the FULL pipeline (feature engineering,
    ColumnTransformer including the TargetEncoder, and the model). Cloning
    happens per fold, so every fitted quantity -- group medians, one-hot
    categories, target encodings -- is learned from that fold's training rows
    only. This is what keeps the CV scores honest.
    """
    res = cross_validate(
        estimator, X, y,
        cv=cv_splitter(n_splits),
        scoring=CV_SCORING,
        n_jobs=n_jobs,
        return_train_score=True,
        error_score="raise",
    )

    out: dict[str, float] = {}
    for key in METRIC_KEYS:
        test = res[f"test_{key}"]
        train = res[f"train_{key}"]
        if key != "r2":                      # undo sklearn's negation
            test, train = -test, -train
        out[f"cv_{key}_mean"] = float(np.mean(test))
        out[f"cv_{key}_std"] = float(np.std(test))
        out[f"cv_train_{key}_mean"] = float(np.mean(train))
    out["cv_fit_time_mean"] = float(np.mean(res["fit_time"]))
    return out


def format_mean_std(mean: float, std: float, decimals: int = 0) -> str:
    """Render a CV result as `mean ± std` for the comparison tables."""
    return f"{mean:,.{decimals}f} ± {std:,.{decimals}f}"


# --------------------------------------------------------------------------- #
# Section 13 -- per-city breakdown
# --------------------------------------------------------------------------- #
def per_city_metrics(y_true, y_pred, cities: pd.Series) -> pd.DataFrame:
    """RMSE / MAE / R2 / MAPE and rent statistics for each city separately.

    Answers "limitations of rent estimation across different cities": a city with
    few listings, high rent variance or very diverse localities is predicted
    worse, and this table is what makes that visible.
    """
    frame = pd.DataFrame({
        "city": np.asarray(cities),
        "y_true": np.asarray(y_true, dtype="float64"),
        "y_pred": np.clip(np.asarray(y_pred, dtype="float64"), 0.0, None),
    })

    rows = []
    for city, g in frame.groupby("city"):
        m = compute_metrics(g["y_true"], g["y_pred"])
        rows.append({
            "City": city,
            "n_test": int(len(g)),
            "mean_rent": float(g["y_true"].mean()),
            "median_rent": float(g["y_true"].median()),
            "std_rent": float(g["y_true"].std()),
            "rmse": m["rmse"],
            "mae": m["mae"],
            "r2": m["r2"],
            "mape": m["mape"],
            # RMSE as a share of the city's mean rent -- the only fair way to
            # compare accuracy between Mumbai and Kolkata, whose rent scales
            # differ by roughly 5x.
            "rmse_pct_of_mean": float(m["rmse"] / g["y_true"].mean() * 100.0),
        })

    return (pd.DataFrame(rows)
            .sort_values("rmse_pct_of_mean")
            .reset_index(drop=True))


# --------------------------------------------------------------------------- #
# Section 10 -- worst predictions
# --------------------------------------------------------------------------- #
def worst_predictions(X_test: pd.DataFrame, y_true, y_pred,
                      n: int = 10) -> pd.DataFrame:
    """The `n` listings with the largest absolute rupee error, with context.

    The accompanying columns (city, locality, size, BHK, how rare the locality is)
    are what make it possible to say *why* each one failed rather than just that
    it did.
    """
    y_true = np.asarray(y_true, dtype="float64")
    y_pred = np.clip(np.asarray(y_pred, dtype="float64"), 0.0, None)

    out = pd.DataFrame({
        "City": X_test[cfg.COL_CITY].to_numpy(),
        "Area Locality": X_test[cfg.COL_LOCALITY].to_numpy(),
        "BHK": X_test[cfg.COL_BHK].to_numpy(),
        "Size": X_test[cfg.COL_SIZE].to_numpy(),
        "Area Type": X_test[cfg.COL_AREA_TYPE].to_numpy(),
        "Furnishing Status": X_test[cfg.COL_FURNISHING].to_numpy(),
        "actual_rent": y_true,
        "predicted_rent": y_pred,
    })
    out["error"] = out["predicted_rent"] - out["actual_rent"]
    out["abs_error"] = out["error"].abs()
    out["pct_error"] = out["error"] / out["actual_rent"] * 100.0
    out["direction"] = np.where(out["error"] > 0, "over-predicted",
                                "under-predicted")

    return (out.sort_values("abs_error", ascending=False)
            .head(n).reset_index(drop=True))


def residual_frame(X_test: pd.DataFrame, y_true, y_pred) -> pd.DataFrame:
    """Tidy frame of residuals with city labels, for the section-10 diagnostics."""
    y_true = np.asarray(y_true, dtype="float64")
    y_pred = np.clip(np.asarray(y_pred, dtype="float64"), 0.0, None)
    return pd.DataFrame({
        "city": X_test[cfg.COL_CITY].to_numpy(),
        "actual": y_true,
        "predicted": y_pred,
        "residual": y_true - y_pred,
        "abs_residual": np.abs(y_true - y_pred),
        "pct_residual": (y_true - y_pred) / y_true * 100.0,
    })


# --------------------------------------------------------------------------- #
# Formatting helpers
# --------------------------------------------------------------------------- #
def format_inr(value: float, decimals: int = 0) -> str:
    """Format a rupee amount with Indian digit grouping: 2850000 -> 28,50,000.

    Python's built-in `,` grouping produces 2,850,000, which is not how rupee
    amounts are written in India. Used by the console report, the README tables
    and the Streamlit app.
    """
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "n/a"
    neg = value < 0
    whole = f"{abs(value):.{decimals}f}"
    frac = ""
    if "." in whole:
        whole, frac = whole.split(".")
        frac = "." + frac
    if len(whole) > 3:
        head, tail = whole[:-3], whole[-3:]
        head = re.sub(r"(\d)(?=(\d\d)+$)", r"\1,", head)
        whole = f"{head},{tail}"
    return ("-" if neg else "") + "₹" + whole + frac


def cv_from_search(search) -> dict[str, float]:
    """Extract per-fold CV statistics for the winning configuration of a search.

    Equivalent in shape to `cv_report`, but read straight out of
    ``RandomizedSearchCV.cv_results_`` so the 5-fold fits already performed
    during tuning are reused instead of being repeated.

    Note on interpretation: for a tuned model this CV score belongs to the
    configuration that was *selected* using those same folds, so it is mildly
    optimistic. The held-out test set -- never touched during tuning -- is the
    unbiased estimate, and both are reported side by side for that reason.
    """
    res = search.cv_results_
    i = search.best_index_
    out: dict[str, float] = {}
    for key in METRIC_KEYS:
        sign = 1.0 if key == "r2" else -1.0
        out[f"cv_{key}_mean"] = float(sign * res[f"mean_test_{key}"][i])
        out[f"cv_{key}_std"] = float(res[f"std_test_{key}"][i])
        if f"mean_train_{key}" in res:
            out[f"cv_train_{key}_mean"] = float(sign * res[f"mean_train_{key}"][i])
    out["cv_fit_time_mean"] = float(res["mean_fit_time"][i])
    return out


def rent_band(values, edges=(15000, 40000, 100000)) -> pd.Series:
    """Bucket rents into interpretable bands for the premium-property analysis.

    Section 9's conclusion is about *where* the error lands, not just how big it
    is on average, so errors are reported within bands rather than pooled.
    """
    v = pd.Series(np.asarray(values, dtype="float64"))
    labels = [
        f"< {format_inr(edges[0])}",
        f"{format_inr(edges[0])} – {format_inr(edges[1])}",
        f"{format_inr(edges[1])} – {format_inr(edges[2])}",
        f"> {format_inr(edges[2])} (premium)",
    ]
    return pd.cut(v, bins=[-np.inf, *edges, np.inf], labels=labels,
                  right=False).astype(object)


def error_concentration(y_true, y_pred,
                        ks=(1, 2, 5, 10, 20, 50)) -> dict[str, float]:
    """How much of the total squared error comes from the few worst rows?

    RMSE squares errors, so on a target this heavy-tailed a handful of listings
    can decide the entire ranking of models. Quantifying that is what justifies
    ranking on MAE instead of RMSE -- see the model-selection section of
    `train.py`.
    """
    y_true = np.asarray(y_true, dtype="float64")
    y_pred = np.clip(np.asarray(y_pred, dtype="float64"), 0.0, None)
    sq = (y_true - y_pred) ** 2
    order = np.argsort(sq)[::-1]
    total = float(sq.sum()) or 1.0

    out = {"n_rows": int(len(sq))}
    for k in ks:
        if k <= len(sq):
            out[f"top_{k}_share_of_mse"] = float(sq[order[:k]].sum() / total)
    keep = np.ones(len(sq), dtype=bool)
    keep[order[:max(1, len(sq) // 100)]] = False
    out["rmse_excluding_worst_1pct"] = float(np.sqrt(np.mean(sq[keep])))
    out["rmse_all"] = float(np.sqrt(np.mean(sq)))
    return out
