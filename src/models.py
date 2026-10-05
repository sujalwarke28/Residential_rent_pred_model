"""The six model variants, their search spaces, and pipeline assembly.

Why six models for five named algorithms
----------------------------------------
The case study names five algorithms. Polynomial Regression is reported as TWO
rows -- plain degree 2, and degree 2 with Ridge regularisation -- because a plain
degree-2 expansion over one-hot encoded features produces a few hundred
correlated terms and overfits badly. Reporting both keeps the deliverable aligned
with the specification while showing the improvement that regularisation buys.

Target handling
---------------
Each model is assembled as

    TransformedTargetRegressor(
        regressor = Pipeline([FeatureEngineer, ColumnTransformer, model]),
        func      = np.log1p,
        inverse_func = np.expm1,
    )

which buys three things at once:

1. ``y`` stays in rupees, so ordinary scorers report R2/MSE/RMSE/MAE in rupees
   with no manual inversion to get wrong.
2. The ``TargetEncoder`` inside the ColumnTransformer sees the *log* target,
   which is far less distorted by the 21.4 skew of raw rent.
3. The saved pipeline predicts rupees directly, so the Streamlit app needs no
   post-processing.

Setting ``log_target=False`` drops the wrapper and trains on raw rent -- that is
setting 1 of the section-9 outlier experiment.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.stats import loguniform, randint, uniform
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor

from src import config as cfg
from src.preprocess import FeatureEngineer, build_preprocessor


@dataclass
class ModelSpec:
    """Everything needed to build, tune and report one model variant."""

    key: str                       # machine name, used in dict keys / filenames
    name: str                      # display name for tables and plots
    family: str                    # "linear" (scale + interactions) or "tree"
    build: Any                     # zero-arg callable -> a fresh estimator
    param_distributions: dict = field(default_factory=dict)
    n_iter: int = 1                # RandomizedSearchCV iterations
    note: str = ""                 # commentary carried into the report


# --------------------------------------------------------------------------- #
# Search spaces
# --------------------------------------------------------------------------- #
def _poly(model):
    """Degree-2 polynomial expansion feeding a linear estimator."""
    return Pipeline([
        ("poly", PolynomialFeatures(degree=2, include_bias=False)),
        ("reg", model),
    ])


def model_specs(fast: bool = False) -> list[ModelSpec]:
    """Return all six model variants in report order.

    Parameters
    ----------
    fast
        Shrink every search space to a single iteration. Used to smoke-test the
        whole training script's structure in seconds before committing to the
        full search. Never used for reported numbers.
    """
    def n(full: int) -> int:
        return 1 if fast else full

    return [
        ModelSpec(
            key="linear",
            name="Linear Regression",
            family="linear",
            build=lambda: LinearRegression(),
            param_distributions={},
            n_iter=1,
            note="Baseline. No hyperparameters to tune, by design.",
        ),
        ModelSpec(
            key="poly2",
            name="Polynomial Regression (deg 2)",
            family="linear",
            build=lambda: _poly(LinearRegression()),
            param_distributions={},
            n_iter=1,
            note=("Plain degree-2 expansion, no regularisation. Expands ~24 "
                  "encoded features into ~324 correlated terms, so it is "
                  "expected to overfit."),
        ),
        ModelSpec(
            key="poly2_ridge",
            name="Polynomial Regression (deg 2) + Ridge",
            family="linear",
            build=lambda: _poly(Ridge(random_state=cfg.RANDOM_STATE)),
            # A wide shrinkage range on purpose. Degree-2 expansion of ~24
            # standardised features produces terms whose squares are large, so
            # the alpha that actually controls this model can be very high; a
            # range truncated at 1e4 would handicap it and make the comparison
            # against the unregularised version unfair.
            param_distributions={"model__reg__alpha": loguniform(1e-2, 1e7)},
            n_iter=n(30),
            note=("Same expansion with L2 shrinkage, which controls the "
                  "collinearity the plain version suffers from."),
        ),
        ModelSpec(
            key="decision_tree",
            name="Decision Tree",
            family="tree",
            build=lambda: DecisionTreeRegressor(random_state=cfg.RANDOM_STATE),
            param_distributions={
                "model__max_depth": [3, 5, 7, 9, 11, 14, 18, 25, None],
                "model__min_samples_leaf": randint(1, 40),
                "model__min_samples_split": randint(2, 40),
                "model__max_features": [None, 0.5, 0.7, 0.9, "sqrt"],
            },
            n_iter=n(40),
            note="Single tree; interpretable but high variance.",
        ),
        ModelSpec(
            key="random_forest",
            name="Random Forest",
            family="tree",
            # n_jobs=1 deliberately: the surrounding RandomizedSearchCV already
            # runs with n_jobs=-1, and letting both parallelise oversubscribes
            # the CPU with nested thread pools and runs slower, not faster.
            build=lambda: RandomForestRegressor(
                random_state=cfg.RANDOM_STATE, n_jobs=1),
            param_distributions={
                "model__n_estimators": randint(200, 600),
                "model__max_depth": [8, 12, 16, 20, 28, None],
                "model__max_features": [0.3, 0.5, 0.7, 1.0, "sqrt"],
                "model__min_samples_leaf": randint(1, 8),
            },
            n_iter=n(25),
            note="Bagged trees; strong default performance, low variance.",
        ),
        ModelSpec(
            key="gradient_boosting",
            name="Gradient Boosting",
            family="tree",
            build=lambda: GradientBoostingRegressor(
                random_state=cfg.RANDOM_STATE),
            param_distributions={
                "model__learning_rate": loguniform(0.01, 0.3),
                "model__n_estimators": randint(150, 600),
                "model__max_depth": randint(2, 7),
                "model__subsample": uniform(0.7, 0.3),   # 0.70 .. 1.00
                "model__min_samples_leaf": randint(1, 20),
            },
            n_iter=n(30),
            note="Sequential boosting; usually the strongest on tabular data.",
        ),
    ]


def spec_by_key(key: str, fast: bool = False) -> ModelSpec:
    for s in model_specs(fast=fast):
        if s.key == key:
            return s
    raise KeyError(f"No model spec with key {key!r}")


# --------------------------------------------------------------------------- #
# Pipeline assembly
# --------------------------------------------------------------------------- #
def build_pipeline(
    spec: ModelSpec,
    log_target: bool = True,
    feature_set: str = "all",
    model: Any = None,
):
    """Assemble the full estimator for one model spec.

    The returned object is what gets fitted, cross-validated and saved. Every
    preprocessing step lives inside it, so the Streamlit app can call
    ``.predict()`` on a raw one-row DataFrame.
    """
    inner = Pipeline([
        ("features", FeatureEngineer()),
        ("prep", build_preprocessor(spec.family, feature_set=feature_set)),
        ("model", spec.build() if model is None else model),
    ])
    if not log_target:
        return inner
    return TransformedTargetRegressor(
        regressor=inner, func=np.log1p, inverse_func=np.expm1,
    )


def prefix_params(params: dict, log_target: bool = True) -> dict:
    """Re-key a spec's search space for the assembled estimator.

    ``TransformedTargetRegressor`` nests the pipeline under ``regressor__``, so
    ``model__reg__alpha`` becomes ``regressor__model__reg__alpha``.
    """
    if not log_target:
        return dict(params)
    return {f"regressor__{k}": v for k, v in params.items()}


def inner_pipeline(estimator):
    """Return the inner Pipeline, unwrapping TransformedTargetRegressor."""
    if isinstance(estimator, TransformedTargetRegressor):
        return estimator.regressor_
    return estimator


def get_fitted_model(estimator):
    """Return the final fitted model step of an assembled estimator."""
    return inner_pipeline(estimator).named_steps["model"]


def get_fitted_preprocessor(estimator):
    """Return the fitted ColumnTransformer of an assembled estimator."""
    return inner_pipeline(estimator).named_steps["prep"]


def get_feature_names(estimator) -> list[str]:
    """Human-readable names of the columns the final model actually sees."""
    return list(get_fitted_preprocessor(estimator).get_feature_names_out())


# --------------------------------------------------------------------------- #
# Section 14 -- quantile models for the app's prediction interval
# --------------------------------------------------------------------------- #
def build_quantile_pipeline(alpha: float, best_params: dict | None = None):
    """Gradient Boosting with the quantile loss, for an 80% prediction interval.

    Trained on ``log1p(Rent)`` like every other model. Quantiles are invariant
    under monotone transforms, so ``expm1(q_alpha(log rent))`` is exactly the
    alpha-quantile of rent in rupees -- no approximation is involved.

    `best_params` lets the interval models inherit the tuned tree geometry of the
    main Gradient Boosting model (learning rate, depth, subsample, ...), which is
    cheaper and more consistent than tuning them separately.
    """
    gb_kwargs: dict[str, Any] = dict(
        loss="quantile", alpha=alpha, random_state=cfg.RANDOM_STATE,
    )
    if best_params:
        keep = ("learning_rate", "n_estimators", "max_depth",
                "subsample", "min_samples_leaf")
        for full_key, value in best_params.items():
            short = full_key.split("__")[-1]
            if short in keep:
                gb_kwargs[short] = value

    spec = ModelSpec(
        key=f"quantile_{alpha:g}",
        name=f"Gradient Boosting (quantile alpha={alpha:g})",
        family="tree",
        build=lambda: GradientBoostingRegressor(**gb_kwargs),
    )
    return build_pipeline(spec, log_target=True)
