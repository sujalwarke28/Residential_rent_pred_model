"""Feature engineering, outlier treatment and the sklearn preprocessing graph.

Design notes
------------
1. `FeatureEngineer` is a *stateful* transformer, not a `FunctionTransformer`.
   The Size and total_floors imputations need group medians, and those medians
   must be learned from the training fold only. A stateless function would
   recompute them from whatever frame it was handed -- including validation and
   test rows -- which leaks information inside cross-validation.

2. It is the first step of the model pipeline, which means the Streamlit app can
   hand over a raw one-row DataFrame and do no preprocessing of its own.

3. `rent_per_sqft` is produced by `add_eda_features` and is used for exploratory
   analysis and outlier detection ONLY. It is derived from the target and would
   leak it directly; see the warning on that function.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, TargetEncoder

from src import config as cfg


# --------------------------------------------------------------------------- #
# Floor parsing
# --------------------------------------------------------------------------- #
def parse_floor(floor: pd.Series) -> pd.DataFrame:
    """Split the messy `Floor` text into `floor_num` and `total_floors`.

    The column stores two facts in one string, e.g. "Ground out of 2" or
    "3 out of 5". Word-valued floors are mapped numerically so that a linear
    model sees a sensible ordering:

        Ground          ->  0.0
        Upper Basement  -> -0.5
        Lower Basement  -> -1.0

    Rows with no "out of " part (only 4 in this dataset, e.g. "3", "Ground")
    genuinely do not record the building height, so `total_floors` comes back as
    NaN. This is the "handle missing floor values" objective: the missing values
    are hidden inside the string rather than stored as NaN in the CSV.
    """
    s = floor.astype("string").fillna("")

    parts = s.str.split(r"\s*out\s+of\s*", n=1, regex=True, expand=True)
    lhs = parts[0].str.strip()
    rhs = parts[1].str.strip() if parts.shape[1] > 1 else pd.Series(
        pd.NA, index=s.index, dtype="string"
    )

    # Word tokens first, then anything that parses as a number.
    floor_num = lhs.str.lower().map(cfg.FLOOR_TOKEN_MAP).astype("float64")
    numeric_lhs = pd.to_numeric(lhs, errors="coerce")
    floor_num = floor_num.fillna(numeric_lhs)

    total_floors = pd.to_numeric(rhs, errors="coerce")

    return pd.DataFrame(
        {cfg.FEAT_FLOOR_NUM: floor_num.to_numpy(dtype="float64"),
         cfg.FEAT_TOTAL_FLOORS: total_floors.to_numpy(dtype="float64")},
        index=floor.index,
    )


# --------------------------------------------------------------------------- #
# The stateful feature engineer (pipeline step 1)
# --------------------------------------------------------------------------- #
class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Parse `Floor`, impute `Size`/`total_floors`, and derive size features.

    Learned in ``fit`` (training fold only)
    ---------------------------------------
    * median `Size` per (City, BHK), per City, and overall
    * median `total_floors` per City, and overall
    * median `floor_num` overall

    Produced in ``transform``
    -------------------------
    * ``floor_num``, ``total_floors``  -- parsed out of the `Floor` string
    * ``floor_ratio``                 -- floor_num / total_floors, i.e. relative
      height within the building
    * ``carpet_equiv_size``           -- `Size` converted to a carpet-area
      equivalent so that the three measurement conventions are comparable
    * ``size_x_<area type>``          -- Size x Area Type interaction terms,
      consumed by the Linear and Polynomial models only

    The transformer accepts EITHER a raw frame containing `Floor`, OR a frame
    that already carries ``floor_num`` / ``total_floors`` as separate columns.
    The Streamlit app collects those two as separate inputs, so both callers work
    against the same fitted pipeline.
    """

    def __init__(self, min_valid_size: float = cfg.MIN_VALID_SIZE):
        self.min_valid_size = min_valid_size

    # -- internal helpers --------------------------------------------------- #
    def _split_floor(self, X: pd.DataFrame) -> pd.DataFrame:
        """Ensure floor_num / total_floors exist, parsing `Floor` if needed."""
        d = X.copy()
        if cfg.COL_FLOOR in d.columns:
            parsed = parse_floor(d[cfg.COL_FLOOR])
            # Pre-supplied columns (the app) take precedence where present.
            for col in (cfg.FEAT_FLOOR_NUM, cfg.FEAT_TOTAL_FLOORS):
                if col in d.columns:
                    d[col] = pd.to_numeric(d[col], errors="coerce").fillna(parsed[col])
                else:
                    d[col] = parsed[col]
        else:
            missing = [c for c in (cfg.FEAT_FLOOR_NUM, cfg.FEAT_TOTAL_FLOORS)
                       if c not in d.columns]
            if missing:
                raise ValueError(
                    f"Input must contain either the raw {cfg.COL_FLOOR!r} column "
                    f"or both {cfg.FEAT_FLOOR_NUM!r} and "
                    f"{cfg.FEAT_TOTAL_FLOORS!r}; missing {missing}."
                )
            for col in (cfg.FEAT_FLOOR_NUM, cfg.FEAT_TOTAL_FLOORS):
                d[col] = pd.to_numeric(d[col], errors="coerce")
        return d

    def _blank_implausible_size(self, d: pd.DataFrame) -> pd.DataFrame:
        """Treat sizes below the plausibility threshold as missing."""
        size = pd.to_numeric(d[cfg.COL_SIZE], errors="coerce")
        d[cfg.COL_SIZE] = size.where(size >= self.min_valid_size)
        return d

    # -- sklearn API -------------------------------------------------------- #
    def fit(self, X: pd.DataFrame, y=None):
        d = self._blank_implausible_size(self._split_floor(X))

        bhk = pd.to_numeric(d[cfg.COL_BHK], errors="coerce")
        city = d[cfg.COL_CITY].astype("object")

        # Medians are learned here, on the training fold only -- never inside
        # transform() -- so that no validation/test statistic can leak in.
        grp = d.assign(_city=city, _bhk=bhk).groupby(
            ["_city", "_bhk"], dropna=False
        )[cfg.COL_SIZE].median()
        self.size_median_city_bhk_ = {
            k: float(v) for k, v in grp.items() if pd.notna(v)
        }
        self.size_median_city_ = {
            str(k): float(v)
            for k, v in d.groupby(city)[cfg.COL_SIZE].median().items()
            if pd.notna(v)
        }
        self.size_median_global_ = float(d[cfg.COL_SIZE].median())

        self.total_floors_median_city_ = {
            str(k): float(v)
            for k, v in d.groupby(city)[cfg.FEAT_TOTAL_FLOORS].median().items()
            if pd.notna(v)
        }
        self.total_floors_median_global_ = float(d[cfg.FEAT_TOTAL_FLOORS].median())
        self.floor_num_median_global_ = float(d[cfg.FEAT_FLOOR_NUM].median())

        self.n_features_in_ = X.shape[1]
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        d = self._blank_implausible_size(self._split_floor(X))

        city = d[cfg.COL_CITY].astype("object")
        bhk = pd.to_numeric(d[cfg.COL_BHK], errors="coerce")

        # ---- Size: (City, BHK) median -> City median -> global median ------ #
        keys = pd.MultiIndex.from_arrays([city, bhk])
        by_city_bhk = pd.Series(
            [self.size_median_city_bhk_.get(k, np.nan) for k in keys],
            index=d.index, dtype="float64",
        )
        by_city = city.map(self.size_median_city_).astype("float64")
        d[cfg.COL_SIZE] = (
            d[cfg.COL_SIZE]
            .fillna(by_city_bhk)
            .fillna(by_city)
            .fillna(self.size_median_global_)
        )

        # ---- total_floors: City median -> global median -------------------- #
        tf_by_city = city.map(self.total_floors_median_city_).astype("float64")
        d[cfg.FEAT_TOTAL_FLOORS] = (
            d[cfg.FEAT_TOTAL_FLOORS]
            .fillna(tf_by_city)
            .fillna(self.total_floors_median_global_)
        )
        d[cfg.FEAT_FLOOR_NUM] = d[cfg.FEAT_FLOOR_NUM].fillna(
            self.floor_num_median_global_
        )

        # A building cannot be shorter than the floor a flat sits on.
        d[cfg.FEAT_TOTAL_FLOORS] = np.maximum(
            d[cfg.FEAT_TOTAL_FLOORS], np.maximum(d[cfg.FEAT_FLOOR_NUM], 1.0)
        )

        # ---- Relative height within the building --------------------------- #
        d[cfg.FEAT_FLOOR_RATIO] = (
            d[cfg.FEAT_FLOOR_NUM] / d[cfg.FEAT_TOTAL_FLOORS].replace(0.0, np.nan)
        ).fillna(0.0)

        # ---- Carpet-area equivalent size ----------------------------------- #
        # Makes Size comparable across the three measurement conventions. The
        # raw Size and Area Type columns are deliberately KEPT as features too,
        # so the models can still exploit the reported figure directly.
        ratio = (
            d[cfg.COL_AREA_TYPE].map(cfg.AREA_TYPE_RATIOS)
            .astype("float64").fillna(1.0)
        )
        d[cfg.FEAT_CARPET_EQUIV] = d[cfg.COL_SIZE] * ratio

        # ---- Size x Area Type interaction terms ---------------------------- #
        for area_type in cfg.AREA_TYPES:
            col = cfg.interaction_feature_name(area_type)
            d[col] = np.where(
                d[cfg.COL_AREA_TYPE].to_numpy() == area_type,
                d[cfg.COL_SIZE].to_numpy(dtype="float64"),
                0.0,
            )

        return d

    def get_feature_names_out(self, input_features=None):
        base = list(self.feature_names_in_) if input_features is None else list(input_features)
        extra = [cfg.FEAT_FLOOR_NUM, cfg.FEAT_TOTAL_FLOORS, cfg.FEAT_FLOOR_RATIO,
                 cfg.FEAT_CARPET_EQUIV] + cfg.INTERACTION_FEATURES
        return np.asarray(base + [c for c in extra if c not in base], dtype=object)


# --------------------------------------------------------------------------- #
# EDA-only derived columns
# --------------------------------------------------------------------------- #
def add_eda_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add `carpet_equiv_size` and `rent_per_sqft` for EDA / outlier detection.

    !! LEAKAGE WARNING !!
    ``rent_per_sqft`` is computed as Rent / carpet_equiv_size. It is a function
    of the target. It is used here for exploratory plots and for flagging
    implausible listings, and it must NEVER be passed to a model as a feature --
    doing so hands the model the answer and produces a suspiciously perfect R².
    It is therefore absent from `config.NUMERIC_FEATURES` and is never selected
    by the ColumnTransformer.
    """
    d = df.copy()

    size = pd.to_numeric(d[cfg.COL_SIZE], errors="coerce")
    size = size.where(size >= cfg.MIN_VALID_SIZE)
    ratio = d[cfg.COL_AREA_TYPE].map(cfg.AREA_TYPE_RATIOS).astype("float64").fillna(1.0)
    d[cfg.FEAT_CARPET_EQUIV] = size * ratio

    if cfg.TARGET in d.columns:
        d[cfg.FEAT_RENT_PER_SQFT] = d[cfg.TARGET] / d[cfg.FEAT_CARPET_EQUIV]

    floor_cols = parse_floor(d[cfg.COL_FLOOR]) if cfg.COL_FLOOR in d.columns else None
    if floor_cols is not None:
        d[cfg.FEAT_FLOOR_NUM] = floor_cols[cfg.FEAT_FLOOR_NUM]
        d[cfg.FEAT_TOTAL_FLOORS] = floor_cols[cfg.FEAT_TOTAL_FLOORS]

    return d


def empirical_area_ratios(df: pd.DataFrame, min_count: int = 5) -> pd.DataFrame:
    """Test the conventional area-conversion ratios against the data (section 5.2).

    Within each (City, BHK) group -- where genuinely comparable flats sit side by
    side -- compare the median reported `Size` of each Area Type against the
    median reported `Size` of Carpet Area listings. If the conventional ratios
    hold, Super Area listings should report roughly 1/0.70 = 1.43x the carpet
    figure.

    Only groups with at least `min_count` listings of both types are compared.
    """
    rows = []
    for (city, bhk), g in df.groupby([cfg.COL_CITY, cfg.COL_BHK]):
        medians = g.groupby(cfg.COL_AREA_TYPE)[cfg.COL_SIZE].agg(["median", "count"])
        if "Carpet Area" not in medians.index:
            continue
        carpet = medians.loc["Carpet Area"]
        if carpet["count"] < min_count:
            continue
        for area_type in medians.index:
            if area_type == "Carpet Area":
                continue
            other = medians.loc[area_type]
            if other["count"] < min_count:
                continue
            rows.append({
                cfg.COL_CITY: city,
                cfg.COL_BHK: bhk,
                "area_type": area_type,
                "median_size_carpet": float(carpet["median"]),
                "median_size_other": float(other["median"]),
                "n_carpet": int(carpet["count"]),
                "n_other": int(other["count"]),
                # Observed multiplier that converts the other basis -> carpet.
                "observed_ratio_to_carpet": float(carpet["median"] / other["median"]),
                "conventional_ratio": cfg.AREA_TYPE_RATIOS[area_type],
            })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Outlier detection and treatment
# --------------------------------------------------------------------------- #
def detect_outliers(df: pd.DataFrame, per_city: bool = True) -> pd.DataFrame:
    """Flag outlier listings using a PER-CITY IQR rule on `Rent`.

    A global IQR filter is wrong for this dataset: Kolkata's 99th percentile rent
    is around Rs 40,000 while Mumbai's *median* is around Rs 52,000, so a global
    rule would delete a large slice of legitimate Mumbai listings. Bounds are
    therefore computed within each city.

    Extreme `rent_per_sqft` values are flagged as a second, independent signal --
    these catch listings whose rent is implausible *given their size*, which a
    rent-only rule misses.

    Returns
    -------
    DataFrame indexed like `df` with boolean columns:
        rent_iqr_outlier, rent_per_sqft_outlier, is_outlier
    and the per-group bounds used, for reporting.
    """
    d = add_eda_features(df)
    grouper = d[cfg.COL_CITY] if per_city else pd.Series("ALL", index=d.index)

    rent = d[cfg.TARGET]
    q1 = rent.groupby(grouper).transform(lambda s: s.quantile(0.25))
    q3 = rent.groupby(grouper).transform(lambda s: s.quantile(0.75))
    iqr = q3 - q1
    lower = q1 - cfg.IQR_MULTIPLIER * iqr
    upper = q3 + cfg.IQR_MULTIPLIER * iqr
    rent_flag = (rent < lower) | (rent > upper)

    rps = d[cfg.FEAT_RENT_PER_SQFT]
    rps_lo = rps.groupby(grouper).transform(
        lambda s: s.quantile(cfg.RENT_PER_SQFT_LOWER_Q))
    rps_hi = rps.groupby(grouper).transform(
        lambda s: s.quantile(cfg.RENT_PER_SQFT_UPPER_Q))
    rps_flag = ((rps < rps_lo) | (rps > rps_hi)).fillna(False)

    return pd.DataFrame({
        "rent_iqr_outlier": rent_flag,
        "rent_per_sqft_outlier": rps_flag,
        "is_outlier": rent_flag | rps_flag,
        "rent_lower_bound": lower,
        "rent_upper_bound": upper,
    }, index=d.index)


def treat_outliers(
    df: pd.DataFrame,
    strategy: str = "none",
    per_city: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """Apply an outlier strategy. Call this on the TRAINING split only.

    Strategies
    ----------
    ``none``   leave the data untouched.
    ``remove`` drop the flagged rows entirely.
    ``cap``    winsorize `Rent` at the per-city 1st and 99th percentiles, keeping
               every row but pulling extreme rents inwards.

    Applying this to the test split too would be cheating: removing hard cases
    makes the metrics look better without the model improving. `train.py`
    therefore treats the training split only and always scores against the same
    untreated held-out test set.
    """
    if strategy not in {"none", "remove", "cap"}:
        raise ValueError(f"Unknown outlier strategy {strategy!r}; "
                         "expected 'none', 'remove' or 'cap'.")

    info = {"strategy": strategy, "n_rows_in": int(len(df))}

    if strategy == "none":
        info.update(n_rows_out=int(len(df)), n_removed=0, n_capped=0)
        return df.copy(), info

    flags = detect_outliers(df, per_city=per_city)
    info["n_flagged"] = int(flags["is_outlier"].sum())
    info["n_flagged_rent_iqr"] = int(flags["rent_iqr_outlier"].sum())
    info["n_flagged_rent_per_sqft"] = int(flags["rent_per_sqft_outlier"].sum())

    if strategy == "remove":
        out = df.loc[~flags["is_outlier"]].copy()
        info.update(n_rows_out=int(len(out)),
                    n_removed=int(len(df) - len(out)),
                    n_capped=0)
        return out, info

    # strategy == "cap"
    out = df.copy()
    grouper = out[cfg.COL_CITY] if per_city else pd.Series("ALL", index=out.index)
    lo = out[cfg.TARGET].groupby(grouper).transform(
        lambda s: s.quantile(cfg.WINSOR_LOWER_PCT))
    hi = out[cfg.TARGET].groupby(grouper).transform(
        lambda s: s.quantile(cfg.WINSOR_UPPER_PCT))
    capped = out[cfg.TARGET].clip(lower=lo, upper=hi)
    info.update(n_rows_out=int(len(out)),
                n_removed=0,
                n_capped=int((capped != out[cfg.TARGET]).sum()))
    out[cfg.TARGET] = capped
    return out, info


# --------------------------------------------------------------------------- #
# The sklearn preprocessing graph
# --------------------------------------------------------------------------- #
def build_preprocessor(
    family: str = "tree",
    feature_set: str = "all",
) -> ColumnTransformer:
    """Build the ColumnTransformer for a given model family and feature subset.

    Parameters
    ----------
    feature_set
        Supports the section-11 locality ablation study:

        ``"all"``            every feature (ablation Model A)
        ``"no_location"``    drop both `Area Locality` and `City` (Model B)
        ``"only_location"``  keep ONLY `Area Locality` and `City` (Model C)

    family
        ``"linear"`` for Linear / Polynomial models: numeric features are
        standardised, the Size x Area Type interaction terms are included, and
        one-hot columns use ``drop="first"`` to avoid the perfect collinearity
        that makes a plain polynomial expansion singular.

        ``"tree"`` for Decision Tree / Random Forest / Gradient Boosting:
        no scaling (trees are scale invariant), no interaction terms (trees learn
        the Size-by-Area-Type split themselves), full one-hot coding.
    """
    if family not in {"linear", "tree"}:
        raise ValueError(f"family must be 'linear' or 'tree', got {family!r}")
    if feature_set not in {"all", "no_location", "only_location"}:
        raise ValueError(
            "feature_set must be 'all', 'no_location' or 'only_location', "
            f"got {feature_set!r}"
        )

    is_linear = family == "linear"

    numeric = list(cfg.NUMERIC_FEATURES)
    if is_linear:
        numeric = numeric + list(cfg.INTERACTION_FEATURES)

    onehot_cols = list(cfg.ONEHOT_FEATURES)
    locality_cols = list(cfg.TARGET_ENCODE_FEATURES)

    if feature_set == "no_location":
        onehot_cols = [c for c in onehot_cols if c != cfg.COL_CITY]
        locality_cols = []
    elif feature_set == "only_location":
        numeric = []
        onehot_cols = [cfg.COL_CITY]

    numeric_steps = [("impute", SimpleImputer(strategy="median"))]
    if is_linear:
        numeric_steps.append(("scale", StandardScaler()))
    numeric_pipe = Pipeline(numeric_steps)

    # handle_unknown="ignore" is REQUIRED, not defensive tidiness: `Built Area`
    # has 2 rows and `Contact Builder` has 1 in the whole dataset, so such a
    # level can appear in the test split having never been seen in training.
    # Without it the encoder raises at predict time.
    onehot_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("encode", OneHotEncoder(
            handle_unknown="ignore",
            drop="first" if is_linear else None,
            sparse_output=False,
        )),
    ])

    # ------------------------------------------------------------------ #
    # Area Locality: ~2,235 levels, 1,465 of them appearing exactly once.
    #
    # CRITICAL -- LEAKAGE: TargetEncoder is placed INSIDE this pipeline, which
    # means it is re-fitted on each training fold during cross-validation and
    # never sees the validation fold's target. Fitting the encoder once on the
    # full training set before cross-validating would leak the answer into every
    # CV score and produce optimistically biased results. sklearn's TargetEncoder
    # additionally cross-fits internally during fit_transform, so even the
    # training-fold encodings are out-of-fold. Do not hoist this out of the
    # pipeline.
    # ------------------------------------------------------------------ #
    locality_steps = [
        ("impute", SimpleImputer(strategy="constant", fill_value="__missing__")),
        ("encode", TargetEncoder(
            target_type="continuous",
            smooth="auto",   # empirical-Bayes shrinkage towards the global mean
            # A seeded KFold makes the encoder's internal cross-fitting
            # reproducible. (scikit-learn >= 1.9 deprecates the `shuffle` /
            # `random_state` arguments in favour of passing a CV splitter here.)
            cv=KFold(n_splits=cfg.N_SPLITS, shuffle=True,
                     random_state=cfg.RANDOM_STATE),
        )),
    ]
    if is_linear:
        locality_steps.append(("scale", StandardScaler()))
    locality_pipe = Pipeline(locality_steps)

    transformers = []
    if numeric:
        transformers.append(("num", numeric_pipe, numeric))
    if onehot_cols:
        transformers.append(("cat", onehot_pipe, onehot_cols))
    if locality_cols:
        transformers.append(("loc", locality_pipe, locality_cols))

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",   # drops Posted On, Floor, Rent and anything else
        verbose_feature_names_out=False,
    )
