"""Dataset loading and the section-4 profiling pass.

`load_data()` is the single entry point used by `train.py`, the notebook and any
ad-hoc script. There is deliberately no synthetic fallback: the real Kaggle
dataset ships with this project, and quietly substituting generated data would
invalidate every number in the report.
"""

from __future__ import annotations

import re
from collections import Counter

import numpy as np
import pandas as pd

from src import config as cfg

_MISSING_MSG = """\
Dataset not found at:
    {path}

The project expects the Kaggle "House Rent Prediction Dataset" CSV at that
exact location. To fix this:
  1. Download `House_Rent_Dataset.csv` from
     https://www.kaggle.com/datasets/iamsouravbanerjee/house-rent-prediction-dataset
  2. Place it at  data/House_Rent_Dataset.csv  (relative to the project root)

No synthetic fallback is generated on purpose -- every metric in the report
refers to the real dataset.
"""


def load_data(validate: bool = True) -> pd.DataFrame:
    """Load the raw rental listings CSV.

    Parameters
    ----------
    validate
        When True, assert that the CSV header matches the expected column
        spellings. The ColumnTransformer selects columns by exact name, so a
        renamed or re-cased header must fail loudly here rather than silently
        dropping a feature later.

    Raises
    ------
    FileNotFoundError
        If the CSV is missing, with instructions for obtaining it.
    ValueError
        If `validate` is True and the header does not match.
    """
    if not cfg.RAW_CSV.exists():
        raise FileNotFoundError(_MISSING_MSG.format(path=cfg.RAW_CSV))

    df = pd.read_csv(cfg.RAW_CSV)

    if validate:
        found = list(df.columns)
        expected = cfg.EXPECTED_RAW_COLUMNS
        if found != expected:
            missing = [c for c in expected if c not in found]
            extra = [c for c in found if c not in expected]
            raise ValueError(
                "Unexpected CSV header.\n"
                f"  expected: {expected}\n"
                f"  found:    {found}\n"
                f"  missing:  {missing}\n"
                f"  extra:    {extra}\n"
                "Column names are matched exactly by the ColumnTransformer; "
                "update src/config.py if the dataset schema really did change."
            )

    return df


def load_glossary() -> str:
    """Return the dataset glossary text, or a placeholder if it is absent."""
    if not cfg.GLOSSARY.exists():
        return "(Dataset Glossary.txt not found in data/)"
    return cfg.GLOSSARY.read_text(encoding="utf-8", errors="replace")


# --------------------------------------------------------------------------- #
# Section 4 -- profile the real data before writing any preprocessing
# --------------------------------------------------------------------------- #
def floor_pattern(value: object) -> str:
    """Collapse a `Floor` string to its structural pattern.

    "3 out of 5" and "12 out of 14" both become "N out of N", which makes the
    handful of distinct shapes visible among ~480 distinct raw strings.
    """
    return re.sub(r"\d+", "N", str(value)).strip()


def profile_data(df: pd.DataFrame) -> dict:
    """Compute the full section-4 profile as a plain dictionary.

    Returned so that `train.py` can print it and the notebook can render it as
    tables without the two diverging.
    """
    size = df[cfg.COL_SIZE]
    rent = df[cfg.TARGET]
    floors = df[cfg.COL_FLOOR].astype(str)
    posted = pd.to_datetime(df[cfg.COL_POSTED_ON], errors="coerce")

    has_out_of = floors.str.contains("out of", case=False, na=False)
    locality_counts = df[cfg.COL_LOCALITY].value_counts()

    span_days = int((posted.max() - posted.min()).days)

    profile = {
        "n_rows": int(len(df)),
        "n_cols": int(df.shape[1]),
        "columns": list(df.columns),
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
        "null_counts": {c: int(n) for c, n in df.isna().sum().items()},
        "total_nulls": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),

        # Target distribution
        "rent": {
            "min": float(rent.min()),
            "p01": float(rent.quantile(0.01)),
            "p25": float(rent.quantile(0.25)),
            "median": float(rent.median()),
            "mean": float(rent.mean()),
            "p75": float(rent.quantile(0.75)),
            "p99": float(rent.quantile(0.99)),
            "max": float(rent.max()),
            "std": float(rent.std()),
            "skew": float(rent.skew()),
            "skew_log1p": float(np.log1p(rent).skew()),
        },

        # Size plausibility
        "size": {
            "min": float(size.min()),
            "median": float(size.median()),
            "max": float(size.max()),
            "below_min_valid": int((size < cfg.MIN_VALID_SIZE).sum()),
            "below_100": int((size < 100).sum()),
            "min_valid_threshold": cfg.MIN_VALID_SIZE,
        },

        # Floor string shapes
        "floor": {
            "n_distinct_raw": int(floors.nunique()),
            "patterns": dict(Counter(floors.map(floor_pattern)).most_common()),
            "missing_total_floors": int((~has_out_of).sum()),
            "examples_without_out_of": floors[~has_out_of].unique().tolist()[:10],
        },

        # Locality cardinality
        "locality": {
            "n_unique": int(locality_counts.size),
            "appearing_once": int((locality_counts == 1).sum()),
            "appearing_twice_or_fewer": int((locality_counts <= 2).sum()),
            "share_of_rows_in_singleton_localities": float(
                (locality_counts == 1).sum() / len(df)
            ),
            "top_10": {str(k): int(v) for k, v in locality_counts.head(10).items()},
        },

        # Posted On -- the basis for dropping it
        "posted_on": {
            "min": str(posted.min().date()),
            "max": str(posted.max().date()),
            "span_days": span_days,
            "n_unique_dates": int(posted.nunique()),
            "n_unparsed": int(posted.isna().sum()),
        },

        # Categorical levels, including the rare ones that need guarding
        "categoricals": {
            col: {str(k): int(v) for k, v in df[col].value_counts().items()}
            for col in cfg.ONEHOT_FEATURES
        },
        "discrete_counts": {
            col: {str(k): int(v) for k, v in df[col].value_counts().sort_index().items()}
            for col in (cfg.COL_BHK, cfg.COL_BATHROOM)
        },
    }

    # Rare categorical levels are called out explicitly: a level with a single
    # row lands in either the train or the test split but never both, so the
    # one-hot encoder must be configured with handle_unknown="ignore".
    profile["rare_levels"] = {
        col: {lvl: n for lvl, n in counts.items() if n <= 5}
        for col, counts in profile["categoricals"].items()
    }

    return profile


def _fmt_inr(value: float) -> str:
    """Format a rupee amount with Indian digit grouping (e.g. 35,00,000)."""
    neg = value < 0
    whole = f"{abs(value):.0f}"
    if len(whole) > 3:
        head, tail = whole[:-3], whole[-3:]
        head = re.sub(r"(\d)(?=(\d\d)+$)", r"\1,", head)
        whole = f"{head},{tail}"
    return ("-" if neg else "") + whole


def print_profile(profile: dict) -> None:
    """Print the section-4 profile as a readable console report."""
    p = profile
    line = "-" * 78

    print(line)
    print("SECTION 4 -- DATA PROFILE (real data, no assumptions)")
    print(line)
    print(f"Rows: {p['n_rows']:,}    Columns: {p['n_cols']}    "
          f"Duplicate rows: {p['duplicate_rows']}")
    print(f"Total explicit NaNs across the whole frame: {p['total_nulls']}")

    print("\nDtypes and null counts per column")
    print(f"  {'column':<20} {'dtype':<16} {'nulls':>6}")
    for col in p["columns"]:
        print(f"  {col:<20} {p['dtypes'][col]:<16} {p['null_counts'][col]:>6}")

    r = p["rent"]
    print("\nTarget `Rent` distribution (INR / month)")
    print(f"  min      {_fmt_inr(r['min']):>12}")
    print(f"  p01      {_fmt_inr(r['p01']):>12}")
    print(f"  p25      {_fmt_inr(r['p25']):>12}")
    print(f"  median   {_fmt_inr(r['median']):>12}")
    print(f"  mean     {_fmt_inr(r['mean']):>12}")
    print(f"  p75      {_fmt_inr(r['p75']):>12}")
    print(f"  p99      {_fmt_inr(r['p99']):>12}")
    print(f"  max      {_fmt_inr(r['max']):>12}")
    print(f"  skew     {r['skew']:>12.2f}   <- extreme right skew; "
          f"log1p skew = {r['skew_log1p']:.2f}")

    s = p["size"]
    print("\n`Size` plausibility (sq ft)")
    print(f"  min {s['min']:.0f}   median {s['median']:.0f}   max {s['max']:.0f}")
    print(f"  implausible (< {s['min_valid_threshold']:.0f} sq ft): "
          f"{s['below_min_valid']} rows      (< 100 sq ft: {s['below_100']} rows)")

    f = p["floor"]
    print(f"\n`Floor` string patterns ({f['n_distinct_raw']} distinct raw strings)")
    for pat, n in f["patterns"].items():
        print(f"  {pat:<28} {n:>5}")
    print(f"  rows with no 'out of' part -> total_floors is missing: "
          f"{f['missing_total_floors']}")
    print(f"  examples: {f['examples_without_out_of']}")

    loc = p["locality"]
    print(f"\n`Area Locality` cardinality")
    print(f"  unique values:              {loc['n_unique']:,}")
    print(f"  appearing exactly once:     {loc['appearing_once']:,} "
          f"({loc['share_of_rows_in_singleton_localities']:.1%} of all rows)")
    print(f"  appearing twice or fewer:   {loc['appearing_twice_or_fewer']:,}")
    print("  -> one-hot is untenable; TargetEncoder with smoothing is used.")

    po = p["posted_on"]
    print(f"\n`Posted On` date range")
    print(f"  {po['min']} -> {po['max']}  ({po['span_days']} days, "
          f"{po['n_unique_dates']} distinct dates)")
    print("  -> span is far too short to express inflation or market drift;")
    print("     dropped after profiling (reasoning recorded in the notebook).")

    print("\nCategorical levels")
    for col, counts in p["categoricals"].items():
        print(f"  {col}:")
        for lvl, n in counts.items():
            flag = "   <- rare, needs handle_unknown='ignore'" if n <= 5 else ""
            print(f"      {lvl:<22} {n:>5}{flag}")

    print("\nDiscrete numeric levels")
    for col, counts in p["discrete_counts"].items():
        print(f"  {col}: {counts}")
    print(line)
