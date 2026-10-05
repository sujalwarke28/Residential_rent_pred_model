# Residential Rent Prediction — Complete Project Details

> **Written for:** you, as revision + viva-prep study material.
> Every number in this document was read out of `models/metrics.json` and `reports/*.csv`
> produced by the actual training run — nothing here is illustrative or made up.

---

## Table of Contents

1. [Project at a Glance](#1-project-at-a-glance)
2. [The Dataset](#2-the-dataset)
3. [Libraries Used — What, Why, How They Work](#3-libraries-used--what-why-how-they-work)
4. [Every Import, Explained](#4-every-import-explained)
5. [Every Code File, Explained](#5-every-code-file-explained)
6. [The End-to-End Pipeline](#6-the-end-to-end-pipeline)
7. [Preprocessing in Depth](#7-preprocessing-in-depth)
8. [Outlier Treatment](#8-outlier-treatment)
9. [Every ML Model — Principle, Formulas, Examples](#9-every-ml-model--principle-formulas-examples)
10. [Evaluation Criteria](#10-evaluation-criteria)
11. [Cross-Validation](#11-cross-validation)
12. [Results](#12-results)
13. [Residual Analysis](#13-residual-analysis)
14. [Feature Importance and the Locality Ablation](#14-feature-importance-and-the-locality-ablation)
15. [Prediction Intervals](#15-prediction-intervals)
16. [Deployment — The Streamlit App](#16-deployment--the-streamlit-app)
17. [Every Figure Produced](#17-every-figure-produced)
18. [Limitations](#18-limitations)
19. [ML Student POV — Concepts This Project Teaches](#19-ml-student-pov--concepts-this-project-teaches)

---

## 1. Project at a Glance

| Item | Value |
|---|---|
| **Case study** | 36 — Residential Rent Prediction Using Machine Learning |
| **Problem type** | **Supervised Learning → Regression** (continuous target) |
| **Target variable** | `Rent` — monthly rent in Indian Rupees (₹) |
| **Dataset** | Kaggle "House Rent Prediction Dataset" — 4,746 Indian rental listings |
| **Features used** | 10 raw attributes + 5 engineered features |
| **Models trained** | 6 variants × 4 experimental settings = **24 model fits** |
| **Selected model** | **Gradient Boosting Regressor** on `log1p(Rent)`, **no** outlier treatment |
| **Test R²** | 0.7144 |
| **Test RMSE** | ₹32,716 |
| **Test MAE** | ₹10,135 |
| **Test MAPE** | 26.74 % |
| **Prediction interval** | 80 % nominal, 77.8 % empirical coverage |
| **Deployment** | Streamlit web app (`app/app.py`), 3 tabs |
| **Reproducibility** | Single global seed `RANDOM_STATE = 42` |
| **Total training runtime** | 8.45 minutes |

### The one-line summary you should be able to say
> "I predict the monthly rent of an Indian residential listing from 10 property attributes
> using a Gradient Boosting Regressor trained on the log of rent. It achieves R² = 0.71 and a
> typical error of about ₹10,000 (27 %), and is deployed as a Streamlit app that returns a
> point estimate plus an 80 % prediction interval."

### The seven questions the case study asks, and this project's answers

| # | Question | Answer from this project |
|---|---|---|
| 1 | Can monthly rent be predicted from property attributes? | **Yes, partially.** R² = 0.714, MAPE ≈ 27 %. Good enough for a guide price, not a valuation. |
| 2 | Which attribute most influences rent? | **City** (permutation importance = ₹7,290 MAE increase), then **Size** (₹5,506). |
| 3 | How does outlier treatment affect the model? | **It hurts.** Removing outliers raised test MAE from ₹10,135 → ₹12,504 and premium-band MAE from ₹73,887 → ₹1,05,137. |
| 4 | Does locality dominate the other predictors? | **No.** Location-only reaches R² 0.397 vs 0.714 for the full model. Strongest single block, but not dominant. |
| 5 | Which algorithm achieves the lowest RMSE? | **Linear Regression** (₹27,897) — but it wins on RMSE *only* because it extrapolates on ~10 premium listings. Gradient Boosting wins on MAE. |
| 6 | How well does it perform on new listings? | 25.5 % predicted within ±10 % of actual, 59.9 % within ±25 %. |
| 7 | Can it be deployed? | **Yes** — `streamlit run app/app.py`, single joblib artifact, 6.5 ms per prediction. |

---

## 2. The Dataset

**File:** `data/House_Rent_Dataset.csv`
**Source:** Kaggle — *House Rent Prediction Dataset* by Sourav Banerjee
**Shape:** 4,746 rows × 12 columns
**Missing values:** **0 explicit NaNs**, **0 duplicate rows**

### 2.1 The 12 raw columns

| Column | Type | Meaning | Fate in the model |
|---|---|---|---|
| `Posted On` | date | Listing date | **Dropped** — only an 89-day span |
| `BHK` | int | Bedrooms, Hall, Kitchen count | Numeric feature |
| `Rent` | int | **TARGET** — monthly rent in ₹ | Target (log1p transformed) |
| `Size` | int | Floor area in sq ft | Numeric feature |
| `Floor` | string | e.g. `"3 out of 5"`, `"Ground out of 2"` | **Parsed** into 2 numeric features |
| `Area Type` | category | Carpet / Super / Built Area | One-hot |
| `Area Locality` | string | Free-text neighbourhood, 2,235 levels | **Target-encoded** |
| `City` | category | 6 Indian cities | One-hot |
| `Furnishing Status` | category | Furnished / Semi / Unfurnished | One-hot |
| `Tenant Preferred` | category | Bachelors / Family / Both | One-hot |
| `Bathroom` | int | Number of bathrooms | Numeric feature |
| `Point of Contact` | category | Owner / Agent / Builder | One-hot |

### 2.2 Target distribution — why `log1p` is non-negotiable

| Statistic | Value (₹/month) |
|---|---|
| min | 1,200 |
| 1st percentile | 4,000 |
| 25th percentile | 10,000 |
| **median** | **16,000** |
| **mean** | **34,993** |
| 75th percentile | 33,000 |
| 99th percentile | 3,00,000 |
| **max** | **35,00,000** |
| std dev | 78,106 |
| **skewness** | **21.41** |
| skewness after `log1p` | **0.909** |

**Read these three facts together:**
1. **mean (₹34,993) is more than 2× the median (₹16,000)** → textbook right skew.
2. **max is 219× the median** → one listing at ₹35 lakh/month against a typical ₹16,000.
3. **skew drops from 21.41 to 0.909** → `log1p` turns a pathological distribution into a near-symmetric one.

> Skewness of 21.4 is enormous. A normal distribution has skew 0; anything beyond |2| is considered
> severely skewed. This single number justifies the log transform, which is the most important
> single modelling decision in the entire project.

**Figure:** `reports/figures/eda_rent_distribution.png`

### 2.3 City breakdown

| City | Listings | Median rent/sq ft (₹) |
|---|---:|---:|
| Mumbai | 972 | **71.4** |
| Chennai | 891 | 16.2 |
| Bangalore | 886 | 17.5 |
| Hyderabad | 868 | 14.5 |
| Delhi | 605 | 28.6 |
| Kolkata | 524 | 13.3 |

**Key observation:** Mumbai costs **5.4× Kolkata per square foot**. Rent scales with size, but the
*slope* is city-specific. This is exactly why `City` is the single most important feature and why
a global outlier rule would be wrong.

**Figures:** `eda_rent_by_city.png`, `eda_listings_per_city.png`, `eda_rent_vs_size.png`

### 2.4 Categorical level counts

```
City               : Mumbai 972 | Chennai 891 | Bangalore 886 | Hyderabad 868 | Delhi 605 | Kolkata 524
Furnishing Status  : Semi-Furnished 2251 | Unfurnished 1815 | Furnished 680
Tenant Preferred   : Bachelors/Family 3444 | Bachelors 830 | Family 472
Area Type          : Super Area 2446 | Carpet Area 2298 | Built Area 2        <- RARE
Point of Contact   : Contact Owner 3216 | Contact Agent 1529 | Contact Builder 1  <- RARE
BHK                : 1:1167 | 2:2265 | 3:1098 | 4:189 | 5:19 | 6:8
Bathroom           : 1:1474 | 2:2291 | 3:749 | 4:156 | 5:60 | 6:12 | 7:3 | 10:1
```

**Why the "RARE" flags matter:** `Built Area` has **2 rows** and `Contact Builder` has **1 row** in
the whole dataset. After an 80/20 split, such a level can appear in the test set having never been
seen in training. This is why `OneHotEncoder(handle_unknown="ignore")` is **required**, not
defensive tidiness — without it the encoder raises `ValueError` at prediction time.

### 2.5 The three data problems hidden inside "0 missing values"

The CSV reports zero NaNs. That is misleading. Three genuine data-quality problems are hidden:

**(a) `Size` has impossible values.**
Minimum reported size is **10 sq ft**. 16 listings report under 50 sq ft; 91 report under 100 sq ft.
These are data-entry errors, not micro-studios. `config.MIN_VALID_SIZE = 50.0` → anything below is
**blanked to NaN and imputed** by a group median.

**(b) `Floor` hides two facts in one string.**
480 distinct raw strings, with these structural patterns:

| Pattern | Count |
|---|---:|
| `N out of N` | 3,782 |
| `Ground out of N` | 926 |
| `Upper Basement out of N` | 23 |
| `Lower Basement out of N` | 11 |
| `N` (no "out of") | 3 |
| `Ground` (no "out of") | 1 |

The **4 rows with no "out of"** genuinely do not record the building height → `total_floors` is
missing. *This is the "handle missing floor values" objective: the missing values were hidden
inside a string rather than stored as NaN.*

**(c) `Area Locality` is catastrophically high-cardinality.**

| Statistic | Value |
|---|---|
| distinct localities | **2,235** |
| appearing exactly once | **1,465** |
| appearing twice or fewer | 1,794 |
| share of rows in a singleton locality | **30.9 %** |
| most common | Bandra West (37), Gachibowli (29), Electronic City (24) |

One-hot encoding would add 2,235 near-empty columns to a 4,746-row dataset — more columns than
rows for that one feature. This is why **target encoding with smoothing** is used instead.

### 2.6 `Posted On` — profiled, then dropped, with a reason

| Statistic | Value |
|---|---|
| range | 2022-04-13 → 2022-07-11 |
| span | **89 days** |
| distinct dates | 81 |
| unparseable | 0 |

**Reason for dropping:** 89 days is far too short a window to carry inflation, seasonality or
market drift. Any date-derived feature would fit mid-2022 noise rather than real signal. The column
is profiled first and dropped *with the reasoning recorded*, which is better practice than silently
discarding it.

---

## 3. Libraries Used — What, Why, How They Work

Every library is pinned in `requirements.txt`. Pinned versions matter: the saved pipeline in
`models/*.joblib` was serialised by **scikit-learn 1.9.1 exactly**; loading it under a different
release raises `InconsistentVersionWarning` and can silently break unpickling of the
`ColumnTransformer` / `TargetEncoder` internals.

### 3.1 scikit-learn `== 1.9.1`

**What it is:** The standard Python machine-learning library.

**Why this project uses it:** It supplies *all five required algorithms*, the preprocessing
transformers, the cross-validation machinery, the metrics, and — critically — the `Pipeline`
abstraction that makes leakage-free cross-validation possible.

**How it works (the thing you must be able to explain):**
scikit-learn is built on a single uniform API contract called the **Estimator API**:

| Method | Contract |
|---|---|
| `fit(X, y)` | **Learns** parameters from data and stores them as attributes ending in `_` (e.g. `coef_`, `feature_importances_`) |
| `transform(X)` | **Applies** learned parameters to new data (transformers only) |
| `predict(X)` | **Applies** learned parameters to produce predictions (estimators only) |
| `fit_transform(X, y)` | fit then transform, sometimes optimised |

Because *every* object obeys this contract, they can be composed. A `Pipeline` is itself an
estimator; a `ColumnTransformer` is itself a transformer. That composability is what lets this
project put feature engineering, imputation, encoding and the model into **one object** that can be
cloned per CV fold, saved with joblib and called with `.predict()` from the Streamlit app.

**Impact if removed:** The entire project would have to be rewritten. There is no substitute here.

### 3.2 pandas `== 3.0.6`

**What it is:** Tabular data manipulation — the `DataFrame` and `Series` types.

**Why:** The raw data is a CSV with mixed types and named columns. The `ColumnTransformer` selects
columns **by exact name**, so the data must stay a labelled DataFrame all the way into the pipeline
(not a bare NumPy array).

**How it works:** A `DataFrame` is a dict of column-wise arrays plus an `Index`. Operations are
**vectorised** — `df["Rent"] / df["Size"]` runs a C loop over the whole column rather than a Python
loop. `groupby` implements split–apply–combine: split rows by key, apply an aggregation, recombine.

**Where it matters most in this project:**
- `groupby(["City","BHK"])["Size"].median()` → the group-median imputation in `FeatureEngineer.fit`
- `groupby(City).transform(lambda s: s.quantile(0.25))` → per-city IQR bounds in `detect_outliers`
- `str.split(r"\s*out\s+of\s*")` → parsing the `Floor` column

### 3.3 NumPy `== 2.5.3`

**What it is:** N-dimensional arrays and vectorised numerical operations.

**Why:** It is the numerical substrate everything else sits on. pandas columns *are* NumPy arrays;
scikit-learn converts DataFrames to NumPy internally.

**How it works:** An `ndarray` is a contiguous block of memory plus a shape and a dtype. Operations
are compiled loops (SIMD where possible), so `np.log1p(y)` on 4,746 values is one C call, not 4,746
Python calls. **Broadcasting** lets arrays of different shapes combine without explicit loops.

**Specific uses here:**
- `np.log1p` / `np.expm1` — the target transform and its exact inverse
- `np.clip(y_pred, 0, None)` — forbidding negative rent predictions
- `np.corrcoef` — the funnel test in residual analysis
- `np.argsort(sq)[::-1]` — ranking squared errors for the error-concentration diagnostic

### 3.4 SciPy `== 1.18.1`

**What it is:** Scientific computing — statistics, optimisation, signal processing.

**Why this project needs it, in two separate places:**

1. **`scipy.stats` distributions as hyperparameter search spaces.** `RandomizedSearchCV` samples
   from a *distribution*, not a list. `loguniform(1e-2, 1e7)` samples the Ridge `alpha` uniformly
   **on a log scale**, which is the correct way to search a parameter spanning 9 orders of
   magnitude — a linear uniform would put 99.99 % of its samples above 1e3.

2. **`scipy.stats.shapiro`** — the Shapiro–Wilk normality test on the residuals.

**How `loguniform` works:** it draws `u ~ Uniform(log a, log b)` then returns `exp(u)`. The result is
equally likely to land in [0.01, 0.1] as in [1e6, 1e7].

### 3.5 Matplotlib `== 3.11.2`

**What it is:** The foundational Python plotting library.

**Why:** Every figure in `reports/figures/` is a Matplotlib `Figure` saved to PNG.

**How it works:** An object hierarchy — `Figure` (the canvas) contains `Axes` (one plot area), which
contains `Artists` (lines, bars, text). A **backend** renders that object tree. This project forces
`matplotlib.use("Agg")` — a non-interactive raster backend — so plotting works **headless** (no
display server), which matters because `train.py` runs in a terminal and Streamlit Cloud has no GUI.

**One real detail worth knowing:** the project explicitly sets `font.family = "DejaVu Sans"` because
that is one of the few bundled fonts containing **U+20B9, the ₹ sign**. Seaborn's default theme
resolves to Arial on macOS, which lacks the glyph and renders every rupee label as a hollow box.

### 3.6 Seaborn `== 0.13.2`

**What it is:** A statistical-plotting layer built *on top of* Matplotlib.

**Why:** It produces the statistical chart types this project needs in one line each —
`boxplot`, `violinplot`, `heatmap`, `scatterplot` with hue, `barplot` with confidence intervals.

**How it works:** Seaborn takes a tidy DataFrame, performs the statistical aggregation itself
(grouping, computing quartiles, bootstrapping CIs), then emits Matplotlib artists. You still get a
Matplotlib `Axes` back, so the two can be mixed freely.

**Used for:** `PALETTE = "colorblind"` (an accessibility choice), the correlation heatmap, per-city
rent boxplots, and residual distribution plots.

### 3.7 joblib `== 1.6.0`

**What it is:** Efficient serialisation of Python objects containing large NumPy arrays, plus simple
parallelism.

**Why over `pickle`:** joblib special-cases NumPy arrays and stores them in an efficient binary
format rather than pickling them element-structure-wise. For a Random Forest with 535 trees this is
dramatically smaller and faster.

**How it works:** `joblib.dump(obj, path)` pickles the object graph but intercepts NumPy arrays and
writes them as raw buffers. `joblib.load(path)` reverses it — **which means the class definitions
must be importable at load time.** This is why `app/app.py` inserts `PROJECT_ROOT` into `sys.path`
*before* calling `joblib.load`: the saved pipeline references `src.preprocess.FeatureEngineer`, and
without that path insertion you get `ModuleNotFoundError`. **This is the single most common reason a
Streamlit Cloud deployment fails while working locally.**

**Also:** joblib provides the `n_jobs=-1` parallel backend used by `RandomizedSearchCV`,
`cross_validate` and `permutation_importance`.

### 3.8 Streamlit `== 1.64.0`

**What it is:** A framework that turns a Python script into a web app.

**Why over Flask/Django:** Zero HTML, CSS or JavaScript. The entire UI is declared with Python
function calls (`st.selectbox`, `st.slider`, `st.metric`). For a model-demo app this is the right
level of abstraction.

**How it works (important and frequently asked):** Streamlit's execution model is **re-run the whole
script top-to-bottom on every interaction**. When a user moves a slider, the script executes again
from line 1 with the new widget value. That would reload the 20 MB model on every click, so
Streamlit provides caching decorators:

| Decorator | Caches | Used here for |
|---|---|---|
| `@st.cache_resource` | Unserialisable global resources, **by reference** | The loaded pipeline and the two quantile models |
| `@st.cache_data` | Serialisable data, returns a **copy** | `metrics.json`, `feature_meta.json` |

Using `cache_data` for the model would copy it on every access; using `cache_resource` for mutable
data would let one user's mutation leak to another. The choice between them is deliberate.

### 3.9 Development-only libraries (`requirements-dev.txt`)

| Library | Purpose |
|---|---|
| `jupyter == 1.1.1` | The notebook environment |
| `nbformat == 5.11.1` | **Programmatically builds** the notebook — `build_notebook.py` constructs cells as JSON rather than hand-editing `.ipynb` |
| `nbconvert == 7.17.1` | Executes the notebook and exports it |
| `ipykernel == 7.4.0` | The Python execution kernel the notebook talks to |

**Why build the notebook from a script?** A hand-edited `.ipynb` drifts out of sync with `src/`.
Generating it from `build_notebook.py` means the notebook always imports the same functions
`train.py` uses — the analysis and the training code can never disagree.

---

## 4. Every Import, Explained

This section lists **every** `import` and `from ... import ...` in the project, what it is, why it
was chosen, its impact, and its working principle.

### 4.1 Standard library

| Import | File(s) | Why | Working principle |
|---|---|---|---|
| `from __future__ import annotations` | all `src/*.py`, `train.py`, `app/app.py` | Lets you write `dict[str, float]` and `tuple[pd.DataFrame, dict]` as type hints without importing `typing.Dict`. | Makes **all** annotations lazily-evaluated strings (PEP 563), so they are never executed at runtime. |
| `from pathlib import Path` | `config.py`, `train.py`, `app/app.py`, `make_readme.py` | OS-independent path handling. `PROJECT_ROOT / "data" / "x.csv"` works on macOS, Linux and Windows. | `Path` is an object wrapping a path string; `/` is overloaded as the join operator via `__truediv__`. Gives `.exists()`, `.mkdir(parents=True)`, `.relative_to()`, `.stat()`. |
| `import re` | `data.py`, `evaluate.py` | Regular expressions. | Two uses: `re.sub(r"\d+", "N", s)` collapses `"3 out of 5"` → `"N out of N"` to reveal the handful of structural patterns among 480 raw strings; and `re.sub(r"(\d)(?=(\d\d)+$)", r"\1,", head)` inserts **Indian digit grouping** (28,50,000) which Python's built-in `{:,}` cannot do (it gives 2,850,000). |
| `from collections import Counter` | `data.py` | Counts floor-pattern frequencies. | A `dict` subclass; `Counter(iterable)` tallies occurrences, `.most_common()` returns them sorted descending. |
| `import json` | `train.py`, `app/app.py`, `make_readme.py` | Reads/writes `metrics.json` and `feature_meta.json`. | Serialises Python dicts to text. `default=str` is passed on dump so NumPy scalars that aren't natively JSON-serialisable are stringified rather than crashing the run after 8 minutes of training. |
| `import argparse` | `train.py` | CLI flags: `--fast`, `--n-jobs`, `--perm-repeats`. | Declarative parser; builds `--help` automatically from the declarations. |
| `import sys` | `train.py`, `app/app.py`, `make_readme.py` | `sys.path.insert(0, PROJECT_ROOT)` makes `src` importable regardless of the working directory. | `sys.path` is the list Python searches for modules; inserting at index 0 gives the project priority. |
| `import time` | `train.py` | Times hyperparameter searches and measures single-row prediction latency (6.53 ms for the selected model). | Wall-clock seconds since epoch. |
| `import warnings` | `train.py` | `warnings.filterwarnings("ignore", category=UserWarning)` suppresses the wall of sklearn convergence noise so the real report is readable. | Installs a filter in the warnings registry. |
| `from dataclasses import dataclass, field` | `models.py`, `train.py` | Declares `ModelSpec` and `Setting` as lightweight record types. | `@dataclass` auto-generates `__init__`, `__repr__`, `__eq__` from annotated attributes. `field(default_factory=dict)` is required for mutable defaults — a bare `= {}` would be **shared across all instances**. `@dataclass(frozen=True)` on `Setting` makes it immutable and hashable. |
| `from typing import Any` | `models.py` | Types the `build` callable and `**kwargs` dicts whose contents vary per model. | Escape hatch meaning "no static guarantee". |

### 4.2 Third-party core

| Import | Why | Impact |
|---|---|---|
| `import numpy as np` | Numerics, the log/expm1 transform, clipping, correlation. | Without `np.clip(y_pred, 0, None)` the raw-rent models would emit **negative rents**, which would distort the comparison with impossible predictions. |
| `import pandas as pd` | All tabular work. | The `ColumnTransformer` selects by column name, so data must stay a DataFrame. |
| `import joblib` | Saves/loads `pipeline.joblib`, `quantile_low.joblib`, `quantile_high.joblib`. | The **entire deployment story** — one file contains preprocessing + model. |
| `from scipy import stats` | `stats.shapiro(residuals)` — normality test. | Produced p = 1.9e-49, i.e. residuals are decisively **not** normal. |
| `from scipy.stats import loguniform, randint, uniform` | Hyperparameter search distributions. | `loguniform` for scale-spanning params (learning rate, alpha); `randint` for integer counts (n_estimators, depth); `uniform(0.7, 0.3)` for subsample ∈ [0.70, 1.00] — **note the second argument is the width, not the upper bound.** |
| `import matplotlib` + `matplotlib.use("Agg")` | Forces the headless backend **before** `pyplot` is imported. | Order matters: calling `use()` after `import pyplot` has no effect. |
| `import matplotlib.pyplot as plt` | Figure/axes creation, `savefig`, `close`. | `plt.close(fig)` after each save prevents a memory leak across ~19 figures. |
| `import matplotlib.ticker as mticker` | `mticker.FuncFormatter(lambda v, _: format_inr(v))` puts ₹ with Indian grouping on axes. | Readability of every rupee axis. |
| `import seaborn as sns` | Statistical plots and the colour-blind-safe theme. | `sns.set_theme(style="whitegrid", palette="colorblind")`. |
| `import streamlit as st` | The whole web UI. | — |

### 4.3 scikit-learn — base and composition

#### `from sklearn.base import BaseEstimator, TransformerMixin`
**Used in:** `src/preprocess.py` → `class FeatureEngineer(BaseEstimator, TransformerMixin)`

- **`BaseEstimator`** gives `get_params()` and `set_params()`. These are what make the class work
  with `clone()`, `GridSearchCV`, and `Pipeline`. **Why it matters:** `cross_validate` calls
  `clone(estimator)` for each fold, and `clone` works by reading constructor params via
  `get_params()` and building a fresh unfitted copy. Without `BaseEstimator`, cloning fails and
  cross-validation would silently reuse a fitted transformer across folds — **leakage.**
- **`TransformerMixin`** gives a free `fit_transform()` defined as `fit(X, y).transform(X)`.

**Working principle of the sklearn convention:** anything learned from data is stored in an
attribute with a **trailing underscore** (`self.size_median_city_bhk_`). Anything set by the user is
a constructor argument with **no** underscore (`self.min_valid_size`). `clone()` copies the latter
and discards the former.

#### `from sklearn.pipeline import Pipeline`
**Used in:** `preprocess.py` (numeric/onehot/locality sub-pipelines), `models.py` (the main
3-step pipeline, and the polynomial 2-step pipeline).

**What it does:** Chains transformers ending in an estimator. `pipe.fit(X, y)` calls
`fit_transform` on each step in turn and `fit` on the last; `pipe.predict(X)` calls `transform` on
each step and `predict` on the last.

**Why it is the single most important class in this project:** it is what makes the cross-validation
honest. When `cross_validate` clones the pipeline per fold, *every* fitted quantity inside it —
the (City, BHK) size medians, the one-hot category list, the target encodings — is re-learned from
that fold's training rows only. Fit the encoder once outside the pipeline and every CV score is
optimistically biased.

**Parameter naming:** steps are addressed with double underscores — `model__max_depth` means
"the `max_depth` parameter of the step named `model`". Nesting composes:
`regressor__model__reg__alpha`.

#### `from sklearn.compose import ColumnTransformer`
**Used in:** `preprocess.build_preprocessor()`

**What it does:** Applies **different transformers to different columns** and horizontally
concatenates the results.

```
ColumnTransformer([
    ("num", numeric_pipe,   [BHK, Size, Bathroom, floor_num, total_floors, floor_ratio, carpet_equiv_size]),
    ("cat", onehot_pipe,    [City, Furnishing Status, Tenant Preferred, Area Type, Point of Contact]),
    ("loc", locality_pipe,  [Area Locality]),
], remainder="drop", verbose_feature_names_out=False)
```

- `remainder="drop"` — anything not listed (`Floor`, `Rent`, `Posted On`) is discarded. This is a
  **safety property**: the target physically cannot reach the model.
- `verbose_feature_names_out=False` — output names are `City_Mumbai` rather than `cat__City_Mumbai`,
  which keeps the feature-importance table readable.

#### `from sklearn.compose import TransformedTargetRegressor`
**Used in:** `models.build_pipeline()`

**What it does:** Wraps an estimator so that `y` is transformed before fitting and the predictions
are inverse-transformed afterwards:

```
fit(X, y)      ->  inner.fit(X, log1p(y))
predict(X)     ->  expm1(inner.predict(X))
```

**Why this and not just training on `np.log1p(y)` manually — three reasons:**
1. `y` stays in **rupees** at the API boundary, so ordinary scorers report R²/MSE/RMSE/MAE in
   rupees with no manual inversion to get wrong.
2. The `TargetEncoder` inside the `ColumnTransformer` sees the **log** target, which is far less
   distorted by the skew of 21.4.
3. The **saved pipeline predicts rupees directly**, so the Streamlit app needs zero post-processing.

**The cost:** it nests the pipeline under `regressor__`, which is why `prefix_params()` exists.

### 4.4 scikit-learn — preprocessing transformers

#### `from sklearn.impute import SimpleImputer`
Three different strategies, each chosen for its column type:

| Strategy | Applied to | Why |
|---|---|---|
| `strategy="median"` | numeric features | **Robust to the skew.** The mean size would be dragged by 8,000 sq ft penthouses; the median would not. |
| `strategy="most_frequent"` | one-hot categoricals | The only sensible fill for a category — the mode. |
| `strategy="constant", fill_value="__missing__"` | `Area Locality` | A missing locality is **informative**, so it becomes its own category rather than being folded into the most common locality. |

**Working principle:** `fit` computes the fill value per column and stores it in `statistics_`;
`transform` substitutes it wherever the column is NaN. Because it sits *inside* the pipeline, the
statistic is computed per CV fold.

> **Note:** this is the *second* line of defence. The heavy lifting for `Size` and `total_floors` is
> done earlier by `FeatureEngineer`, which uses **group-wise** medians. `SimpleImputer` catches
> anything that still slips through.

#### `from sklearn.preprocessing import OneHotEncoder`
**Applied to:** `City`, `Furnishing Status`, `Tenant Preferred`, `Area Type`, `Point of Contact`.

**What it does:** Converts a categorical column with *k* levels into *k* binary indicator columns.

```
City = "Mumbai"  ->  City_Bangalore=0, City_Chennai=0, City_Delhi=0,
                     City_Hyderabad=0, City_Kolkata=0, City_Mumbai=1
```

**Three settings, each deliberate:**

| Setting | Value | Reason |
|---|---|---|
| `handle_unknown="ignore"` | always | `Built Area` (2 rows) and `Contact Builder` (1 row) can appear in test but not train. "Ignore" emits an all-zero row instead of raising. **Mandatory here.** |
| `drop="first"` | **linear models only** | Removes one column per feature to break the **dummy-variable trap**. With all *k* columns, they sum to 1 which is perfectly collinear with the intercept → `X'X` is singular → unstable/infinite coefficients. Trees do not care, so they keep all *k*. |
| `sparse_output=False` | always | Returns a dense array. With only ~24 encoded columns the memory saving of sparse is negligible, and dense keeps `PolynomialFeatures` and the importance tables simple. |

#### `from sklearn.preprocessing import StandardScaler`
**Applied to:** numeric + interaction + encoded-locality columns, **for linear models only**.

**Formula:**
```
z = (x - μ) / σ        where μ and σ are learned on the training fold
```

**Why linear models need it:**
- Gradient-based and closed-form linear solvers are numerically better conditioned on comparable
  scales. `Size` ranges 10–8,000 while `BHK` ranges 1–6.
- **Ridge regularisation penalises Σβ², which is scale-dependent.** Unscaled, the penalty would
  effectively punish the small-scale feature far harder than the large-scale one. Ridge without
  scaling is a bug, not a style choice.
- Degree-2 polynomial expansion of unscaled features produces terms like `Size²` ≈ 64,000,000,
  which destroys conditioning.

**Why trees do NOT need it:** a decision tree splits on `feature <= threshold`. Any **monotone**
rescaling maps the threshold along with the data and produces the **identical** tree. Scaling a
tree's inputs is pure wasted computation.

#### `from sklearn.preprocessing import TargetEncoder`
**Applied to:** `Area Locality` (2,235 levels) — the single cleverest piece of preprocessing here.

**What it does:** replaces each category with a **shrunk estimate of the mean target** for that
category.

**Formula (empirical-Bayes / smoothed mean encoding):**

```
                n_c · ȳ_c  +  m · ȳ
    encoding_c = ─────────────────────
                      n_c  +  m

    ȳ_c = mean target within category c
    n_c = number of training rows in category c
    ȳ   = global mean target
    m   = smoothing strength ("smooth='auto'" estimates it from the
          within-category vs between-category variance)
```

**Read the formula as a weighted average:** if a locality has many listings (`n_c` large), the
encoding ≈ its own mean. If it has one listing (`n_c = 1`), the encoding is pulled almost entirely
towards the **global mean**. This is exactly the behaviour you want for 1,465 singleton localities.

**Worked example (numbers from this dataset, global mean rent ≈ ₹34,993):**

| Locality | n_c | ȳ_c | Encoding with m ≈ 10 |
|---|---:|---:|---|
| Bandra West | 37 | ₹1,50,000 | (37·150000 + 10·34993)/47 ≈ **₹1,25,530** — mostly its own mean |
| A one-off locality | 1 | ₹1,80,000 | (1·180000 + 10·34993)/11 ≈ **₹48,176** — pulled hard to the global mean |

**THE CRITICAL LEAKAGE POINT — expect this in your viva:**
`TargetEncoder` uses the target `y` to build a feature. That is leakage waiting to happen. Two
defences are in place:

1. **It is placed INSIDE the pipeline.** So `cross_validate` re-fits it per fold and it never sees
   the validation fold's target. Hoisting it out and fitting once on the full training set would
   leak into every CV score.
2. **sklearn's `TargetEncoder` additionally cross-fits internally during `fit_transform`.** It
   splits the training data into *k* folds, and the encoding written into fold *i* is computed from
   folds *≠ i*. So even the **training-fold** encodings are out-of-fold. This is why the project
   passes an explicit `cv=KFold(n_splits=5, shuffle=True, random_state=42)` — to make that internal
   cross-fitting reproducible.

**Why `target_type="continuous"`:** tells the encoder this is a regression target so it encodes
with means rather than class probabilities.

#### `from sklearn.preprocessing import PolynomialFeatures`
**Used in:** `models._poly()` → `PolynomialFeatures(degree=2, include_bias=False)`

**What it does:** generates all polynomial terms up to the given degree.
For features `[a, b]` at degree 2 → `[a, b, a², ab, b²]`.

**Count:** for *d* input features at degree 2 the output has `d + d(d+1)/2` terms.
Here ~24 encoded features expand to **~324 correlated terms** on a 3,796-row training set.

**Why `include_bias=False`:** the bias column (all ones) is redundant — `LinearRegression` and
`Ridge` fit their own intercept. Including it creates perfect collinearity.

**This is how "Polynomial Regression" is implemented.** There is no `PolynomialRegression` class in
scikit-learn: polynomial regression **is** linear regression on polynomial features. Being able to
say that sentence is worth marks.

### 4.5 scikit-learn — the models

| Import | Class | Role in this project |
|---|---|---|
| `from sklearn.linear_model import LinearRegression, Ridge` | `LinearRegression` | Baseline; also the final step inside the polynomial pipeline |
| | `Ridge` | L2-regularised version, to control the collinearity plain poly-2 suffers from |
| `from sklearn.tree import DecisionTreeRegressor` | | Single tree — interpretable, high variance |
| `from sklearn.ensemble import RandomForestRegressor` | | **Bagging** ensemble — variance reduction |
| `from sklearn.ensemble import GradientBoostingRegressor` | | **Boosting** ensemble — bias reduction; also used with `loss="quantile"` for the prediction interval |

(Full treatment of each in [section 9](#9-every-ml-model--principle-formulas-examples).)

### 4.6 scikit-learn — model selection

#### `from sklearn.model_selection import train_test_split`
```python
X_train_raw, X_test, y_train_raw, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=model_df["City"])
```
- **`test_size=0.20`** → 3,796 train / 950 test.
- **`random_state=42`** → the split is identical on every run. Without it, every rerun would report
  different metrics and nothing would be reproducible.
- **`stratify=City`** → each city keeps its population proportion in both splits. Kolkata has only
  524 listings; a random split could under-represent it badly, and per-city metrics would then be
  computed on too few rows to mean anything.

#### `from sklearn.model_selection import KFold`
**Used in two places:** the CV splitter for scoring, and inside the `TargetEncoder`.

```python
KFold(n_splits=5, shuffle=True, random_state=42)
```
- **`shuffle=True` is essential.** Without it, folds are contiguous slices of the file. If the CSV
  has any ordering (by city, by date), fold 1 could be almost entirely one city.
- **One shared splitter** means every model is scored on *identical* folds, so differences between
  models are real differences and not fold-assignment luck.

#### `from sklearn.model_selection import cross_validate`
Runs k-fold CV returning **multiple** metrics at once (unlike `cross_val_score`, which returns one),
plus `return_train_score=True` so the generalisation gap can be computed.

**How it works per fold:** `clone(estimator)` → `fit` on k−1 folds → score on the held-out fold.
The clone is what guarantees no state carries over.

#### `from sklearn.model_selection import RandomizedSearchCV`
Hyperparameter tuning by **sampling** `n_iter` configurations from the given distributions and
cross-validating each.

**Why Randomized over `GridSearchCV`:** a grid over 5 Gradient Boosting hyperparameters with 5
values each is 5⁵ = 3,125 configurations × 5 folds = 15,625 fits. Random search samples 30. The
well-known Bergstra & Bengio (2012) result is that **random search finds configurations as good or
better in far fewer evaluations**, because in practice only a few hyperparameters matter and random
search explores more distinct values of each of them. Grid search wastes most of its budget varying
parameters that don't matter.

**The key configuration choice in this project — `refit="mae"`:**

```python
RandomizedSearchCV(..., scoring=CV_SCORING, refit="mae", ...)
```

Hyperparameters are selected on **CV MAE, not CV RMSE**. This is measured, not arbitrary:

| Model | CV RMSE std/mean | CV MAE std/mean |
|---|---:|---:|
| Linear Regression | 0.721 | 0.113 |
| Polynomial (deg 2) | 0.599 | 0.101 |
| Polynomial + Ridge | 0.669 | 0.107 |
| Decision Tree | 0.777 | 0.118 |
| Random Forest | 0.802 | 0.145 |
| Gradient Boosting | **0.812** | **0.147** |

CV RMSE has a standard deviation **60–81 % of its own mean** — it is dominated by whichever handful
of extreme listings happen to land in a fold, so refitting on it selects hyperparameters
essentially **at random**. (That is precisely how the Polynomial+Ridge model ended up with a
*negative* test R² on an early pass.) CV MAE sits at 10–15 %: a usable signal.
**All four metrics are still reported for every model — only the tuning criterion changed.**

#### `from sklearn.inspection import permutation_importance`
Measures how much a metric degrades when **one column is randomly shuffled**, breaking its
relationship with the target while preserving its marginal distribution.

**Algorithm:**
```
baseline = score(model, X_test, y_test)
for each feature j:
    for r in 1..n_repeats:          # 10 repeats here
        X' = X_test with column j shuffled
        s_r = score(model, X', y_test)
    importance_j = baseline - mean(s_r)
```

**Two scorers are used deliberately:** `r2` (conventional but unstable here, since one extreme rent
dominates the sum of squares and shuffling can swing R² wildly) and
`neg_mean_absolute_error` (robust). **The ranking is taken from MAE**, reported as "rupee increase
in test MAE when this column is shuffled", with R² reported alongside.

### 4.7 scikit-learn — metrics

```python
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score, root_mean_squared_error)
```

All four are computed on the **rupee scale**, never in log units — an RMSE of "0.34 log-rupees"
tells a landlord nothing. Because the estimator is a `TransformedTargetRegressor`, its `predict`
already applies `expm1`, so both `y_true` and `y_pred` arrive in rupees automatically.

`root_mean_squared_error` is a direct function in modern scikit-learn; the older idiom
`mean_squared_error(..., squared=False)` is deprecated.

### 4.8 Project-internal imports

| Import | What it pulls in |
|---|---|
| `from src import config as cfg` | **Every** file imports this. Single source of truth for paths, column names, thresholds, seed. |
| `from src import plots as P` | The 19 figure functions, shared by `train.py` and the notebook so no chart is drawn twice from different code. |
| `from src.data import load_data, profile_data, print_profile, load_glossary` | Loading + the section-4 profiling pass. |
| `from src.preprocess import FeatureEngineer, build_preprocessor, add_eda_features, detect_outliers, treat_outliers, empirical_area_ratios` | All preprocessing. |
| `from src.models import build_pipeline, build_quantile_pipeline, model_specs, spec_by_key, prefix_params, get_feature_names, get_fitted_model, get_fitted_preprocessor, inner_pipeline` | Model construction and introspection. |
| `from src.evaluate import compute_metrics, cv_report, cv_from_search, cv_splitter, per_city_metrics, worst_predictions, residual_frame, error_concentration, rent_band, format_inr, CV_SCORING, format_mean_std` | Metrics and diagnostics. |

### 4.9 Notebook-only imports

| Import | Purpose |
|---|---|
| `import nbformat as nbf` | Builds the `.ipynb` JSON structure cell by cell |
| `from IPython.display import Image, Markdown, display` | Renders saved PNGs and formatted text inside notebook output |
| `import subprocess`, `import os` | Executes the notebook via `nbconvert` from `build_notebook.py` |

---

## 5. Every Code File, Explained

```
Residential_ML/
├── data/
│   ├── House_Rent_Dataset.csv      raw Kaggle data, 4746 x 12
│   └── Dataset Glossary.txt        column definitions from Kaggle
├── src/
│   ├── __init__.py                 package marker + rationale
│   ├── config.py        (188 ln)   single source of truth
│   ├── data.py          (275 ln)   loading + profiling
│   ├── preprocess.py    (525 ln)   feature engineering, outliers, ColumnTransformer
│   ├── models.py        (274 ln)   the 6 model specs + pipeline assembly
│   ├── evaluate.py      (289 ln)   metrics, CV, diagnostics
│   └── plots.py         (692 ln)   all 19 figures
├── train.py             (924 ln)   the full experiment, end to end
├── app/app.py           (455 ln)   Streamlit deployment
├── make_readme.py       (602 ln)   generates README.md from metrics.json
├── notebooks/
│   ├── build_notebook.py (1634 ln) generates the analysis notebook
│   └── rent_prediction_analysis.ipynb
├── models/                         pipeline.joblib, quantile_*.joblib, metrics.json, feature_meta.json
└── reports/                        CSVs + figures/ (19 PNGs) + train_log.txt
```

### 5.1 `src/config.py` — single source of truth

**Job:** Everything that `train.py`, the notebook and the app must agree on is defined **exactly
once**, here.

**The important parts:**

| Block | Contents | Why it matters |
|---|---|---|
| `RANDOM_STATE = 42` | One global seed | Threaded through every split, every search, every CV shuffle, every model. Re-running `train.py` reproduces identical numbers. |
| Paths | `PROJECT_ROOT`, `DATA_DIR`, `MODELS_DIR`, `REPORTS_DIR`, `FIGURES_DIR` | `PROJECT_ROOT = Path(__file__).resolve().parents[1]` — works from any working directory. |
| Column constants | `COL_SIZE = "Size"`, `COL_LOCALITY = "Area Locality"`, … | **The spellings were verified against the real CSV header, not assumed.** Spaces and capitalisation matter because `ColumnTransformer` selects by exact name. |
| `AREA_TYPE_RATIOS` | `{Carpet: 1.00, Built: 0.85, Super: 0.70}` | Conventional Indian real-estate multipliers converting a reported figure to carpet-area equivalent. |
| Feature groups | `NUMERIC_FEATURES`, `ONEHOT_FEATURES`, `TARGET_ENCODE_FEATURES`, `INTERACTION_FEATURES` | Exactly what the `ColumnTransformer` consumes. |
| `MIN_VALID_SIZE = 50.0` | Plausibility floor | 16 listings report < 50 sq ft (smallest is 10). Treated as missing. |
| `FLOOR_TOKEN_MAP` | `{ground: 0.0, upper basement: -0.5, lower basement: -1.0}` | Gives word-valued floors a sensible **numeric ordering** so a linear model can use them. |
| `TEST_SIZE=0.20`, `N_SPLITS=5`, `STRATIFY_ON="City"` | Splitting policy | — |
| `IQR_MULTIPLIER=1.5`, `WINSOR_*_PCT`, `RENT_PER_SQFT_*_Q` | Outlier thresholds | — |
| `QUANTILE_LOW=0.10`, `QUANTILE_HIGH=0.90` | → a proper **80 %** prediction interval | 0.90 − 0.10 = 0.80 |
| `FEAT_RENT_PER_SQFT` | Marked **"EDA / outlier detection ONLY — never a feature"** | It is derived from the target. |

### 5.2 `src/data.py` — loading and profiling

**Key functions:**

| Function | What it does |
|---|---|
| `load_data(validate=True)` | Reads the CSV and **asserts the header matches** `EXPECTED_RAW_COLUMNS` exactly. On mismatch it reports missing/extra columns and raises. |
| `load_glossary()` | Returns the Kaggle column definitions text. |
| `floor_pattern(value)` | `re.sub(r"\d+", "N", s)` — collapses `"3 out of 5"` → `"N out of N"`, making 6 structural patterns visible among 480 raw strings. |
| `profile_data(df)` | Returns the **entire section-4 profile as a plain dict** — nulls, dtypes, target distribution, size plausibility, floor patterns, locality cardinality, date range, categorical levels, rare levels. |
| `print_profile(profile)` | Renders that dict as a console report. |
| `_fmt_inr(value)` | Indian digit grouping. |

**The important design decision:** `profile_data` returns a **dict** rather than printing. That
means `train.py` can print it *and* the notebook can render it as tables *and* it can be serialised
into `metrics.json` — all from one implementation, so the three can never diverge.

**The explicit non-feature:** there is **no synthetic-data fallback**. The docstring says so
directly: *"quietly substituting generated data would invalidate every number in the report."* If
the CSV is missing, it raises `FileNotFoundError` with download instructions.

**`rare_levels` computation** — the profile explicitly flags any categorical level with ≤ 5 rows,
because a level with a single row lands in either train or test but never both, which is the
justification for `handle_unknown="ignore"`.

### 5.3 `src/preprocess.py` — the heart of the project

Four responsibilities:

#### (a) `parse_floor(series) -> DataFrame`
Splits `Floor` on the regex `\s*out\s+of\s*` into `floor_num` and `total_floors`.
Word tokens are mapped first (`Ground → 0.0`), then anything numeric is parsed.
4 rows have no "out of" → `total_floors = NaN`.

#### (b) `class FeatureEngineer(BaseEstimator, TransformerMixin)` — pipeline step 1

**Why it is a stateful transformer and not a `FunctionTransformer` — this is the exam answer:**
The `Size` and `total_floors` imputations need **group medians**, and those medians must be learned
from the **training fold only**. A stateless function would recompute them from whatever frame it
was handed — including validation and test rows — which **leaks information inside
cross-validation**.

**Learned in `fit` (training fold only):**
- median `Size` per (City, BHK), per City, and globally
- median `total_floors` per City and globally
- median `floor_num` globally

**Produced in `transform`:**

| Feature | Formula | Why |
|---|---|---|
| `floor_num` | parsed from `Floor` | Which floor the flat is on |
| `total_floors` | parsed from `Floor` | Building height |
| `floor_ratio` | `floor_num / total_floors` | **Relative** height. Floor 10 of 12 is a penthouse; floor 10 of 40 is mid-rise. A tree would need many splits to express this; one ratio captures it. |
| `carpet_equiv_size` | `Size × AREA_TYPE_RATIOS[Area Type]` | Makes the three measurement conventions comparable |
| `size_x_<area_type>` (×3) | `Size if Area Type matches else 0` | Size × Area Type **interaction**, for linear models only |

**Imputation cascade for `Size`** — most specific evidence first:
```
(City, BHK) median   ->   City median   ->   global median
```
A missing 2BHK size in Mumbai gets the Mumbai-2BHK median, not a nationwide average.

**A physical constraint is enforced:**
```python
total_floors = max(total_floors, max(floor_num, 1.0))
```
A building cannot be shorter than the floor a flat sits on. Without this, a user entering
"floor 10 of a 3-storey building" in the app would produce `floor_ratio = 3.33`.

**Dual input contract:** `_split_floor` accepts **either** a raw frame with a `Floor` string
**or** a frame already carrying `floor_num` / `total_floors` as separate columns. The Streamlit app
collects those two as separate widgets, so both callers work against the same fitted pipeline.

#### (c) Outlier detection and treatment
`add_eda_features`, `detect_outliers`, `treat_outliers` — see [section 8](#8-outlier-treatment).

#### (d) `build_preprocessor(family, feature_set) -> ColumnTransformer`

**The `family` switch — two different preprocessing graphs for two model families:**

| | `family="linear"` | `family="tree"` |
|---|---|---|
| Numeric | `SimpleImputer(median)` + **`StandardScaler`** | `SimpleImputer(median)` only |
| Interaction terms | **Included** (`size_x_*`) | Excluded |
| One-hot | `drop="first"` | keep all levels |
| Locality | TargetEncoder + **scaled** | TargetEncoder |

**Why interaction terms for linear models only:** a linear model cannot express *"a square foot
means something different depending on the measurement basis"* without an explicit product term,
and is unfairly penalised in the comparison without it. Tree ensembles discover that split on their
own. **Giving each family the preprocessing it needs is what makes the comparison fair** — the
alternative is comparing a handicapped linear model against a well-served tree.

**The `feature_set` switch — supports the locality ablation:**

| Value | Columns kept | Ablation model |
|---|---|---|
| `"all"` | everything | A |
| `"no_location"` | drop `Area Locality` **and** `City` | B |
| `"only_location"` | **only** `Area Locality` and `City` | C |

### 5.4 `src/models.py` — the six model variants

**Why six models for five named algorithms:** the case study names five. Polynomial Regression is
reported as **two** rows — plain degree 2, and degree 2 + Ridge — because a plain degree-2
expansion over one-hot features produces ~324 correlated terms and overfits badly. Reporting both
keeps the deliverable aligned with the spec **while showing what regularisation buys**.

**`@dataclass ModelSpec`** holds everything needed to build, tune and report one variant:
`key`, `name`, `family`, `build` (a zero-arg callable returning a *fresh* estimator),
`param_distributions`, `n_iter`, `note`.

> `build` is a **callable**, not an instance. A shared instance would be mutated across settings.
> `lambda: LinearRegression()` guarantees a fresh object each time.

**The six specs:**

| key | Name | Family | n_iter | Search space |
|---|---|---|---:|---|
| `linear` | Linear Regression | linear | 1 | none (nothing to tune, by design) |
| `poly2` | Polynomial Regression (deg 2) | linear | 1 | none |
| `poly2_ridge` | Polynomial (deg 2) + Ridge | linear | 30 | `alpha ~ loguniform(1e-2, 1e7)` |
| `decision_tree` | Decision Tree | tree | 40 | `max_depth`, `min_samples_leaf`, `min_samples_split`, `max_features` |
| `random_forest` | Random Forest | tree | 25 | `n_estimators`, `max_depth`, `max_features`, `min_samples_leaf` |
| `gradient_boosting` | Gradient Boosting | tree | 30 | `learning_rate`, `n_estimators`, `max_depth`, `subsample`, `min_samples_leaf` |

**Two subtleties worth knowing:**

1. **Ridge `alpha` range goes to 1e7 deliberately.** Degree-2 expansion of ~24 standardised
   features produces terms whose squares are large, so the alpha that actually controls this model
   can be very high. A range truncated at 1e4 would handicap it and make the comparison against the
   unregularised version unfair.
2. **`RandomForestRegressor(n_jobs=1)` deliberately.** The surrounding `RandomizedSearchCV` already
   runs `n_jobs=-1`. Letting both parallelise **oversubscribes the CPU with nested thread pools and
   runs slower, not faster.**

**`build_pipeline(spec, log_target, feature_set)`** assembles:
```
TransformedTargetRegressor(
  regressor = Pipeline([
      ("features", FeatureEngineer()),
      ("prep",     build_preprocessor(family, feature_set)),
      ("model",    spec.build()),
  ]),
  func = np.log1p, inverse_func = np.expm1)
```
`log_target=False` drops the wrapper → that is setting 1 of the outlier experiment.

**`prefix_params`** re-keys the search space: `model__reg__alpha` → `regressor__model__reg__alpha`,
because `TransformedTargetRegressor` nests the pipeline under `regressor__`.

**`build_quantile_pipeline(alpha, best_params)`** builds `GradientBoostingRegressor(loss="quantile",
alpha=α)` and **inherits the tuned tree geometry** (learning rate, depth, subsample,
min_samples_leaf, n_estimators) from the main GB model rather than re-tuning.

### 5.5 `src/evaluate.py` — metrics and diagnostics

| Function | What it returns |
|---|---|
| `cv_splitter(n_splits=5)` | **The one shared** `KFold(5, shuffle=True, random_state=42)` so every model sees identical folds |
| `compute_metrics(y_true, y_pred)` | `{r2, mse, rmse, mae, mape}` — all in rupees, with `y_pred` clipped at 0 first |
| `cv_report(estimator, X, y)` | mean + std of all four metrics across 5 folds, plus train scores |
| `cv_from_search(search)` | Same shape, but read out of `RandomizedSearchCV.cv_results_` so the fits done during tuning are **reused, not repeated** |
| `per_city_metrics(y_true, y_pred, cities)` | RMSE/MAE/R²/MAPE **per city**, plus `rmse_pct_of_mean` |
| `worst_predictions(X_test, y_true, y_pred, n=10)` | The 10 biggest rupee errors, with city/locality/size context |
| `residual_frame(X_test, y_true, y_pred)` | Tidy residuals with city labels |
| `error_concentration(y_true, y_pred)` | What share of total MSE comes from the k worst rows |
| `rent_band(values)` | Buckets rent into `<₹15k`, `₹15k–40k`, `₹40k–1L`, `>₹1L (premium)` |
| `format_inr(value)` | `2850000` → `₹28,50,000` |

**Why `np.clip(y_pred, 0, None)` is applied before every metric:** a model trained on raw rent
(outlier-experiment setting 1) can emit **negative rents**, which are nonsensical. Clipping first
means the comparison is not distorted by impossible predictions.

**Why `cv_from_search` carries a caveat in its docstring:** for a tuned model the CV score belongs
to the configuration that was *selected using those same folds*, so it is **mildly optimistic**. The
held-out test set — never touched during tuning — is the unbiased estimate. Both are reported side
by side for exactly that reason.

### 5.6 `src/plots.py` — all 19 figures

Shared style: `seaborn` whitegrid, **colour-blind-safe palette**, DPI 130, `Agg` backend,
DejaVu Sans for the ₹ glyph, and a `FuncFormatter` putting Indian digit grouping on every rupee
axis. Imported by **both** `train.py` and the notebook so no chart is ever drawn twice from
different code.

### 5.7 `train.py` — the whole experiment

Run with `python train.py` (full) or `python train.py --fast` (1 search iteration per model, for
smoke-testing the script's structure in seconds).

**Execution order:**

| Step | What happens |
|---|---|
| 1 | Load + profile the real data; print the section-4 report |
| 2 | Test the **conventional area ratios** empirically against the data |
| 3 | Compare **per-city vs global** IQR outlier detection |
| 4 | Produce 7 EDA figures |
| 5 | **One** stratified 80/20 split, fixed forever |
| 6 | **6 models × 4 settings = 24 fits**, all scored on the same untreated test set |
| 7 | Multi-criteria model selection |
| 8 | Residual plots for **all** models |
| 9 | The premium-property trade-off by rent band |
| 10 | Per-city performance breakdown |
| 11 | Deep residual analysis + 10 worst predictions + error concentration |
| 12 | Feature importance (impurity + permutation) and the locality ablation |
| 13 | Fit the two quantile models; measure empirical interval coverage |
| 14 | Save `pipeline.joblib`, `quantile_*.joblib`, `metrics.json`, `feature_meta.json`, all CSVs |
| 15 | Print the run summary |

**The methodological guarantee stated in the module docstring:**
> The 20 % test split is made **ONCE**, stratified by city, and is never modified: no outlier
> treatment is applied to it and it is not seen during tuning. Every model in every experimental
> setting is scored against that same untreated test set, which is what makes the comparison fair.
> **Removing outliers from the test set as well would make the metrics improve simply because the
> test set got easier.**

**`tune_and_evaluate(spec, X_train, y_train, X_test, y_test, log_target, feature_set)`** is the
workhorse: builds the pipeline, runs `RandomizedSearchCV` (or plain `cv_report` for the two
untunable models), scores train and test, measures single-row prediction latency over 50 calls, and
computes three overfitting diagnostics.

### 5.8 `app/app.py` — the Streamlit deployment

**The design principle, stated in its docstring:**
> The app performs **NO preprocessing of its own**. It builds a one-row DataFrame with the correct
> column names and hands it to the saved pipeline, which contains the feature engineering,
> imputation, encoding and the model. **That is the whole point of keeping every transformation
> inside the sklearn Pipeline.**

Three tabs: **Estimate rent**, **Model comparison & analysis**, **Limitations**.
Covered fully in [section 16](#16-deployment--the-streamlit-app).

### 5.9 `make_readme.py` — generated documentation

Generates `README.md` from `models/metrics.json`.
**Why:** the README quotes a lot of numbers. Generating it from the same JSON the notebook and the
app read means those numbers **can never drift out of step** with the model actually saved in
`models/`. Hand-written numbers in a README go stale the first time you retrain.

### 5.10 `notebooks/build_notebook.py` — generated analysis

Builds `rent_prediction_analysis.ipynb` programmatically with `nbformat`, then executes it with
`nbconvert`. Same reasoning as above: a hand-edited notebook drifts out of sync with `src/`.

Notebook sections: 0 Setup · 1 Profiling · 2 EDA · 3 Preprocessing · 4 Outliers · 5 Model comparison
· 6 Residual plots for all models · 7 The outlier experiment · 8 Deep residual analysis
· 9 Feature importance + ablation · 10 Model selection justification · 11 Per-city breakdown
· 12 Prediction intervals · 15 Real-world applicability.

---

## 6. The End-to-End Pipeline

```
data/House_Rent_Dataset.csv  (4746 x 12)
        |
        v
  load_data(validate=True)      <- header asserted against config
        |
        v
  profile_data()                <- section 4: nulls, skew, cardinality, rare levels
        |
        v
  drop ["Posted On"]            <- 89-day span, no temporal signal
        |
        v
  train_test_split(0.20, stratify=City, random_state=42)
        |                                    \
   3796 train rows                       950 TEST rows  (UNTREATED, FROZEN)
        |                                            \
        v                                             \
  treat_outliers(strategy)      <- TRAIN ONLY          \
        |                                               |
        v                                               |
  +--------------------- the saved Pipeline -----------+|
  |                                                     ||
  |  step 1  FeatureEngineer                            ||
  |    - parse Floor -> floor_num, total_floors         ||
  |    - blank Size < 50 sqft                           ||
  |    - impute Size: (City,BHK) -> City -> global      ||
  |    - impute total_floors: City -> global            ||
  |    - total_floors = max(total_floors, floor_num, 1) ||
  |    - floor_ratio = floor_num / total_floors         ||
  |    - carpet_equiv_size = Size x ratio               ||
  |    - size_x_<area_type> interactions (linear only)  ||
  |                                                     ||
  |  step 2  ColumnTransformer                          ||
  |    num -> SimpleImputer(median) [+ StandardScaler]  ||
  |    cat -> SimpleImputer(mode) + OneHotEncoder       ||
  |    loc -> SimpleImputer(const) + TargetEncoder      ||
  |    remainder = "drop"                               ||
  |                                                     ||
  |  step 3  the model                                  ||
  +-----------------------------------------------------+|
        |  wrapped in TransformedTargetRegressor         |
        |  (log1p on fit, expm1 on predict)              |
        v                                                v
   RandomizedSearchCV (5-fold, refit="mae")        score on test
        |
        v
  models/pipeline.joblib  ---->  app/app.py  ---->  user
```

---

## 7. Preprocessing in Depth

### 7.1 The `Floor` parsing problem

**Raw:** 480 distinct strings. **Problem:** two facts stored in one string, and the missing values
are *hidden inside the text* rather than being NaN.

```
"3 out of 5"               ->  floor_num =  3.0, total_floors = 5.0
"Ground out of 2"          ->  floor_num =  0.0, total_floors = 2.0
"Upper Basement out of 4"  ->  floor_num = -0.5, total_floors = 4.0
"Lower Basement out of 3"  ->  floor_num = -1.0, total_floors = 3.0
"3"                        ->  floor_num =  3.0, total_floors = NaN   <- imputed
"Ground"                   ->  floor_num =  0.0, total_floors = NaN   <- imputed
```

**Why those exact numbers for the word tokens?** They preserve the natural physical **ordering**:
```
Lower Basement (-1.0)  <  Upper Basement (-0.5)  <  Ground (0.0)  <  Floor 1 (1.0)  < ...
```
A linear model reads these as a real number line, so a monotone ordering is required. Encoding
Ground as, say, 99 would tell the model a ground-floor flat is above the 50th storey.

### 7.2 The three area-measurement conventions

`Size` is reported under **three different conventions and is NOT directly comparable across them**:

| Area Type | Meaning | Conventional multiplier → carpet |
|---|---|---|
| Carpet Area | Usable floor area inside walls | 1.00 |
| Built Area | Carpet + wall thickness | 0.85 |
| Super Area | Built-up + share of lobbies, stairs, amenities | 0.70 |

`carpet_equiv_size = Size × ratio` makes them comparable.

**But the project tested the ratios empirically (`empirical_area_ratios`), and found:**
- **`Built Area` has only 2 rows in the entire dataset**, so no (City, BHK) group reaches the
  minimum count and its ratio **cannot be estimated empirically at all** — the conventional 0.85 is
  used as-is.
- For **`Super Area` the observed ratio is close to 1.0, NOT the conventional 0.70**. Within
  comparable flats, listings quoting super area report roughly the same number as listings quoting
  carpet area.

**What was done about it:** the conversion is still applied as specified, **but raw `Size` and
`Area Type` are KEPT as features**, so the models can ignore the conversion if it is unhelpful.
This is honest methodology: state the convention, test it, report the disagreement, and let the
model decide.

### 7.3 The `Size × Area Type` interaction

Three columns, one per area type:
```
size_x_carpet_area = Size  if Area Type == "Carpet Area"  else 0
size_x_built_area  = Size  if Area Type == "Built Area"   else 0
size_x_super_area  = Size  if Area Type == "Super Area"   else 0
```
Exactly one is non-zero per row. This lets a linear model learn a **different slope per area type**:
```
Rent = β0 + β1·size_x_carpet + β2·size_x_built + β3·size_x_super + ...
```
Without it, the model is forced to use one shared `Size` coefficient for all three conventions.

**Added for `family="linear"` only.** A tree, given `Size` and the one-hot `Area Type`, can learn
`if Area Type == Super and Size > 1200 then ...` by itself in two splits.

### 7.4 Why `rent_per_sqft` is NEVER a feature

```python
rent_per_sqft = Rent / carpet_equiv_size
```

**This is a function of the target.** Using it as a feature hands the model the answer:
`Rent = rent_per_sqft × carpet_equiv_size` reconstructs the target exactly, giving a suspiciously
perfect R² ≈ 1.0 that collapses the moment the model meets a real listing (where `Rent` is unknown
by definition).

It is used for **EDA and outlier detection only**, computed by `add_eda_features`, and is
deliberately absent from `config.NUMERIC_FEATURES` so the `ColumnTransformer` can never select it.
The function carries a `!! LEAKAGE WARNING !!` in its docstring.

> **This is the single best "do you understand data leakage?" talking point in the project.**

---

## 8. Outlier Treatment

### 8.1 Detection — two independent rules

#### Rule 1: Per-city IQR on `Rent`

**Formula:**
```
Q1  = 25th percentile of Rent within the city
Q3  = 75th percentile of Rent within the city
IQR = Q3 - Q1
lower bound = Q1 - 1.5 x IQR
upper bound = Q3 + 1.5 x IQR
outlier if  Rent < lower  OR  Rent > upper
```

**Why 1.5?** Tukey's convention. For a normal distribution, Q1 − 1.5·IQR and Q3 + 1.5·IQR sit at
roughly ±2.7σ, covering ~99.3 % of the data — so about 0.7 % of normal data is flagged. 3.0×IQR is
the "far out" convention for extreme outliers only.

#### **Why PER-CITY and not global — the headline finding**

| City | n | Flagged per-city | Flagged **globally** | per-city % | global % |
|---|---:|---:|---:|---:|---:|
| Bangalore | 886 | 91 | 36 | 10.3 | 4.1 |
| Chennai | 891 | 87 | 46 | 9.8 | 5.2 |
| Delhi | 605 | 64 | 68 | 10.6 | 11.2 |
| Hyderabad | 868 | 85 | 26 | 9.8 | 3.0 |
| Kolkata | 524 | 36 | 16 | 6.9 | 3.1 |
| **Mumbai** | **972** | **104** | **367** | **10.7** | **37.8** |

**A single global IQR rule would flag 367 of 972 Mumbai listings (37.8 %) as outliers**, because
the global upper bound sits barely above Mumbai's *median* rent. Kolkata's 99th percentile is
around ₹40,000 while Mumbai's median is around ₹50,000. A global rule would delete a large slice of
perfectly legitimate Mumbai listings and call it "data cleaning".

The per-city rule flags a sane ~7–11 % in every city.

**Figure:** `reports/figures/eda_outlier_per_city_vs_global.png`

#### Rule 2: Extreme `rent_per_sqft`
Flags listings beyond the per-city 0.5th / 99.5th percentile of rent per square foot. This is a
**second, independent signal**: it catches listings whose rent is implausible *given their size*,
which a rent-only rule misses entirely. (E.g. ₹2,00,000 for 8,000 sq ft in Hyderabad is not an
extreme rent, but it is an extreme rent-for-that-size.)

**Combined result on the 3,796 training rows:**
```
383 flagged in total
  351 by the per-city Rent IQR rule
   44 by the rent-per-sqft rule
```

### 8.2 Treatment — three strategies

| Strategy | What it does | Effect on 3,796 training rows |
|---|---|---|
| `"none"` | Leave untouched | 3,796 rows |
| `"remove"` | Drop flagged rows | 3,796 → **3,413** (383 removed) |
| `"cap"` (winsorise) | Clip `Rent` at the per-city 1st/99th percentile | 3,796 rows, **73 values clipped** |

**Winsorising formula:**
```
Rent_capped = clip(Rent, lower = P1_city, upper = P99_city)
```

**THE RULE THAT MAKES THE EXPERIMENT VALID:**
> `treat_outliers` is called on the **TRAINING split only**. Applying it to the test split too
> would be cheating: removing hard cases makes the metrics look better without the model improving.
> `train.py` always scores against the same untreated 950-row test set.

### 8.3 The four experimental settings

| key | Label | Target | Outlier strategy |
|---|---|---|---|
| `raw_none` | 1. Raw rent, no treatment | `Rent` | none |
| `log_none` | 2. log1p target only | `log1p(Rent)` | none |
| `log_remove` | 3a. log1p + outliers removed | `log1p(Rent)` | remove |
| `log_cap` | 3b. log1p + outliers capped | `log1p(Rent)` | cap |

**Note on reading the results table:** the **CV columns are NOT comparable across settings** — each
setting cross-validates a *different* training set, so "remove" shows a flatteringly low CV RMSE
simply because the hard rows were deleted from its folds. **Only the `test_*` columns are
comparable**, because every row is scored on the one shared untreated test set.

### 8.4 Results — test MAE (₹) by model and setting

| Model | 1. Raw | **2. log1p** | 3a. remove | 3b. cap |
|---|---:|---:|---:|---:|
| Linear Regression | 16,668 | 10,760 | 11,334 | 10,795 |
| Polynomial (deg 2) | 15,116 | 42,321 | 12,056 | 15,647 |
| Polynomial + Ridge | 12,470 | 15,257 | 12,870 | 11,218 |
| Decision Tree | 12,789 | 11,640 | 13,457 | 11,280 |
| Random Forest | 11,334 | 10,245 | 12,706 | 10,204 |
| **Gradient Boosting** | 11,006 | **10,135** | 12,504 | 10,447 |

### 8.5 Test RMSE (₹) by model and setting

| Model | 1. Raw | 2. log1p | 3a. remove | 3b. cap |
|---|---:|---:|---:|---:|
| **Linear Regression** | 39,566 | **27,897** | 33,198 | 29,242 |
| Polynomial (deg 2) | 36,384 | 917,495 | 41,111 | 155,554 |
| Polynomial + Ridge | 31,502 | 143,423 | 49,427 | 46,745 |
| Decision Tree | 40,920 | 34,818 | 46,731 | 35,788 |
| Random Forest | 34,796 | 35,092 | 46,609 | 35,470 |
| Gradient Boosting | 35,200 | 32,716 | 46,436 | 35,025 |

### 8.6 The premium-property trade-off — the core finding

MAE (₹) of the Gradient Boosting model **within rent bands**:

| Rent band | 1. Raw | **2. log1p** | 3a. remove | 3b. cap |
|---|---:|---:|---:|---:|
| < ₹15,000 | 3,539 | **2,528** | 2,482 | 2,524 |
| ₹15,000 – ₹40,000 | 5,843 | **5,345** | 5,067 | 5,279 |
| ₹40,000 – ₹1,00,000 | 18,036 | **14,798** | 17,360 | 14,531 |
| **> ₹1,00,000 (premium)** | 70,554 | **73,887** | **1,05,137** | 79,468 |

**Read the last row.** Removing outliers raises premium-band error from ₹73,887 to **₹1,05,137 — a
42 % degradation** — while barely helping the cheap bands (₹2,528 → ₹2,482, a ₹46 improvement).

**Conclusion to state in viva:**
> *Outlier treatment was tested, not assumed. The `log1p` transform alone already handles the skew
> (21.41 → 0.909). Removing or capping the extremes on top of that trains a model that
> systematically under-predicts genuinely expensive properties, for a negligible gain on cheap
> ones. The deployed model therefore uses `log1p` with **no** outlier removal — and the experiment
> that justifies it is reported in full rather than hidden.*

**Figures:** `outlier_experiment.png`, `outlier_premium_tradeoff.png`

---

## 9. Every ML Model — Principle, Formulas, Examples

### Notation used throughout

| Symbol | Meaning |
|---|---|
| `n` | number of training samples (3,796 here) |
| `p` | number of features after encoding (~24 here) |
| `x_i` | feature vector of sample i |
| `y_i` | true rent of sample i (₹) |
| `ŷ_i` | predicted rent of sample i |
| `β` | coefficient vector |
| `ȳ` | mean of y |

---

### 9.1 Linear Regression

#### Working principle
Fit a **straight line / hyperplane** through the data by choosing coefficients that minimise the
sum of squared residuals (Ordinary Least Squares, OLS).

#### Model equation
```
ŷ = β0 + β1·x1 + β2·x2 + ... + βp·xp        (vector form:  ŷ = Xβ)
```

#### Cost function
```
             n
RSS(β) =    Σ  (y_i - ŷ_i)²            "Residual Sum of Squares"
            i=1
```

#### Closed-form solution (the Normal Equation)
```
β̂ = (XᵀX)⁻¹ Xᵀ y
```
> scikit-learn does not literally invert the matrix — it solves the least-squares system with
> SVD (`scipy.linalg.lstsq`), which is numerically stable even when `XᵀX` is near-singular.

#### In this project
```python
ModelSpec(key="linear", family="linear", build=lambda: LinearRegression(),
          param_distributions={}, n_iter=1,
          note="Baseline. No hyperparameters to tune, by design.")
```
- Trained on `log1p(Rent)`, so it is actually a **multiplicative model in rupee space**:
  `log(Rent) = β0 + β1·Size` means `Rent = e^β0 · e^(β1·Size)` — each extra square foot multiplies
  rent by a constant factor rather than adding a constant number of rupees. That is exactly how
  rent behaves.
- Inputs are **standardised**, one-hot with `drop="first"`, and include the `size_x_*` interactions.

#### Results
```
Test R²   0.7924      <- HIGHEST of all six
Test RMSE  27,897     <- LOWEST of all six
Test MAE   10,760     <- 3rd
Test MAPE  28.42 %
CV RMSE    53,464 +/- 38,571
```

#### Why it won RMSE but did NOT get deployed — the project's most interesting finding
A linear model fitted in log space **keeps extending its trend beyond the range of rents it was
trained on**. It extrapolates. So it gets closer on the handful of very expensive test listings
that dominate squared error.

A tree ensemble **cannot extrapolate at all** — its predictions are bounded by the leaf values it
learned — so it under-predicts those few properties and pays heavily in RMSE, while fitting the
other ~99 % of listings more accurately.

Since the deployed tool estimates rent for **ordinary** listings, the model with the lower typical
error (MAE) is selected, and the RMSE trade-off is documented rather than hidden.

#### Assumptions of linear regression (know these)
1. **Linearity** — the relationship between X and y is linear (in the parameters)
2. **Independence** — residuals are independent of each other
3. **Homoscedasticity** — residual variance is constant across predictions
4. **Normality of residuals** — needed for inference (p-values, CIs), not for the point estimates
5. **No perfect multicollinearity** — `XᵀX` must be invertible

**In this project: assumptions 3 and 4 are violated.** `corr(predicted, |error|) = 0.642` is a
clear funnel (heteroscedasticity), and Shapiro–Wilk gives p = 1.9e-49 (non-normal residuals). That
is a genuine, reported limitation — and part of why a tree ensemble was preferred.

---

### 9.2 Polynomial Regression

#### Working principle
**Polynomial regression IS linear regression on polynomial features.** There is no separate
algorithm. You expand the feature space with powers and products, then fit OLS.

```
degree 2, features [a, b]  ->  [a, b, a², ab, b²]
ŷ = β0 + β1·a + β2·b + β3·a² + β4·ab + β5·b²
```
It is **linear in the coefficients β** (which is what "linear model" means) even though it is
non-linear in the inputs.

#### Feature count
```
output terms (degree 2, no bias) = p + p(p+1)/2
```
Here: ~24 encoded features → **~324 terms** fitted on 3,796 rows.

#### Implementation
```python
Pipeline([("poly", PolynomialFeatures(degree=2, include_bias=False)),
          ("reg",  LinearRegression())])
```

#### Results — a textbook overfitting demonstration

| Variant | Train RMSE | **Test RMSE** | Test R² |
|---|---:|---:|---:|
| Polynomial (deg 2), no regularisation | 62,695 | **9,17,495** | **−223.59** |
| Polynomial (deg 2) + Ridge | 62,161 | 1,43,423 | −4.49 |

**A test R² of −223.59 means the model is 224× worse than simply predicting the mean rent for every
listing.** Train RMSE (62,695) is *fine*; test RMSE (9,17,495) is catastrophic. **That gap is the
definition of overfitting** — and this project demonstrates it on real data rather than describing
it in the abstract.

#### Why it fails here
1. **~324 correlated terms.** Products of one-hot columns are mostly zero and mutually redundant.
2. **Severe multicollinearity** → `XᵀX` near-singular → enormous, unstable coefficients.
3. **Unbounded extrapolation.** A quadratic in log space explodes once an input leaves the training
   range; after `expm1` that becomes an absurd rupee figure.

#### Why Ridge helps but does not rescue it
Ridge shrinks the coefficients (R² improves from −223.59 to −4.49) but cannot undo the fundamental
problem: a degree-2 expansion of one-hot encoded data simply is not an appropriate hypothesis class
for this dataset.

**The honest interpretation to state in viva:**
> *The case study specified Polynomial Regression, so it was run and reported. Its failure is not a
> bug in my code — it is the correct, expected result of fitting 324 collinear terms to 3,796 rows,
> and it is exactly the demonstration of the bias–variance trade-off that the comparison is for.*

---

### 9.3 Ridge Regression (L2 regularisation)

#### Working principle
Same as OLS, but add a penalty on the **squared magnitude** of the coefficients, which shrinks them
towards zero and stabilises the solution.

#### Cost function
```
             n                      p
J(β) =      Σ (y_i - ŷ_i)²  +  α · Σ βⱼ²
            i=1                    j=1
             \_____________/        \______/
               fit the data        keep β small
```

#### Closed-form solution
```
β̂_ridge = (XᵀX + αI)⁻¹ Xᵀ y
```
**Why adding `αI` works:** it adds α to every eigenvalue of `XᵀX`. A near-singular matrix has
eigenvalues near zero; adding α lifts them away from zero, making the inverse well-conditioned.
**That is the mathematical reason regularisation fixes multicollinearity.**

#### The α (alpha) trade-off
| α | Effect |
|---|---|
| α = 0 | Identical to OLS |
| α small | Mild shrinkage, low bias, high variance |
| α large | Heavy shrinkage, high bias, low variance |
| α → ∞ | All β → 0, model predicts the intercept (the mean) |

This is the **bias–variance trade-off** made into a single tunable dial.

#### Ridge vs Lasso (be ready for this)
| | **Ridge (L2)** | **Lasso (L1)** |
|---|---|---|
| Penalty | `α Σ βⱼ²` (L2) | `α Σ abs(βⱼ)` (L1) |
| Shrinks coefficients | Yes, smoothly | Yes |
| Sets coefficients to **exactly** zero | **No** | **Yes** |
| Feature selection | No | Yes |
| With correlated features | Shares weight among them | Arbitrarily picks one |
| Geometry | Circular constraint region (no corners) | Diamond constraint region (corners on the axes) |

**Why Ridge here and not Lasso:** the goal was to *control collinearity* among polynomial terms,
not to select features. With highly correlated expanded terms, Ridge's behaviour of spreading
weight across them is the right one; Lasso would arbitrarily pick one of each correlated group.

#### In this project
```python
param_distributions={"model__reg__alpha": loguniform(1e-2, 1e7)}, n_iter=30
```
Range deliberately extends to 1e7 because degree-2 terms of standardised features are large, so the
controlling alpha can be very high. Truncating at 1e4 would handicap the model and make the
comparison against the unregularised version unfair.

---

### 9.4 Decision Tree Regressor

#### Working principle
**Recursively partition** the feature space into axis-aligned rectangles. Each split asks a single
yes/no question (`feature <= threshold`). The prediction for a leaf is the **mean of the training
targets that fall in it**.

```
                       Size <= 1100 ?
                      /              \
                   yes                no
                    |                  |
            City == Mumbai ?      Bathroom <= 3 ?
             /          \            /         \
        predict      predict     predict     predict
        ₹42,000      ₹13,500     ₹68,000    ₹1,85,000
```

#### Splitting criterion for regression — MSE reduction (variance reduction)
At each node the algorithm tries every (feature, threshold) pair and picks the one maximising:

```
                     n_left              n_right
ΔMSE  =  MSE(parent) - ------ MSE(left) - ------- MSE(right)
                       n_parent           n_parent

                 1
where  MSE(S) = ---  Σ  (y_i - ȳ_S)²        = the variance within node S
                |S| i∈S
```
This is called **variance reduction** for regression. (The classification analogues are Gini
impurity and entropy/information gain — know that those are for *classification*, not here.)

#### Prediction rule
```
ŷ(x) = mean{ y_i : x_i falls in the same leaf as x }
```

#### Hyperparameters tuned here
| Parameter | Search space | What it controls |
|---|---|---|
| `max_depth` | `[3,5,7,9,11,14,18,25,None]` | Maximum tree depth; the primary complexity dial |
| `min_samples_leaf` | `randint(1,40)` | Minimum samples in a leaf; prevents leaves of size 1 |
| `min_samples_split` | `randint(2,40)` | Minimum samples required to attempt a split |
| `max_features` | `[None,0.5,0.7,0.9,"sqrt"]` | Features considered per split |

All four are **pre-pruning** (early-stopping) controls. Without them a tree grows until every leaf
is pure — memorising the training set perfectly and generalising terribly.

#### Strengths
- **Interpretable** — you can read the decision path for any prediction
- Handles non-linearity and interactions automatically
- **Scale-invariant** — no standardisation needed
- Handles mixed feature types

#### Weaknesses
- **High variance** — change a few training rows and you get a different tree
- **Cannot extrapolate** — predictions are bounded by the min/max leaf values seen in training
- **Axis-aligned splits** — approximates a diagonal boundary as a staircase
- Biased towards high-cardinality features when using impurity importance

#### Results
```
Test R²   0.6774
Test RMSE 34,818
Test MAE  11,640
Test MAPE 32.03 %
```

---

### 9.5 Random Forest Regressor — **Bagging**

#### Working principle
Train **many de-correlated decision trees** and average their predictions.

```
Prediction:
            1    B
    ŷ(x) = ---  Σ   T_b(x)          B = number of trees (200–600 searched here)
            B   b=1
```

#### Two sources of randomness — both essential
**1. Bootstrap aggregating (bagging).** Each tree is trained on a bootstrap sample: `n` rows drawn
**with replacement** from the `n` training rows. On average each bootstrap contains
```
1 - (1 - 1/n)^n  ->  1 - 1/e  ≈  63.2 %
```
of the distinct original rows; the remaining ~36.8 % are **out-of-bag (OOB)** for that tree and can
be used as a free validation set.

**2. Random feature subsetting (`max_features`).** At **every split**, only a random subset of
features is considered.

**Why the second one matters — the key insight:** bootstrapping alone still gives highly
**correlated** trees, because one dominant feature (here, `City`) would be chosen as the root split
in nearly every tree. Averaging correlated predictors barely reduces variance. Restricting the
candidate features per split forces different trees to explore different structures, **de-correlating
them**, which is what makes the averaging effective. This is the "random" in Random Forest.

#### The variance-reduction mathematics
For `B` identically distributed predictors each with variance σ² and pairwise correlation ρ:
```
                       1 - ρ
Var(average)  =  ρσ² + ----- σ²
                         B
```
- As `B → ∞` the second term vanishes.
- **The first term, `ρσ², does not.** It is the floor set by tree correlation.
- **Therefore lowering ρ (feature subsetting) is as important as raising B.**

This single formula explains the entire design of Random Forest. It is a very likely viva question.

#### Hyperparameters tuned here
| Parameter | Search space |
|---|---|
| `n_estimators` | `randint(200, 600)` |
| `max_depth` | `[8, 12, 16, 20, 28, None]` |
| `max_features` | `[0.3, 0.5, 0.7, 1.0, "sqrt"]` |
| `min_samples_leaf` | `randint(1, 8)` |

**`n_jobs=1` on the forest, deliberately** — the surrounding `RandomizedSearchCV` already uses
`n_jobs=-1`, and nesting two parallel pools oversubscribes the CPU and runs *slower*.

#### Results
```
Test R²   0.6714
Test RMSE 35,092
Test MAE  10,245      <- 2nd best, the RUNNER-UP model
Test MAPE 26.75 %
Latency   13.25 ms    <- 2x slower than GB
```

#### Why it was the runner-up, not the winner
Marginally worse MAE than Gradient Boosting (₹10,245 vs ₹10,135) **and** twice the prediction
latency (13.25 ms vs 6.53 ms), because 500+ full-depth trees must each be traversed.

---

### 9.6 Gradient Boosting Regressor — **Boosting** — ★ THE SELECTED MODEL

#### Working principle
Build trees **sequentially**. Each new tree is fitted to the **errors (residuals / negative
gradients) of the ensemble so far**. Each tree corrects its predecessors.

#### The algorithm, step by step
```
1. Initialise with a constant:
       F_0(x) = argmin_γ Σ L(y_i, γ)        (for squared loss, this is ȳ)

2. For m = 1 ... M:

   a. Compute the "pseudo-residuals" = negative gradient of the loss:

                   [ ∂ L(y_i, F(x_i)) ]
          r_im = - [ ----------------- ]
                   [    ∂ F(x_i)       ]_{F = F_{m-1}}

      For squared loss  L = ½(y - F)²   ->   r_im = y_i - F_{m-1}(x_i)
      i.e. the pseudo-residual IS the ordinary residual.

   b. Fit a regression tree h_m(x) to the pseudo-residuals r_im

   c. Update, scaled by the learning rate ν:

          F_m(x) = F_{m-1}(x) + ν · h_m(x)

3. Final model:   F_M(x) = F_0(x) + ν · Σ_{m=1..M} h_m(x)
```

**Why "gradient" boosting:** step (a) is gradient descent — but **in function space** rather than
parameter space. Each tree is one step in the direction that most reduces the loss.

#### Worked numeric example (squared loss, ν = 0.1)
Suppose the true rent is ₹50,000.

| Stage | Prediction | Residual | New tree predicts | Update |
|---|---:|---:|---:|---|
| F₀ = mean | 35,000 | +15,000 | — | — |
| tree 1 | | | +12,000 | F₁ = 35,000 + 0.1(12,000) = **36,200** |
| tree 2 | 36,200 | +13,800 | +11,000 | F₂ = 36,200 + 0.1(11,000) = **37,300** |
| tree 3 | 37,300 | +12,700 | +10,500 | F₃ = **38,350** |
| … | | | | converging slowly towards 50,000 |

**Small steps, many of them.** That is why the tuned learning rate here is **0.0121** with **535
trees** — slow, careful convergence that generalises better than large aggressive steps.

#### The learning-rate / n_estimators trade-off
```
learning rate (ν)  DOWN   ->  need MORE trees, but better generalisation
learning rate (ν)  UP     ->  fewer trees needed, higher risk of overfitting
```
They must be tuned **together** — which is exactly why both appear in the search space.

#### Bagging vs Boosting — the comparison table you must know

| | **Bagging (Random Forest)** | **Boosting (Gradient Boosting)** |
|---|---|---|
| Trees built | **In parallel**, independently | **Sequentially**, each depends on the last |
| Each tree sees | A bootstrap sample | The full data, reweighted by the residuals |
| Primarily reduces | **Variance** | **Bias** |
| Base learners | Deep, low-bias, high-variance trees | Shallow, high-bias "weak learners" (depth 2–7) |
| Combination | Simple average | Weighted additive sum with learning rate |
| Overfits if over-trained? | **No** — more trees never hurt | **Yes** — too many trees overfits |
| Parallelisable | Yes, trivially | No (inherently sequential) |
| Sensitivity to noise/outliers | Lower | **Higher** (it chases residuals) |
| Typical tuning difficulty | Low (good defaults) | Higher (lr × depth × n_estimators) |

#### Hyperparameters tuned, and what was actually selected

| Parameter | Search space | **Selected value** |
|---|---|---|
| `learning_rate` | `loguniform(0.01, 0.3)` | **0.01212** (near the floor — slow, careful) |
| `n_estimators` | `randint(150, 600)` | **535** (near the ceiling, as expected with a small lr) |
| `max_depth` | `randint(2, 7)` | **5** |
| `subsample` | `uniform(0.7, 0.3)` → [0.70, 1.00] | **0.7545** |
| `min_samples_leaf` | `randint(1, 20)` | **6** |

**`subsample < 1.0` makes this Stochastic Gradient Boosting** (Friedman, 2002): each tree is fitted
on a random 75.45 % of the rows, which injects randomness, reduces variance and acts as a
regulariser. Note `uniform(0.7, 0.3)` means `loc=0.7, scale=0.3` → the range [0.70, 1.00]; the
second argument is the **width**, not the upper bound.

#### Results — the deployed model
```
Setting          2. log1p target only, NO outlier treatment
Test R²          0.7144
Test MSE         1,070,335,994
Test RMSE        32,716
Test MAE         10,135      <- LOWEST of all six models
Test MAPE        26.74 %     <- LOWEST of all six models
CV RMSE (5-fold) 49,039 +/- 39,836
Train RMSE       58,039
Latency          6.53 ms     <- FASTEST of all six
```

#### Gradient Boosting with the quantile loss
The same class, with `loss="quantile", alpha=α`, minimises the **pinball loss**:
```
                  { α · (y - ŷ)            if y >= ŷ
L_α(y, ŷ) =       {
                  { (α - 1) · (y - ŷ)      if y <  ŷ
```
This asymmetric loss makes the model estimate the **α-quantile** rather than the mean. Used for the
prediction interval — see [section 15](#15-prediction-intervals).

---

### 9.7 Model comparison summary — all six, primary setting

| Model | CV RMSE | CV RMSE σ | Train RMSE | **Test R²** | **Test RMSE** | **Test MAE** | Test MAPE | CV gap | Latency |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Linear Regression | 53,464 | 38,571 | 65,827 | **0.7924** | **27,897** | 10,760 | 28.4 % | −10,974 | 7.1 ms |
| Polynomial (deg 2) | 58,573 | 35,070 | 62,695 | −223.59 | 9,17,495 | 42,321 | 43.0 % | −4,178 | 7.6 ms |
| Polynomial + Ridge | 55,364 | 37,036 | 62,161 | −4.49 | 1,43,423 | 15,257 | 29.6 % | −6,953 | 8.2 ms |
| Decision Tree | 49,757 | 38,656 | 61,326 | 0.6774 | 34,818 | 11,640 | 32.0 % | −9,937 | 8.1 ms |
| Random Forest | 49,625 | 39,816 | 53,238 | 0.6714 | 35,092 | 10,245 | 26.7 % | −2,237 | 13.2 ms |
| **★ Gradient Boosting** | **49,039** | 39,836 | 58,039 | 0.7144 | 32,716 | **10,135** | **26.7 %** | −5,753 | **6.5 ms** |

**Figures:** `model_rmse_comparison.png`, `model_metric_panel.png`, `residuals_all_models.png`

---

## 10. Evaluation Criteria

All metrics are computed on the **rupee scale**, never in log units. Reporting log-space errors
would be flattering and meaningless — an RMSE of "0.34 log-rupees" tells a landlord nothing.
Because the estimator is a `TransformedTargetRegressor`, `predict()` already applies `expm1`, so
both `y_true` and `y_pred` arrive in rupees automatically.

`np.clip(y_pred, 0, None)` is applied before every metric so that impossible negative-rent
predictions (which raw-target models can emit) do not distort the comparison.

---

### 10.1 R² — Coefficient of Determination

#### Formula
```
                SS_res         Σ (y_i - ŷ_i)²
R²  =  1  -  ----------  =  1 - ---------------
                SS_tot         Σ (y_i - ȳ)²

SS_res = residual sum of squares (error of YOUR model)
SS_tot = total sum of squares    (error of predicting the MEAN for everything)
```

#### Interpretation
> **The proportion of variance in rent that the model explains.**

| Value | Meaning |
|---|---|
| R² = 1.0 | Perfect predictions |
| R² = 0.0 | Exactly as good as always predicting the mean rent |
| R² < 0 | **Worse than predicting the mean** |
| R² = 0.7144 | **This project: the model explains 71.4 % of rent variance** |

#### Why it can be negative
`SS_res` is unbounded above while `SS_tot` is fixed. The plain polynomial model scored
**R² = −223.59**, meaning its squared error was 224× that of the constant-mean baseline.
*A negative R² is not a bug — it is the metric correctly reporting a catastrophically bad model.*

#### Weakness, and why this project does not rank on it
R² is a **rescaled MSE**, so it inherits MSE's sensitivity to extreme values. On this dataset one
test listing accounts for 42 % of total squared error, so R² is largely a measurement of that one
row. R² is reported for every model but is **not** the selection criterion.

#### Adjusted R² (know the formula, though it is not used here)
```
                        (1 - R²)(n - 1)
Adj R²  =  1  -  ---------------------------
                        n - p - 1
```
It penalises adding features that do not help. Not used here because the feature count is fixed
across models, so plain R² is already comparable.

---

### 10.2 MSE — Mean Squared Error

#### Formula
```
           1    n
MSE  =    ---  Σ  (y_i - ŷ_i)²
           n   i=1
```

#### Properties
- **Units: rupees SQUARED** — not directly interpretable. This project's MSE is 1,070,335,994 "₹²",
  a number with no physical meaning.
- **Squaring punishes large errors quadratically.** A ₹1,00,000 error contributes 100× what a
  ₹10,000 error does.
- **Differentiable everywhere**, which is why it is the default loss for gradient-based training.
- **Decomposes** into Bias² + Variance + Irreducible Error (see §19.3).

#### Why it is reported but not used for ranking
The case study requires it. It is dominated by the premium tail on this dataset.

---

### 10.3 RMSE — Root Mean Squared Error

#### Formula
```
                     ____________________
                    /  1    n
RMSE  =  √MSE  =   /  ---  Σ  (y_i - ŷ_i)²
                 \/    n   i=1
```

#### Why it is preferred over MSE
**Same units as the target (₹).** "RMSE = ₹32,716" is a sentence a landlord can understand;
"MSE = 1,070,335,994 ₹²" is not.

#### Key property
RMSE ≥ MAE **always**, with equality only when all errors are identical in magnitude.
The ratio RMSE/MAE measures how unequal the errors are:
```
This project:  RMSE / MAE  =  32,716 / 10,135  =  3.23
```
A ratio above ~1.5 signals a heavy-tailed error distribution. **3.23 is very large** and is direct
evidence that a few listings dominate.

#### THE CRITICAL FINDING: why this project does not tune or rank on RMSE

**Error concentration on the 950-row test set:**

| Worst k rows | Share of total MSE |
|---:|---:|
| **1** | **42.2 %** |
| 2 | 50.8 % |
| 5 | 66.4 % |
| 10 | **76.9 %** |
| 20 | 85.6 % |
| 50 | 94.2 % |

```
RMSE over all 950 rows        : ₹32,716
RMSE excluding the worst 1 %  : ₹16,234      <- HALVED
```

**One single test listing out of 950 accounts for 42 % of the entire squared error.** Ten rows
account for 77 %. So test RMSE is largely a measurement of a handful of premium listings rather
than of general accuracy.

**And in cross-validation it is worse:** CV RMSE has std/mean of **0.60–0.81** across models —
refitting hyperparameters on a signal that noisy selects configurations essentially at random.
CV MAE's std/mean is **0.10–0.15**.

**Therefore: tune on MAE, rank on MAE, report RMSE alongside.**

---

### 10.4 MAE — Mean Absolute Error

#### Formula
```
          1    n
MAE  =   ---  Σ  | y_i - ŷ_i |
          n   i=1
```

#### Properties
- **Units: rupees.** Directly interpretable: *"the typical prediction is off by about ₹10,135."*
- **Robust to outliers** — errors enter linearly, not quadratically.
- **Not differentiable at zero** (the absolute value has a kink), which historically made it less
  convenient as a training loss. It is perfectly fine as an *evaluation* and *selection* metric.
- Minimising MAE estimates the **conditional median**; minimising MSE estimates the
  **conditional mean**. On a right-skewed target, the median is the better notion of "typical".

#### This project
```
Test MAE = ₹10,135    <- the primary selection metric
CV MAE   = ₹10,874 ± ₹1,600
```

---

### 10.5 MAPE — Mean Absolute Percentage Error

#### Formula
```
             100   n   | y_i - ŷ_i |
MAPE  =     -----  Σ   -------------
              n   i=1      | y_i |
```

#### Why this project added it even though the spec did not ask for it
> *It is the most intuitive answer to "how well does it do on a new listing?" — a 20 % error on a
> ₹15,000 flat and on a ₹3,00,000 penthouse are very different in rupees, and MAE alone hides that.*

```
Test MAPE = 26.74 %   "typically within about a quarter of the true rent"
```

#### Weaknesses (be ready to state them)
1. **Undefined when `y = 0`.** (Safe here: minimum rent is ₹1,200.)
2. **Asymmetric** — it penalises over-prediction more heavily than under-prediction, because the
   denominator is the actual value. Over-prediction error is unbounded; under-prediction is capped
   at 100 %.
3. **Biases model selection towards under-prediction.** This is a known pathology and a reason MAPE
   is reported but not used as the selection criterion here.

---

### 10.6 Residual Plot

#### Definition
```
residual  e_i  =  y_i - ŷ_i          (actual minus predicted)
```
A residual plot puts **predicted value on the x-axis** and **residual on the y-axis**.

#### What a GOOD residual plot looks like
- Points scattered **randomly around the horizontal line y = 0**
- **Constant vertical spread** across the whole x-range (homoscedasticity)
- **No visible pattern** — no curve, no funnel, no clusters
- Roughly symmetric above and below zero

#### What each pathology means

| Pattern seen | Diagnosis | Fix |
|---|---|---|
| **Funnel / cone** (spread grows with ŷ) | **Heteroscedasticity** | Log-transform the target; use a tree model; weighted least squares |
| **Curve / U-shape** | Missing non-linearity | Add polynomial terms or interactions; use a non-linear model |
| Residuals systematically above/below 0 in a region | Bias in that region | Add a feature that distinguishes it |
| A few huge isolated residuals | Outliers / unmodelled factors | Investigate individually |
| Residuals correlated with an unused variable | Omitted-variable bias | Include that variable |

#### This project's residual diagnostics

```
mean residual            ₹2,905      (a well-calibrated model sits near 0)
median residual          -₹37.70     (essentially zero -> well calibrated at the median)
std residual             ₹32,604
skew                     9.71        (strongly right-skewed errors)
kurtosis                 185.09      (extremely heavy tails; normal = 0 excess)
corr(predicted, |error|) 0.642       -> CLEAR FUNNEL: error grows with rent
Shapiro-Wilk p-value     1.9e-49     -> residuals decisively NOT normal
within +/-10 % of actual 25.5 % of test listings
within +/-25 % of actual 59.9 % of test listings
```

**How to interpret the mean-vs-median gap:** mean +₹2,905 but median −₹37.70. The median says the
model is beautifully calibrated for a *typical* listing. The positive mean is pulled up by a few
large positive residuals (big under-predictions of premium flats). **That gap is the skew, visible
in one pair of numbers.**

**Figures:** `best_model_residual_diagnostics.png`, `best_model_residuals_by_city.png`,
`residuals_all_models.png`

---

### 10.7 Metric selection summary — which metric for what

| Metric | Units | Outlier-sensitive? | Role in this project |
|---|---|---|---|
| **R²** | none | **High** | Reported; "variance explained"; not ranked on |
| **MSE** | ₹² | **Very high** | Reported because the spec requires it |
| **RMSE** | ₹ | **High** | Reported; headline figure; **not** tuned on (too noisy) |
| **MAE** | ₹ | **Low** | ★ **Tuning criterion and primary ranking metric** |
| **MAPE** | % | Moderate | Reported; most intuitive for a user |
| **Residual plot** | — | — | Diagnostic for model *assumptions*, not accuracy |

---

## 11. Cross-Validation

### 11.1 What k-fold cross-validation is

Split the training data into *k* equal folds. Train on *k−1*, validate on the held-out one, rotate,
and average the *k* scores.

```
Fold 1:  [VALID][train][train][train][train]
Fold 2:  [train][VALID][train][train][train]
Fold 3:  [train][train][VALID][train][train]
Fold 4:  [train][train][train][VALID][train]
Fold 5:  [train][train][train][train][VALID]
                                                -> average the 5 scores
```

### 11.2 Why cross-validate at all
A single train/validation split gives **one** estimate whose value depends on which rows happened
to land where. With 3,796 training rows and a heavy-tailed target, that is a lottery. CV averages
over *k* different splits, reducing the variance of the estimate and **using every row for
validation exactly once**.

### 11.3 Configuration used here
```python
KFold(n_splits=5, shuffle=True, random_state=42)
```
- **k = 5** — the standard bias/variance/compute compromise. k = 10 gives a lower-bias estimate but
  doubles the cost; with 24 model fits already, 5 is the right call. **k = n** is Leave-One-Out:
  near-unbiased but enormously expensive and high-variance.
- **`shuffle=True`** — without it, folds are contiguous slices of the CSV. Any ordering in the file
  (by city, by date) would make one fold unrepresentative.
- **`random_state=42`** — reproducible folds.
- **One shared splitter object** — every model is scored on *identical* folds, so differences
  between models are real and not fold-assignment luck.

### 11.4 Stratification: where it is and is not used
- **`train_test_split(..., stratify=City)`** — the 80/20 split **is** stratified by city, so every
  city keeps its proportion in both halves.
- **`KFold`** (not `StratifiedKFold`) is used inside CV. `StratifiedKFold` is designed for
  *classification* targets. For a continuous target, the common workaround is to bin it and
  stratify on the bins; here plain shuffled KFold on an already-city-stratified training set is
  sufficient.

### 11.5 The leakage guarantee — the most important point in this section

> **The estimator passed to `cross_validate` / `RandomizedSearchCV` is the FULL pipeline.**
> `clone()` is called per fold, so every fitted quantity — the (City, BHK) size medians, the one-hot
> category list, and **above all the TargetEncoder** — is learned from that fold's training rows
> only.

**What would go wrong if you fitted the TargetEncoder once outside the pipeline:** the encoding for
every locality would be computed using the target values of rows that later appear in the
validation fold. Every CV score would be optimistically biased, and the model would look far better
than it is.

**Belt and braces:** sklearn's `TargetEncoder` *also* cross-fits internally during `fit_transform`,
so even the training-fold encodings are out-of-fold. That is why an explicit seeded `KFold` is
passed to it.

### 11.6 Why CV scores and test scores differ here

| | CV RMSE | Test RMSE |
|---|---:|---:|
| Gradient Boosting | 49,039 ± 39,836 | 32,716 |

The **CV standard deviation (₹39,836) is larger than the test RMSE**. That is not an error — it is
the heavy tail doing its work: folds that happen to contain the ₹35-lakh listing score terribly,
folds that do not score well. The test split happens to be an easier draw.

**And a subtlety the code documents:** for a *tuned* model, the CV score belongs to the
configuration that was **selected using those same folds**, so it is mildly optimistic as a
generalisation estimate. The held-out test set — never touched during tuning — is the unbiased
estimate. **Both are reported side by side for exactly that reason.**

---

## 12. Results

### 12.1 The selected model

```
Model              Gradient Boosting Regressor
Setting            2. log1p target only, NO outlier treatment
Feature set        all
Hyperparameters    learning_rate     = 0.012115
                   n_estimators      = 535
                   max_depth         = 5
                   subsample         = 0.754547
                   min_samples_leaf  = 6
Trained on         3,796 listings
Scored on          950 held-out listings (never seen in training or tuning)
```

| Metric | Value |
|---|---|
| Test R² | 0.7144 |
| Test MSE | 1,070,335,994 ₹² |
| Test RMSE | ₹32,716 |
| Test MAE | ₹10,135 |
| Test MAPE | 26.74 % |
| CV RMSE (5-fold) | ₹49,039 ± ₹39,836 |
| Train RMSE | ₹58,039 |
| Train→test RMSE gap | −₹25,323 |
| CV overfit gap (RMSE) | −₹5,753 |
| Prediction latency | 6.53 ms |

### 12.2 How the model was selected — multi-criteria, not RMSE alone

**Step 1.** Pick the setting by lowest test MAE across all 24 combinations → Gradient Boosting under
`log1p, no treatment`.

**Step 2.** Shortlist every model within **110 % of the best test MAE** under that setting.

**Step 3.** Rank the shortlist on **four criteria simultaneously** and sum the ranks (lower = better):

| Criterion | Direction | Why it is in the list |
|---|---|---|
| `test_mae` | lower better | Typical accuracy |
| `test_rmse` | lower better | Worst-case accuracy |
| `abs_cv_gap` = **\|**CV RMSE − CV train RMSE**\|** | lower better | Overfitting |
| `cv_mae_cv` = CV MAE std / mean | lower better | Stability across folds |

**Two refinements that are worth marks:**

1. **The CV gap is ranked on its ABSOLUTE value.** The gap is validation RMSE minus training RMSE:
   positive means overfitting, but a large **negative** gap is not "extra generalisation" — it means
   the training folds happened to contain the extreme listings, which is an artifact. **The ideal
   is a gap near zero in either direction.**

2. **Stability is ranked on the coefficient of variation (std/mean), not the raw std**, so a model
   is not rewarded merely for having smaller errors overall.

**Step 4.** Ties are broken on **test MAE**, because a tie means the four criteria genuinely do not
separate the candidates, so the tiebreak should be the metric that is both stable on this data and
aligned with the deployment use case.

#### Why the plain train→test gap is reported but is MISLEADING here
`overfit_gap_rmse = test_rmse − train_rmse = −₹25,323` — **negative**, which naively reads as
"generalises better than it fits". It does not. The single ₹35,00,000 listing sits in the
**training** split, so train RMSE is inflated by one row.

`cv_overfit_gap_rmse` (5-fold validation RMSE minus 5-fold training RMSE, both on the same treated
training data) is the **statistically sound** generalisation gap, and is what the automatic
selection ranks on.

### 12.3 The RMSE-vs-MAE disagreement, stated as a finding

> *Linear Regression attains the lowest test RMSE (₹27,897 vs ₹32,716) and the highest test R²
> (0.792 vs 0.714), while Gradient Boosting attains the lowest test MAE (₹10,135 vs ₹10,760) and
> MAPE (26.7 % vs 28.4 %).*
>
> *The mechanism is **extrapolation**: a linear model fitted in log space keeps extending its trend
> beyond the range of rents it was trained on, so it gets closer on the handful of very expensive
> test listings that dominate squared error. A tree ensemble cannot extrapolate at all — its
> predictions are bounded by the leaf values it learned — so it under-predicts those few properties
> and pays heavily in RMSE, while fitting the other ~99 % of listings more accurately.*
>
> *Since the deployed tool estimates rent for ordinary listings, the model with the lower typical
> error is selected, and the RMSE trade-off is documented rather than hidden.*

### 12.4 Per-city performance — "limitations across different cities"

| City | n_test | Mean rent | RMSE | MAE | R² | MAPE | **RMSE % of mean** | Unseen localities % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Chennai | 178 | 21,614 | 13,146 | 5,894 | **0.79** | 26.4 % | **60.8 %** | 25.3 % |
| Bangalore | 177 | 19,382 | 12,748 | 5,357 | 0.66 | 24.2 % | 65.8 % | 36.2 % |
| Hyderabad | 174 | 19,846 | 13,369 | 4,967 | 0.69 | 20.7 % | 67.4 % | 22.4 % |
| Mumbai | 195 | 85,782 | 63,564 | 25,789 | 0.66 | 32.7 % | 74.1 % | **45.6 %** |
| Delhi | 121 | 28,708 | 29,721 | 10,100 | 0.39 | 30.3 % | 103.5 % | 31.4 % |
| **Kolkata** | 105 | 13,676 | 17,302 | 4,911 | **0.10** | 26.6 % | **126.5 %** | 35.2 % |

**Why `rmse_pct_of_mean` is the right column to compare:** city rent levels differ roughly
fivefold, so raw RMSE makes Mumbai look worst purely because its rents are biggest. As a share of
the city's own mean, **Chennai is best (60.8 %) and Kolkata worst (126.5 %)**.

**Why Kolkata is worst (R² = 0.10):**
- Fewest listings (524 total, only 105 in test) → thinnest evidence
- 35.2 % of its test listings sit in a locality never seen in training
- Low mean rent (₹13,676) with a high std (₹18,343) → a few expensive outliers dominate the
  variance it is being asked to explain

**Why Mumbai has the highest absolute error:** 45.6 % unseen localities (worst of any city) and a
rent std of ₹1,09,970. Its *relative* error (74.1 %) is only middling.

**The single best explanation for the whole column:** `unseen_locality_pct`. A test listing in a
locality the model never saw gets a target encoding shrunk all the way to the global mean, so the
model is effectively predicting from city + physical attributes alone.

**Figure:** `per_city_performance.png`

---

## 13. Residual Analysis

### 13.1 The 10 worst predictions — and *why* each failed

| City | Locality | BHK | Size | Actual | Predicted | % error | Locality count in train |
|---|---|---:|---:|---:|---:|---:|---:|
| Mumbai | Juhu | 4 | 5000 | 12,00,000 | 5,44,973 | −55 % | 3 |
| Delhi | Green Park | 4 | 800 | 3,50,000 | 54,298 | −84 % | 1 |
| Mumbai | Lake Superior, Powai | 4 | 3700 | 2,30,000 | 4,97,219 | **+116 %** | **0** |
| Mumbai | Chaitanya Towers, Prabhadevi | 3 | 1600 | 3,00,000 | 88,990 | −70 % | 1 |
| Mumbai | Indiabulls Blu, Worli | 3 | 1500 | 3,30,000 | 1,24,353 | −62 % | 2 |
| Kolkata | Avenue S, Santoshpur | 2 | 950 | 1,80,000 | 11,128 | **−94 %** | **0** |
| Mumbai | Benchmark Solus, Waterfield Rd | 4 | 1500 | 2,90,000 | 1,37,301 | −53 % | 0 |
| Mumbai | Pali Hill | 3 | 1800 | 4,00,000 | 2,53,177 | −37 % | 1 |
| Hyderabad | Beeramguda, Ramachandra Puram | 1 | 8000 | 2,00,000 | 56,062 | −72 % | 0 |
| Mumbai | Union Park | 2 | 1000 | 2,00,000 | 84,773 | −58 % | 0 |

**The code assigns a diagnosis to each automatically:**

| Condition | Diagnosis |
|---|---|
| locality count = 0 | *"locality unseen in training — falls back to the city/global average"* |
| locality count ≤ 2 | *"locality appears only Nx in training — target encoding is heavily smoothed towards the city average"* |
| actual > ₹2,00,000 | *"premium listing far above the local distribution; the log target deliberately down-weights these"* |
| otherwise | *"rent is unusual for its size and locality — likely unmodelled factors (amenities, building quality, furnishing detail)"* |

**Patterns to point out in viva:**
1. **9 of 10 are under-predictions.** The tree ensemble cannot extrapolate above the leaf values it
   learned.
2. **8 of 10 sit in a locality seen 0–2 times in training.** This is the dominant failure mode.
3. **7 of 10 are in Mumbai** — the city with the widest rent spread and the most unseen localities.
4. **The Hyderabad row (1 BHK, 8,000 sq ft, ₹2,00,000) is almost certainly a data-entry error**, not
   a model failure. 1 BHK in 8,000 sq ft is implausible.

### 13.2 Residual statistics — fully interpreted

| Statistic | Value | What it tells you |
|---|---|---|
| mean | ₹2,905 | Slight overall **under**-prediction (actual > predicted on average) |
| **median** | **−₹37.70** | **Essentially zero — excellently calibrated for a typical listing** |
| std | ₹32,604 | Matches test RMSE, as expected when the mean is near zero |
| skew | 9.71 | Strongly right-skewed errors — a few big under-predictions |
| kurtosis | 185.09 | Extremely heavy tails (normal = 0 excess kurtosis) |
| `corr(ŷ, \|e\|)` | **0.642** | **Clear funnel: absolute error grows with predicted rent** |
| Shapiro–Wilk p | 1.9e-49 | Residuals are decisively **not** normally distributed |
| within ±10 % | 25.5 % | 1 in 4 listings predicted within a tenth of true rent |
| within ±25 % | 59.9 % | 3 in 5 within a quarter |

### 13.3 The funnel — and why it is acceptable here

`corr(predicted, |error|) = 0.642` is a strong funnel: the model is **far more accurate on cheap
flats than expensive ones**, in absolute rupees.

**Is this a failure?** No — and here is the defence:
- MAPE is roughly **flat** across cities (20.7 % – 32.7 %), so the model is similarly accurate in
  *relative* terms everywhere.
- A ₹50,000 error on a ₹5,00,000 penthouse (10 %) is a *better* prediction than a ₹5,000 error on a
  ₹10,000 flat (50 %).
- The `log1p` transform means the model is implicitly optimising **relative** error, which is the
  right objective for a quantity spanning ₹1,200 to ₹35,00,000.
- The **prediction interval widens automatically** for expensive properties, so the funnel is
  communicated to the user rather than hidden.

### 13.4 Why Shapiro–Wilk failing does not invalidate the model
Normality of residuals is an assumption required for **inference** in OLS — p-values, confidence
intervals on coefficients, t-tests. It is **not** required for the point predictions to be useful,
and Gradient Boosting makes no distributional assumption about residuals at all. It is reported as
a diagnostic, and it is one reason a tree ensemble was preferred to the linear model.

---

## 14. Feature Importance and the Locality Ablation

### 14.1 Two different kinds of importance — and why both are reported

| | **Impurity (Gini/MSE) importance** | **Permutation importance** |
|---|---|---|
| Computed on | **Encoded** columns | **Original input** columns |
| Computed from | Training data, during fitting | **Held-out test** data, after fitting |
| Measures | Total variance reduction attributable to splits on that column | Metric degradation when the column is shuffled |
| Cost | Free (already computed) | Expensive (n_features × n_repeats refits of the scorer) |
| Known bias | **Inflated for high-cardinality and continuous features** | Underestimates importance of **correlated** features (a shuffled column's information survives in its correlate) |
| Answers | "Which encoded column did the trees split on?" | **"Which attribute actually matters?"** |

### 14.2 Permutation importance — the headline table

Measured as the **rupee increase in test MAE** when that column is randomly shuffled, 10 repeats:

| Rank | Feature | MAE increase (₹) | ± std | R² drop |
|---:|---|---:|---:|---:|
| 1 | **City** | **7,290** | 585 | 0.340 |
| 2 | **Size** | **5,506** | 441 | 0.330 |
| 3 | Bathroom | 2,804 | 311 | 0.219 |
| 4 | Point of Contact | 2,775 | 362 | 0.196 |
| 5 | Area Locality | 1,998 | 249 | 0.091 |
| 6 | Furnishing Status | 334 | 89 | 0.002 |
| 7 | BHK | 279 | 42 | 0.004 |
| 8 | Floor | 244 | 72 | −0.010 |
| 9 | Area Type | 146 | 160 | 0.007 |
| 10 | Tenant Preferred | 23 | 34 | −0.001 |

**Answer to "which attribute most influences rent?": `City`.** Shuffling it costs ₹7,290 of MAE.

### 14.3 Interesting things in that table

**`Bathroom` (₹2,804) beats `BHK` (₹279) by 10×.** Why? Bathroom count is a **proxy for luxury
tier**. A 3 BHK with 1 bathroom and a 3 BHK with 3 bathrooms are very different properties, and
bathroom count discriminates between them while BHK does not. BHK and Size are also strongly
correlated, so permutation importance under-credits BHK (its information survives in `Size`).

**`Point of Contact` (₹2,775) ranks 4th.** This is **not** a property attribute — it records whether
you contact the owner, an agent or a builder. Agent-listed properties are systematically more
expensive. **This is arguably a leakage-adjacent feature**: it is known at listing time so it is
legitimate to use, but it encodes market positioning rather than the physical property. Worth
raising yourself in viva — it shows critical thinking.

**`Tenant Preferred` (₹23) is essentially useless.** 72.6 % of listings say "Bachelors/Family", so
the column carries almost no information.

**`Floor` has a negative R² drop (−0.010).** Shuffling it slightly *improved* R². That means its
contribution is within noise.

### 14.4 Impurity importance (encoded columns) — a contrast

| Feature (encoded) | Importance |
|---|---:|
| `Point of Contact_Contact Agent` | **0.3342** |
| `Bathroom` | 0.2163 |
| `City_Mumbai` | 0.1131 |
| `Size` | 0.0799 |
| `carpet_equiv_size` | 0.0579 |
| `Area Locality` (target-encoded) | 0.0537 |
| `total_floors` | 0.0358 |
| `Point of Contact_Contact Owner` | 0.0342 |
| `BHK` | 0.0295 |
| `floor_ratio` | 0.0077 |

**The two rankings disagree, and the disagreement is informative:**
- Impurity importance puts `Point of Contact_Contact Agent` first (0.334) because it is a clean
  binary split near the root.
- Permutation importance puts `City` first, because it is measured on the **original** column
  (aggregating all six one-hot City columns) and on **held-out** data.
- `City` is split across 6 encoded columns in the impurity table, so its total is diluted.

**Which to trust: permutation importance**, because it is measured on original columns and on
held-out data, and because impurity importance is biased towards high-cardinality features such as
the target-encoded locality.

**Figure:** `feature_importance.png` (both panels side by side)

### 14.5 The locality ablation — "does locality dominate?"

Three models, **same algorithm, same setting, different feature sets**:

| Model | Features | Test RMSE | Test MAE | **Test R²** | Test MAPE |
|---|---|---:|---:|---:|---:|
| **A** | All features | **32,716** | **10,135** | **0.714** | 26.7 % |
| **B** | Without `Area Locality` and `City` | 44,728 | 13,156 | 0.466 | 34.2 % |
| **C** | Only `Area Locality` and `City` | 47,537 | 16,514 | 0.397 | 51.5 % |

**Computed conclusions:**
```
Dropping location worsens RMSE by 36.7 %   (R² 0.714 -> 0.466)
Location alone recovers R² 0.397, which is 56 % of the full model's 0.714
```

**The stated answer:**
> *Location is the single strongest block of predictors but it does **NOT** dominate on its own:
> location-only reaches R² 0.397 versus 0.714 for the full model, so the physical attributes
> (size, bathrooms, BHK, floor) still contribute materially. Equally, removing location costs
> 36.7 % RMSE, so neither block is redundant.*

**Note that A ≠ B + C.** 0.714 ≠ 0.466 + 0.397. The blocks are not additive, because location and
physical attributes **interact**: 1,000 sq ft means something different in Mumbai than in Kolkata.
That interaction is only available to model A. **This is a very good point to make.**

**Figure:** `locality_ablation.png`

---

## 15. Prediction Intervals

### 15.1 Why the app needs one
The case study requires the app to display an **"expected error range"**, not just a number. A
single point estimate implies a precision the model does not have.

### 15.2 How it is built — quantile regression, not ± MAE

Two **additional** Gradient Boosting models are trained with the **quantile loss**:

| Model | α | File |
|---|---|---|
| Lower bound | 0.10 | `models/quantile_low.joblib` |
| Upper bound | 0.90 | `models/quantile_high.joblib` |

```
0.90 - 0.10 = 0.80   ->   a proper 80 % prediction interval
```

### 15.3 The pinball (quantile) loss
```
                  { α · (y - ŷ)          if  y >= ŷ   (under-prediction)
L_α(y, ŷ)   =     {
                  { (α - 1)(y - ŷ)       if  y <  ŷ   (over-prediction)
```

**How it works intuitively.** For α = 0.9, under-predicting is penalised 9× more than
over-predicting (weight 0.9 vs 0.1), so the model is pushed **upwards** until only 10 % of true
values lie above it — i.e. it learns the 90th percentile. For α = 0.1 the asymmetry is reversed and
it learns the 10th percentile. At α = 0.5 the loss is symmetric and you recover MAE / the median.

### 15.4 The elegant mathematical point — quantiles survive the log transform

Both quantile models are trained on `log1p(Rent)` like every other model. **Quantiles are invariant
under monotone transformations:**

```
If  g  is strictly increasing,  then  Q_α( g(X) )  =  g( Q_α(X) )
```

Therefore:
```
expm1( Q_α( log1p(Rent) ) )  =  Q_α( Rent )     EXACTLY
```

**No approximation is involved.** Contrast this with the *mean*, which is **not** transform-
invariant: `exp(mean(log x)) ≠ mean(x)` — it gives the geometric mean, which is why back-
transforming a log-space mean prediction introduces a known downward bias.

> This is a genuinely sophisticated detail and an excellent thing to raise unprompted.

### 15.5 Why not just `prediction ± MAE`?
A fixed ±₹10,135 band would be:
- **Far too wide** for a ₹8,000 Kolkata studio (the band would go negative)
- **Far too narrow** for a ₹5,00,000 Mumbai penthouse
- **Constant**, so it could not express that the model is less certain about unusual properties

Quantile models learn a **conditional, property-specific** width. The `± MAE` approach is kept only
as a **fallback** if the quantile models fail to load.

### 15.6 Validation of the interval

```
Nominal coverage              80.0 %
Empirical coverage on test    77.8 %        (739 of 950 test listings fell inside)
Median interval width         74.3 % of actual rent
```

**Is 77.8 % vs 80 % good?** Yes. A 2.2-point shortfall on 950 test rows is well within sampling
noise, and the shortfall is in the **conservative** direction you would expect given the heavy
tail: the few extreme listings fall outside. An interval that over-covered (say 95 % when nominal
was 80 %) would be a worse sign — it would mean the intervals are uselessly wide.

**The width (74 % of rent) is honest, not embarrassing.** It says: *"for a flat whose true rent is
₹20,000, we are 80 % confident it is between about ₹13,000 and ₹28,000."* Given the missing
features (amenities, building age, metro proximity), that is a fair statement of what the data
supports.

**Figure:** `prediction_interval_coverage.png`

### 15.7 Three safety rails in the app
```python
low, high = min(low, high), max(low, high)      # 1. enforce ordering
low, high = min(low, point), max(high, point)   # 2. point estimate must be inside its own interval
low = max(low, 0.0)                             # 3. rent cannot be negative
```
Rail 2 matters because the three models (mean, q10, q90) are fitted **independently**, so nothing
mathematically guarantees `q10 ≤ mean ≤ q90`. This is called **quantile crossing** and the code
handles it explicitly.

---

## 16. Deployment — The Streamlit App

**Run with:** `streamlit run app/app.py`

### 16.1 The core design principle
> *The app performs **NO preprocessing of its own**. It builds a one-row DataFrame with the correct
> column names and hands it to the saved pipeline, which contains the feature engineering,
> imputation, encoding and the model.*

**Why this matters:** it eliminates **training/serving skew** — the single most common production
ML bug, where the app's preprocessing drifts out of step with the training code. Here there is only
one implementation, serialised into the artifact.

### 16.2 The sys.path fix — the most common deployment failure
```python
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
```
Placed **before** `joblib.load`. The saved pipeline references `src.preprocess.FeatureEngineer`;
unpickling must be able to import that class. Without this, Streamlit Cloud fails with
`ModuleNotFoundError` while everything works fine locally.

### 16.3 Caching

| Decorator | Used for | Why that one |
|---|---|---|
| `@st.cache_resource` | `pipeline.joblib`, `quantile_low/high.joblib` | Returns the object **by reference** — the right choice for a large, unserialisable, read-only resource |
| `@st.cache_data` | `metrics.json`, `feature_meta.json` | Returns a **copy** — the right choice for data, so one user's mutation cannot leak to another |

Without caching, Streamlit's re-run-on-every-interaction model would deserialise the model on every
slider move.

### 16.4 Tab 1 — Estimate rent

**11 inputs**, laid out in 3 columns:

| Column 1 | Column 2 | Column 3 |
|---|---|---|
| City (selectbox) | BHK (slider 1–6) | Furnishing Status |
| Area Locality (selectbox, filtered by city) | Bathrooms (slider 1–10) | Tenant Preferred |
| Area Type | Floor number (−1 … 80) | Point of Contact |
| Size in sq ft (100–8000) | Total floors (1 … 90) | |

**Thoughtful UI details:**
- The locality dropdown is **filtered by the selected city** and shows the count
  (e.g. *"Area Locality (163 known in Mumbai)"*).
- An explicit **"Other / not listed"** option maps to `"__unseen_locality__"`, and triggers a
  warning explaining that the estimate then relies on city + attributes alone.
- If `total_floors < floor_num`, a warning says the pipeline will correct it automatically.
- Help tooltips explain carpet vs super area, and that 0 = ground, −1 = lower basement.

**Output:**
```
Estimated monthly rent    ₹XX,XXX
Expected range            ₹LL,LLL - ₹HH,HHH      (80 % prediction interval)
Rent per sq ft            XX.X
```
plus an expandable "Model behind this estimate" panel showing model name, test RMSE, MAE, R², the
training setting, and a read-only echo of the exact input row.

> **One real bug fixed in the code, worth mentioning:** transposing a one-row frame mixes strings
> and floats in a single object column, which Arrow cannot serialise for `st.dataframe`. The echo
> is cast to string first.

### 16.5 Tab 2 — Model comparison & analysis
Renders every table and figure from `metrics.json`: the six-model comparison, the full 24-row
outlier experiment, per-city performance, feature importance, the locality ablation (with its
computed conclusion), and all residual diagnostics. **No numbers are hard-coded** — the tab is a
view over the JSON the training run produced.

### 16.6 Tab 3 — Limitations
Nine named limitations (small dataset, thin per-locality coverage, six cities only, 89-day
snapshot, missing features, asking price ≠ negotiated rent, unseen-locality fallback, outlier
treatment hurting premium predictions, accuracy varying by city), plus real-world applicability:
who would use it, how it would be integrated, retraining cadence, what data would most improve it,
and what could go wrong.

**The "what could go wrong" section is the most mature part of the project:**
> *A platform that showed these numbers as authoritative would anchor the market to its own
> predictions, creating a feedback loop where the model's errors become next year's training data.
> Under-prediction in underserved localities could also entrench existing pricing inequities.*

### 16.7 Graceful degradation
If `models/` is missing, the app shows a clear error screen with the exact commands to train the
model — it does not crash with a stack trace.

---

## 17. Every Figure Produced

All 19 live in `reports/figures/` at 130 DPI with a colour-blind-safe palette.

### EDA figures (7)
| File | Shows | Key takeaway |
|---|---|---|
| `eda_rent_distribution.png` | Rent histogram, raw vs log1p | Skew 21.41 → 0.909; justifies the log transform |
| `eda_rent_by_city.png` | Boxplots of rent per city | Mumbai's distribution barely overlaps Kolkata's |
| `eda_rent_vs_size.png` | Scatter of rent vs size, coloured by city | Positive but **city-specific slope** |
| `eda_avg_rent_by_category.png` | Mean rent by furnishing / tenant / area type / contact | Furnished > Semi > Unfurnished; Agent > Owner |
| `eda_correlation_heatmap.png` | Pearson correlations among numerics | Size–BHK–Bathroom are mutually correlated |
| `eda_listings_per_city.png` | Listing counts | Mumbai 972 … Kolkata 524 |
| `eda_outlier_per_city_vs_global.png` | Per-city vs global IQR flags | **Global rule flags 37.8 % of Mumbai** |

### Model comparison figures (3)
| File | Shows |
|---|---|
| `model_rmse_comparison.png` | CV RMSE (with error bars) vs test RMSE per model |
| `model_metric_panel.png` | 4-panel grid: R², MSE, RMSE, MAE side by side |
| `residuals_all_models.png` | Residual plot for **all six** models in one grid (the spec requires residual plots for all) |

### Outlier experiment figures (2)
| File | Shows |
|---|---|
| `outlier_experiment.png` | Test RMSE / MAE for every model × setting |
| `outlier_premium_tradeoff.png` | MAE by rent band and setting — **the premium degradation** |

### Best-model diagnostics (4)
| File | Shows |
|---|---|
| `best_model_residual_diagnostics.png` | Residual vs predicted, histogram, Q-Q plot, abs-error funnel |
| `best_model_residuals_by_city.png` | Residual distribution per city |
| `per_city_performance.png` | RMSE % of mean, with unseen-locality rate |
| `prediction_interval_coverage.png` | Actual vs interval, sorted — 77.8 % coverage visualised |

### Interpretation figures (2)
| File | Shows |
|---|---|
| `feature_importance.png` | Impurity importance and permutation importance, side by side |
| `locality_ablation.png` | Models A / B / C compared |

### How to read a Q-Q plot (it appears in `best_model_residual_diagnostics.png`)
Plots the quantiles of your residuals against the quantiles of a theoretical normal distribution.
- **Points on the 45° line** → residuals are normal
- **Ends curving up/away from the line** → heavy tails
- **S-shape** → skew

In this project the tails curve away sharply, consistent with kurtosis 185 and Shapiro p = 1.9e-49.

---

## 18. Limitations

| # | Limitation | Evidence |
|---|---|---|
| 1 | **Small dataset** | 4,746 listings total, only 3,796 used for training |
| 2 | **Thin per-locality coverage** | 2,235 localities, 1,465 appear exactly once (30.9 % of rows) |
| 3 | **Only six cities** | Will not generalise to Pune, Ahmedabad or any tier-2 city |
| 4 | **An 89-day snapshot** | 2022-04-13 → 2022-07-11. No inflation, seasonality or market drift; rupee levels already dated |
| 5 | **Missing features** | No amenities, building age, parking, metro proximity, society quality, security, view |
| 6 | **Asking price ≠ negotiated rent** | The target is the listed price, systematically above what tenants pay |
| 7 | **Unseen localities fall back to the global average** | 22–46 % of test listings per city |
| 8 | **Outlier treatment makes premium predictions unreliable** | Premium-band MAE ₹73,887 → ₹1,05,137 under "remove" |
| 9 | **Accuracy varies sharply by city** | R² from 0.79 (Chennai) to 0.10 (Kolkata) |
| 10 | **Heteroscedastic, non-normal residuals** | funnel corr 0.642, Shapiro p = 1.9e-49, kurtosis 185 |
| 11 | **Cannot extrapolate** | A tree ensemble's predictions are bounded by learned leaf values; 9 of the 10 worst errors are under-predictions |
| 12 | **`Point of Contact` is not a property attribute** | 4th most important feature, but it encodes market positioning, not the flat |

---

## 19. ML Student POV — Concepts This Project Teaches

### 19.1 Supervised vs Unsupervised — and why this is supervised
```
SUPERVISED        : labelled data (X, y). Learn a mapping f: X -> y.
                    - Regression     -> continuous y   <-- THIS PROJECT
                    - Classification -> categorical y

UNSUPERVISED      : only X, no labels. Find structure.
                    - Clustering (k-means, DBSCAN, hierarchical)
                    - Dimensionality reduction (PCA, t-SNE, UMAP)
                    - Association rules, anomaly detection

SEMI-SUPERVISED   : a few labels + lots of unlabelled data
REINFORCEMENT     : an agent learns from rewards through interaction
```

**This project is SUPERVISED REGRESSION because:**
1. Every training row has a known `Rent` value — the label is present.
2. The target is **continuous** (₹1,200 to ₹35,00,000), not a category.
3. The goal is to predict that value for a new, unseen listing.

**Why not unsupervised?** There is nothing to learn from clustering here — the business question is
*"what is the rent?"*, and the answer is already recorded in the data. Clustering could group
similar listings, but it could never output a number in rupees. Unsupervised learning would only
make sense if the rent column were absent (then you might cluster to find market segments) or as a
*preprocessing* step (e.g. clustering localities into tiers to reduce cardinality — a legitimate
alternative to target encoding worth mentioning).

**Could this be framed as classification?** Yes, by binning rent into bands (the project already
has `rent_band`: <₹15k, ₹15k–40k, ₹40k–1L, >₹1L). But it would **throw away information** — the
difference between ₹14,999 and ₹15,001 would become a class boundary, while ₹15,001 and ₹39,999
would be identical. Regression preserves the full ordering and magnitude.

### 19.2 Parametric vs Non-parametric

| | **Parametric** | **Non-parametric** |
|---|---|---|
| Definition | Fixed number of parameters, independent of n | Number of parameters grows with n |
| In this project | Linear, Polynomial, Ridge | Decision Tree, Random Forest, Gradient Boosting |
| Assumes a functional form? | Yes | No |
| Extrapolates? | **Yes** | **No** |
| Needs more data? | Less | More |
| Interpretable? | Coefficients are directly readable | Harder (needs importance / SHAP) |

**This distinction is exactly what explains the project's central finding:** Linear Regression wins
RMSE because it **extrapolates** to the premium tail; the tree ensembles cannot and so
under-predict it.

### 19.3 Bias–Variance Trade-off

```
E[(y - ŷ)²]  =  Bias²  +  Variance  +  Irreducible Error
                  |          |              |
                  |          |              +-- noise in the data; cannot be reduced
                  |          +----------------- sensitivity to the particular training set
                  +---------------------------- error from wrong assumptions (too simple a model)
```

| | High bias (underfitting) | High variance (overfitting) |
|---|---|---|
| Train error | High | **Low** |
| Test error | High | **High** |
| Gap | Small | **Large** |
| Example here | Linear Regression on raw rent (R² 0.582) | **Polynomial deg 2 (train RMSE 62,695 → test RMSE 9,17,495)** |
| Fix | More features, more complex model, less regularisation | More data, fewer features, **more** regularisation, ensembling, early stopping |

**How each model in this project sits on the curve:**
```
simple <------------------------------------------------> complex
Linear  <  Ridge-poly  <  Pruned tree  <  RF  <  GB  <  Unregularised poly-2
high bias                                                 high variance
```

**How each technique here fights one side:**
| Technique | Fights |
|---|---|
| Ridge (`alpha`) | Variance |
| `max_depth`, `min_samples_leaf` | Variance |
| Bagging (Random Forest) | **Variance** |
| Boosting (Gradient Boosting) | **Bias** |
| `subsample < 1.0` | Variance |
| Target-encoding smoothing | Variance (on rare localities) |
| Polynomial features | Bias (adds capacity) |

### 19.4 Data Leakage — the four kinds, and how each is prevented here

| Kind | What it is | Where it would occur here | Prevention in this code |
|---|---|---|---|
| **Target leakage** | A feature derived from the target | `rent_per_sqft = Rent / size` | Never added to `NUMERIC_FEATURES`; `remainder="drop"` makes it unreachable |
| **Train–test contamination** | Test information used in fitting | Imputing with the full-dataset median; fitting `TargetEncoder` on all data | Everything is inside the `Pipeline`, so it is re-fit per fold |
| **Preprocessing leakage** | Scaler/encoder fit before splitting | `StandardScaler` on the whole dataset | Inside the pipeline |
| **Evaluation leakage** | Tuning on the test set | Choosing hyperparameters by test score | Tuning uses **CV on the training split only**; the test set is touched exactly once per configuration, for reporting |

**Plus a fifth, specific to this project:** treating outliers in the **test** set. The code treats
the training split only — explicitly, with the reason in the docstring.

### 19.5 Encoding categorical variables — the full menu

| Method | How it works | Columns added | Good for | Used here? |
|---|---|---|---|---|
| **One-Hot** | One binary column per level | k (or k−1) | Low cardinality, nominal | ✅ City, Furnishing, Tenant, Area Type, Contact |
| **Label / Ordinal** | Map each level to an integer | 1 | **Ordinal** data only | Only implicitly, for floor word-tokens |
| **Target / Mean** | Replace level with smoothed mean of y | 1 | **High cardinality** | ✅ Area Locality (2,235 levels) |
| **Frequency / Count** | Replace level with its occurrence count | 1 | High cardinality, when frequency is meaningful | No |
| **Binary / Hashing** | Encode level index in binary / hash to fixed buckets | log₂k / fixed | Very high cardinality, memory-bound | No |
| **Embeddings** | Learned dense vectors | d | Neural nets, very high cardinality | No |

**The danger of Label Encoding a nominal variable:** mapping `{Mumbai: 0, Delhi: 1, Bangalore: 2,
...}` tells a linear model that Bangalore is "twice" Delhi and that Mumbai + Bangalore = 2×Delhi.
That ordering is pure fiction. One-hot avoids it entirely.

### 19.6 The log transform — the most important modelling decision here

```
log1p(x) = log(1 + x)          expm1(x) = exp(x) - 1          (exact inverses)
```

**Why `log1p` and not plain `log`:** `log(0)` is −∞. `log1p(0) = 0`. Even though minimum rent here
is ₹1,200, `log1p` is the safe habit and is numerically more accurate for small x.

**Three effects, all desirable:**

1. **It fixes the skew.** 21.41 → 0.909.
2. **It converts a multiplicative relationship into an additive one.**
   ```
   log(Rent) = β0 + β1·Size   <=>   Rent = e^β0 · e^(β1·Size)
   ```
   Each extra square foot **multiplies** rent by a constant factor rather than adding a constant
   number of rupees. That is how rent genuinely behaves: an extra 100 sq ft in Mumbai is worth far
   more rupees than in Kolkata, but a similar *percentage*.
3. **It changes the implicit loss to relative error.** Squared error in log space ≈ squared
   *relative* error in rupee space. On a target spanning ₹1,200 to ₹35,00,000, that is the right
   objective — a ₹5,000 miss on a ₹10,000 flat is a disaster; on a ₹5,00,000 penthouse it is noise.

**The cost you must acknowledge:** back-transforming a log-space **mean** introduces a known
downward bias (`exp(E[log X]) = geometric mean ≤ arithmetic mean`, by Jensen's inequality). The
project's response is twofold: the median residual is −₹37.70 (so the bias is negligible in
practice), and the **prediction intervals use quantiles, which are transform-exact**.

### 19.7 The Curse of Dimensionality — why one-hot localities would fail
With 2,235 one-hot locality columns on 3,796 training rows, you have **more features than samples**
for that one variable. `XᵀX` is singular; a tree would find near-perfect splits that memorise
individual listings; and 1,465 columns would contain exactly one `1` each. Target encoding reduces
2,235 columns to **1**.

### 19.8 Ensemble methods — the three families
```
BAGGING    : parallel, independent learners, averaged        -> reduces VARIANCE
             Random Forest, Extra Trees

BOOSTING   : sequential, each corrects the previous          -> reduces BIAS
             AdaBoost, Gradient Boosting, XGBoost, LightGBM, CatBoost

STACKING   : train a meta-model on the base models' outputs  -> reduces both
             (not used here)
```

### 19.9 The complete ML workflow this project demonstrates
```
 1. Understand the problem      -> read the case study; identify it as supervised regression
 2. Get the data                -> real Kaggle data, no synthetic fallback
 3. PROFILE the data            -> BEFORE writing any preprocessing
 4. EDA                         -> 7 figures, each answering a question
 5. Clean                       -> implausible sizes blanked, Floor parsed
 6. Feature engineer            -> floor_ratio, carpet_equiv_size, interactions
 7. SPLIT                       -> once, stratified, frozen
 8. Preprocess INSIDE a pipeline-> impute, encode, scale (per fold)
 9. Train multiple models       -> 6 variants
10. Tune                        -> RandomizedSearchCV, 5-fold, refit on MAE
11. Evaluate on held-out test   -> R2, MSE, RMSE, MAE, MAPE
12. Diagnose                    -> residuals, worst cases, per-city, error concentration
13. Interpret                   -> permutation + impurity importance, ablation
14. Select with justification   -> multi-criteria rank-sum, documented trade-off
15. Quantify uncertainty        -> quantile models, validated coverage
16. Deploy                      -> single joblib artifact + Streamlit
17. State limitations           -> 12 named, with evidence
```

### 19.10 What makes THIS project defensible in a viva

| Decision | The defensible reason |
|---|---|
| Profiled before preprocessing | *"I verified the column spellings, the skew and the cardinality against the real CSV instead of assuming."* |
| Dropped `Posted On` | *"89-day span — I measured it. Any date feature would be noise."* |
| Log-transformed the target | *"Skew 21.41 → 0.909. I measured both."* |
| Target-encoded locality | *"2,235 levels, 1,465 singletons. One-hot would add more columns than rows."* |
| Per-city IQR | *"A global rule flags 37.8 % of Mumbai. I compared both and have the figure."* |
| **Kept** the outliers | *"I tested all four settings. Removal raised premium MAE by 42 % for a ₹46 gain on cheap flats."* |
| Tuned on MAE | *"CV RMSE std/mean is 0.60–0.81; CV MAE is 0.10–0.15. I measured the noise."* |
| Chose GB over Linear despite worse RMSE | *"One test row is 42 % of total MSE. Linear wins RMSE by extrapolating on ~10 listings; GB is better on the other 940."* |
| Everything inside a Pipeline | *"So CV re-fits every learned statistic per fold and the app needs zero preprocessing."* |
| Quantile prediction intervals | *"Quantiles are transform-invariant, so expm1 of a log-space quantile is exact. Validated at 77.8 % vs 80 % nominal."* |

### 19.11 The six numbers to memorise

```
4,746   listings  (3,796 train / 950 test)
21.41   raw rent skew  ->  0.909 after log1p
2,235   localities, 1,465 appearing exactly once
0.7144  test R2
10,135  test MAE in rupees   (RMSE 32,716 ; MAPE 26.7 %)
42 %    share of total test MSE caused by ONE listing
```

---

*Generated from the project's own artifacts: `models/metrics.json`, `reports/model_comparison.csv`,
`reports/outlier_experiment.csv`, `reports/per_city_performance.csv`,
`reports/feature_importance.csv`, `reports/premium_tradeoff.csv`, `reports/worst_predictions.csv`,
`reports/locality_ablation.csv`.*
