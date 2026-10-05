# Tech Stack & Dependency Reference

A complete inventory of everything this project is built from: the language, the libraries, the modules we wrote ourselves, and the file formats in between. For each one: **what it is**, **why it's here**, **how it works**, **how we actually use it**, **what it bought us**, and **what we could have used instead**.

Numbers quoted throughout come from the last training run recorded in `models/metrics.json`.

---

## Part 1 — The Stack at a Glance

| Layer | Technology | Version | Role in this project |
|---|---|---|---|
| **Language** | Python | 3.14.5 | Everything. No second language in the codebase. |
| **Data handling** | pandas | 3.0.6 | Loading the CSV, cleaning, all table manipulation |
| **Numerics** | NumPy | 2.5.3 | Array math, the log/exp target transform, metric arithmetic |
| **Machine learning** | scikit-learn | 1.9.1 *(pinned exactly)* | Preprocessing, all 6 models, tuning, cross-validation, metrics |
| **Statistics** | SciPy | 1.18.1 | Hyperparameter search distributions, normality tests, Q–Q plots |
| **Plotting (base)** | Matplotlib | 3.11.2 | Every one of the 18 figures is ultimately a Matplotlib figure |
| **Plotting (style)** | Seaborn | 0.13.2 | Statistical chart types and a consistent colour-blind-safe theme |
| **Model persistence** | joblib | 1.6.0 | Saving/loading the trained pipeline as `.joblib` files |
| **Web app** | Streamlit | 1.64.0 | The interactive prediction UI |
| **Notebook tooling** | jupyter / nbformat / nbconvert / ipykernel | 1.1.1 / 5.11.1 / 7.17.1 / 7.4.0 | Building and executing the analysis notebook (dev-only) |
| **Data format (in)** | CSV | — | `data/House_Rent_Dataset.csv`, 4,746 rows × 12 columns |
| **Data format (out)** | JSON, CSV, PNG, `.joblib`, `.ipynb`, Markdown | — | Metrics, report tables, figures, models, notebook, docs |

**Two dependency tiers.** `requirements.txt` holds the 8 packages needed to train a model and run the app. `requirements-dev.txt` adds the 4 Jupyter packages needed only to rebuild the analysis notebook. Someone deploying the app never installs the notebook stack.

### Why scikit-learn is pinned with `==` and nothing else is

The other packages use `==` too, but scikit-learn's pin is load-bearing for a specific reason. A `.joblib` file is a **pickled Python object graph** — it stores the live scikit-learn objects (the `ColumnTransformer`, the `TargetEncoder`, the fitted trees), not a neutral description of them. Those objects' internal attribute names change between scikit-learn releases. Load `models/pipeline.joblib` under a different version and you get an `InconsistentVersionWarning`, or worse, a pipeline that loads without complaint and quietly predicts wrong because an internal attribute silently defaulted. The pin is the only thing preventing that.

---

## Part 2 — How the Pieces Fit Together

The project has one hard architectural rule, and most of the technology choices follow from it:

> **Every transformation lives inside the scikit-learn `Pipeline` object.**

Nothing is done to the data "before" the model and then re-done by hand elsewhere. The consequence is that `models/pipeline.joblib` is *self-contained*: hand it a raw one-row DataFrame with the right column names and it returns a rupee figure. The Streamlit app contains zero preprocessing code — the comment at the top of [app/app.py](app/app.py#L1-L9) says exactly this.

That rule is what makes train-time and predict-time behaviour provably identical. The classic ML bug — the app imputes missing values slightly differently than training did — is structurally impossible here, because there's only one implementation.

### Data flow

```
data/House_Rent_Dataset.csv  (4,746 raw listings)
        │
        ▼
  src/data.py        load + validate header + profile
        │
        ▼
  train.py           80/20 split, stratified on City  →  3,796 train / 950 test
        │
        ├─────────────────────────────────────────────────────┐
        ▼                                                     │
  THE PIPELINE  (one sklearn object)                          │
    1. FeatureEngineer      parse Floor, impute Size,         │
       (src/preprocess.py)  derive carpet_equiv_size          │
    2. ColumnTransformer    scale / one-hot / target-encode   │
    3. The model            Linear | Poly | Tree | RF | GB    │
    wrapped in TransformedTargetRegressor (log1p ↔ expm1)     │
        │                                                     │
        ▼                                                     ▼
  RandomizedSearchCV  ×  5-fold CV                      src/evaluate.py
        │                                               metrics in rupees
        ▼                                                     │
  models/pipeline.joblib  +  quantile_low/high.joblib         │
  models/metrics.json  ◄──────────────────────────────────────┘
        │
        ├──────────────────┬────────────────────┬──────────────────┐
        ▼                  ▼                    ▼                  ▼
   app/app.py        make_readme.py    build_notebook.py    reports/*.csv
   (Streamlit)        (README.md)        (.ipynb)            + 18 PNGs
```

Note the convergence at `models/metrics.json`. The app, the README and the notebook all read their numbers from that one file rather than hard-coding them, so a retrain updates all three consistently and no document can drift out of step with the model actually on disk.

---

## Part 3 — The Libraries, One by One

### 3.1 Python 3.14.5

**Why this language.** The entire mature ML tooling ecosystem — scikit-learn, pandas, NumPy — is Python-first. Choosing anything else would mean reimplementing or wrapping that ecosystem.

**How we use the modern features.** Every module opens with `from __future__ import annotations`, which makes type hints lazily evaluated strings instead of objects resolved at import time. Practically: we can write `dict | None` and `list[ModelSpec]` without import-order headaches or runtime cost. We also lean on `@dataclass` for the configuration objects (`ModelSpec` in [src/models.py:52](src/models.py#L53), `Setting` in [train.py:67](train.py#L67)) — these generate `__init__`, `__repr__` and `__eq__` automatically, so a model definition reads as a declaration of facts rather than boilerplate.

**Alternative.** R has excellent statistics and `tidymodels`, but weaker deployment story — nothing as frictionless as Streamlit. Julia is faster but the ML ecosystem is far thinner. For a tabular regression project that ends in a web app, Python is not a close call.

---

### 3.2 pandas 3.0.6

**What it is.** The DataFrame library — a spreadsheet you can program against. Rows, named columns, mixed data types per column.

**How it works.** A DataFrame is a collection of columns, each backed by a typed array. Operations are **vectorised**: `df["Size"] * 0.7` multiplies all 4,746 values in one compiled loop rather than 4,746 Python iterations. That's roughly a 100× speedup over hand-written loops, and it's why pandas code looks declarative.

**How we use it.**
- **Loading and validation** — `pd.read_csv` in [src/data.py:54](src/data.py#L54), immediately followed by an exact header check. If the CSV's column spellings change, we fail loudly right there instead of silently dropping a feature later (the `ColumnTransformer` selects columns by exact string name, spaces and capitalisation included).
- **Parsing messy text** — the `Floor` column stores two facts in one string (`"Ground out of 2"`). [`parse_floor`](src/preprocess.py#L36) uses `.str.split()` with a regex to break it apart, `.map()` to convert word tokens (`Ground → 0.0`, `Upper Basement → -0.5`, `Lower Basement → -1.0`) and `pd.to_numeric(errors="coerce")` to turn anything unparseable into `NaN` rather than crashing.
- **Group-wise imputation** — `groupby(["City", "BHK"]).median()` learns a sensible fill value for missing `Size` from comparable listings, not a single global number.
- **Profiling** — [`profile_data`](src/data.py#L95) builds the entire section-4 data audit as a dictionary of pandas aggregations.
- **Every report table** — `reports/*.csv` are DataFrames written with `.to_csv()`.

**Impact.** The dataset is small (4,746 rows), so pandas isn't here for speed — it's here for *expressiveness*. The `Floor` parsing in particular would be 50+ lines of fragile string handling in plain Python.

**Alternatives.** **Polars** is substantially faster and has a stricter, cleaner API, but scikit-learn's integration with it is still less seamless than with pandas. **DuckDB** would be excellent if the data lived in a database. At 4,746 rows, none of the performance arguments apply — the ecosystem fit decides it.

---

### 3.3 NumPy 2.5.3

**What it is.** The numerical array library every other scientific Python package is built on. pandas columns *are* NumPy arrays underneath.

**How it works.** A NumPy array is a contiguous block of memory holding one data type, with operations implemented in compiled C. Where Python's lists are flexible and slow, NumPy arrays are rigid and fast.

**How we use it.** NumPy is mostly invisible infrastructure here, but it does three visible jobs:

1. **The target transform** — `np.log1p` and `np.expm1`, passed into `TransformedTargetRegressor` in [src/models.py:206](src/models.py#L206). This is the single most consequential line in the modelling code; see §3.4.2.
2. **Metric arithmetic** — [`compute_metrics`](src/evaluate.py#L49) coerces everything to `float64` arrays and computes MAPE with `np.mean(np.abs(...))`.
3. **Clipping impossible predictions** — `np.clip(y_pred, 0.0, None)`. A model trained on raw rent (outlier-experiment setting 1) can predict a *negative* monthly rent. Scoring those unclipped would distort the comparison against the log-target models, so we floor them at zero before measuring.

**Why `log1p` and not plain `log`.** `log1p(x)` computes `log(1 + x)`. The `+1` guarantees the function is defined at zero and stays numerically precise for small values. `expm1` is its exact inverse. Using plain `log`/`exp` would be mathematically equivalent here (minimum rent is ₹1,200, so no zeros) but loses precision and would break on any dataset containing a zero.

**Alternative.** There isn't a realistic one. NumPy is the substrate — pandas, scikit-learn, SciPy and Matplotlib all speak it natively. Replacing it would mean replacing all of them.

---

### 3.4 scikit-learn 1.9.1 — the core of the project

This deserves its own subsections. We use roughly a dozen distinct scikit-learn components, and the choices between them are where most of the project's reasoning lives.

#### 3.4.1 `Pipeline` and `ColumnTransformer` — the skeleton

**What they are.** `Pipeline` chains transformation steps so the whole chain behaves like a single model with one `.fit()` and one `.predict()`. `ColumnTransformer` routes *different columns* down *different* transformation paths and glues the results back together.

**How `ColumnTransformer` works here.** Built in [`build_preprocessor`](src/preprocess.py#L418), it splits our 11 input columns into three streams:

| Stream | Columns | Treatment |
|---|---|---|
| `num` | BHK, Size, Bathroom, floor_num, total_floors, floor_ratio, carpet_equiv_size | median-impute → *(scale, linear models only)* |
| `cat` | City, Furnishing Status, Tenant Preferred, Area Type, Point of Contact | mode-impute → one-hot encode |
| `loc` | Area Locality | constant-impute → **target encode** |

`remainder="drop"` throws away everything not explicitly routed — `Posted On`, the raw `Floor` string, and the target itself. That's a safety net: a column can't accidentally reach the model.

**Why this is non-negotiable.** Fitting these steps *outside* cross-validation and then CV-ing the model alone is the single most common cause of inflated ML scores. Inside a `Pipeline`, scikit-learn **clones the whole chain on every fold**, so every learned quantity — the group medians, the one-hot vocabulary, the target encodings — comes from that fold's training rows only. The comment at [src/evaluate.py:76-81](src/evaluate.py#L76-L81) spells this out.

**Alternative.** Hand-rolled preprocessing functions. Faster to write, and reliably wrong — it's exactly how train/serve skew and leakage get introduced.

#### 3.4.2 `TransformedTargetRegressor` — predicting log-rent, reporting rupees

**The problem it solves.** Rent in this dataset has a **skew of 21.4** — wildly lopsided. Median rent is ₹16,000; the maximum is ₹35,00,000. A squared-error model fitted to raw rent spends nearly all its effort chasing a handful of penthouses and fits ordinary flats poorly. After `log1p`, the skew drops to **0.91**, which is close to symmetric.

**How it works.** The wrapper applies `func=np.log1p` to `y` before fitting and `inverse_func=np.expm1` to predictions on the way out. Three things fall out of this at once ([src/models.py:17-27](src/models.py#L17-L27)):

1. Standard scorers report R²/RMSE/MAE **in rupees**, with no manual inversion for us to get wrong.
2. The `TargetEncoder` inside the pipeline sees the *log* target, so locality averages aren't distorted by the skew.
3. The saved pipeline outputs rupees directly, so the app needs no post-processing.

Setting `log_target=False` drops the wrapper entirely — that's setting 1 of the outlier experiment, which exists precisely to demonstrate what the transform buys.

**Alternative.** Transform `y` manually and remember to invert everywhere. Workable, and one forgotten `expm1` silently corrupts a whole results table.

#### 3.4.3 `TargetEncoder` — the locality problem

**The problem.** `Area Locality` has **~2,235 distinct values, 1,465 of which appear exactly once.** One-hot encoding would add 2,235 near-empty columns to a 3,796-row training set. Every model would memorise individual listings.

**How target encoding works.** Replace each category with the *mean target value* for that category. "Bandra West" becomes the average log-rent of Bandra West listings — one informative number instead of 2,235 sparse flags.

**The danger, and the two defences.** Naively, this leaks the answer: a locality seen once gets encoded as *its own rent*. We defend twice:

- **`smooth="auto"`** applies empirical-Bayes shrinkage. A locality with 200 listings keeps its own mean; a locality with 1 listing is pulled almost entirely toward the global mean. Rare categories are mathematically prevented from carrying much signal.
- **Internal cross-fitting** via a seeded `KFold` passed as `cv=`. During `fit_transform`, scikit-learn encodes each fold using only the *other* folds' targets, so even training-set encodings are out-of-fold.

And because the encoder sits **inside** the pipeline, it's re-fitted per CV fold on top of all that. The warning block at [src/preprocess.py:488-497](src/preprocess.py#L488-L497) says in capitals not to hoist it out.

**Impact.** The locality ablation study (`reports/locality_ablation.csv`) measures exactly what this feature contributes by training models with it, without it, and with *only* location.

**Alternatives.** **One-hot** — infeasible at 2,235 levels. **Frequency encoding** — leak-free but discards rent information. **Hashing** — fixed width, loses interpretability. **CatBoost's ordered target statistics** — arguably better, but requires leaving scikit-learn.

#### 3.4.4 `OneHotEncoder` — the small categoricals

Five columns with few levels each. One-hot turns `City` into six 0/1 columns, so the model can't infer a false ordering (Mumbai > Delhi > Bangalore means nothing).

Two settings are deliberate:

- **`handle_unknown="ignore"`** is *required*, not defensive tidiness. `Built Area` appears in **2 rows** of the entire dataset and `Contact Builder` in **1**. A random split can easily put such a level in test but not train. Without this flag the encoder raises at predict time — and the same applies the moment a real user picks an unusual option in the app.
- **`drop="first"` for linear models only.** Six columns summing to 1 are perfectly collinear, which makes the matrix singular and the degree-2 polynomial expansion blow up. Dropping one level fixes it. Trees don't care, so they keep the full coding.

#### 3.4.5 `SimpleImputer` — filling gaps

Three strategies, one per stream: `median` for numerics (robust to the skew in a way `mean` is not), `most_frequent` for one-hot categoricals, and `constant` with `"__missing__"` for locality — where "we don't know the locality" is genuinely informative and deserves its own category rather than being blended into the most common one.

The subtle case is handled *before* imputation: 16 listings report a `Size` below 50 sq ft (smallest: 10). Those are data-entry errors, not micro-studios, so [`_blank_implausible_size`](src/preprocess.py#L130) converts them to `NaN` so the imputer can replace them with a group median.

#### 3.4.6 `StandardScaler` — only where it matters

Rescales each feature to mean 0, standard deviation 1. Applied to **linear models only**. Linear and Ridge regression are sensitive to feature scale — `Size` in the thousands and `BHK` in single digits means the penalty term in Ridge would effectively punish one far more than the other. Tree models split on thresholds and are completely scale-invariant, so scaling them would be pure overhead. The `family` parameter in `build_preprocessor` encodes this distinction.

#### 3.4.7 The six models

Five algorithms, six rows — Polynomial Regression is reported both plain and Ridge-regularised, because the plain version's failure is itself a finding worth showing.

| Model | How it works | Test MAE | Test RMSE | Test R² |
|---|---|---|---|---|
| **Linear Regression** | Fits one straight-line coefficient per feature. No hyperparameters, by design. | ₹10,760 | ₹27,897 | 0.7924 |
| **Polynomial (deg 2)** | Adds every squared term and pairwise product — ~24 features become ~324. | ₹42,321 | ₹917,495 | **−223.59** |
| **Polynomial (deg 2) + Ridge** | Same expansion with L2 shrinkage pulling coefficients toward zero. | ₹15,257 | ₹143,423 | −4.4882 |
| **Decision Tree** | Recursive yes/no splits. Interpretable, high variance. | ₹11,640 | ₹34,818 | 0.6766 |
| **Random Forest** | Hundreds of trees on bootstrapped samples, averaged. Variance reduction. | ₹10,245 | ₹35,092 | 0.6714 |
| **Gradient Boosting** ✅ | Trees added sequentially, each correcting the previous ones' errors. | **₹10,135** | ₹32,716 | 0.7144 |

A negative R² means *worse than always guessing the average* — the plain polynomial model is a textbook demonstration of overfitting, and Ridge improving it from −223 to −4.5 shows what regularisation buys. Both rows stay in the report for that reason.

One implementation detail worth noting: `RandomForestRegressor` is built with **`n_jobs=1`**, deliberately. The surrounding `RandomizedSearchCV` already runs with `n_jobs=-1`, and letting both parallelise creates nested thread pools that oversubscribe the CPU and run *slower*, not faster ([src/models.py:143-147](src/models.py#L143-L147)).

#### 3.4.8 `RandomizedSearchCV` — hyperparameter tuning

**How it works.** Samples N random combinations from the parameter distributions, scores each with 5-fold cross-validation, keeps the best. Our budgets: Ridge 30 iterations, Decision Tree 40, Random Forest 25, Gradient Boosting 30.

**Why random rather than exhaustive grid search.** With 5 Gradient Boosting hyperparameters at 5 values each, a grid is 3,125 fits × 5 folds = 15,625 model trainings. Random search with 30 iterations is 150. The research result that justifies this (Bergstra & Bengio, 2012): when only some hyperparameters actually matter, random sampling explores the *important* dimensions far more efficiently than a grid, which wastes most of its budget varying parameters that don't affect the outcome.

**Alternatives.** **`GridSearchCV`** — exhaustive, affordable only on tiny spaces. **Bayesian optimisation** (Optuna, scikit-optimize) — smarter still, uses past trials to pick the next, but adds a dependency for a gain that wouldn't show on 3,796 rows. **`HalvingRandomSearchCV`** — scikit-learn's own successive-halving variant, a reasonable upgrade path.

#### 3.4.9 `KFold` and `cross_validate` — honest scoring

The 80% training portion is split into 5 folds; each model trains on 4 and validates on the 1 held out, five times. We get a **mean and a standard deviation**, and that standard deviation is the point — it tells us whether a good score was skill or luck.

[`cv_splitter`](src/evaluate.py#L41) returns a single shared, seeded `KFold` so **every model is scored on byte-identical folds**. Without that, model A might get an easier split than model B and the comparison would be meaningless.

One quirk the code handles: scikit-learn's convention is "higher is always better", so error metrics are returned negated (`neg_root_mean_squared_error`). [`cv_report`](src/evaluate.py#L73) flips the signs back so the tables read naturally.

#### 3.4.10 Metrics

`r2_score`, `mean_squared_error`, `root_mean_squared_error`, `mean_absolute_error`, plus MAPE computed by hand. All reported **in rupees**, never in log units — as the module docstring puts it, an RMSE of "0.34 log-rupees" tells a landlord nothing.

#### 3.4.11 `permutation_importance` — which features matter

**How it works.** Take the fitted model, shuffle one feature column at random, re-score. If the error jumps, that feature mattered. Repeat 10 times per feature for a mean and standard deviation.

**Why not the trees' built-in `feature_importances_`?** Tree importance is biased toward high-cardinality features — it would almost certainly overstate `Area Locality` simply because that column offers more places to split. Permutation importance measures what the model's *accuracy* actually depends on. We compute both (`reports/feature_importance.csv` and `feature_importance_tree.csv`) and the difference is itself informative.

The result is reported in **rupees of added MAE**, which is directly interpretable: "shuffling this column costs ₹X of accuracy."

#### 3.4.12 Quantile Gradient Boosting — the prediction interval

The app's "Expected range" comes from two *extra* models with `loss="quantile"`, trained at `alpha=0.10` and `alpha=0.90`.

**Why a different loss function.** Ordinary squared-error regression predicts the conditional *mean*. Quantile loss asymmetrically penalises over- and under-prediction, so the model learns a conditional *percentile* instead. Two of them bracket an 80% interval.

**The clean mathematical property.** Quantiles are invariant under monotone transforms, so `expm1(q_alpha(log rent))` is *exactly* the alpha-quantile of rent in rupees — no approximation, unlike the mean, where `exp(mean(log x)) ≠ mean(x)`.

**Honest reporting of its limits.** Measured empirical coverage is **77.8%** against a nominal 80%, and the median interval width is **74% of predicted rent**. Both numbers are stored in `metrics.json` and surfaced in the app rather than hidden.

The interval models inherit the main model's tuned tree geometry rather than being tuned separately — cheaper and more consistent ([src/models.py:246-262](src/models.py#L246-L262)).

#### 3.4.13 `BaseEstimator` / `TransformerMixin` — our custom step

`FeatureEngineer` subclasses these to become a first-class scikit-learn transformer. `BaseEstimator` supplies `get_params`/`set_params` (needed for cloning and `RandomizedSearchCV`); `TransformerMixin` supplies `fit_transform` for free. Inheriting from them is what lets our own code sit inside a `Pipeline` and be cloned per CV fold like any built-in step. Details in §4.2.

---

### 3.5 SciPy 1.18.1

**What it is.** Scientific computing built on NumPy — optimisation, signal processing, statistics.

**How we use it.** Two distinct jobs:

**1. Hyperparameter search distributions** ([src/models.py:40](src/models.py#L40)). `RandomizedSearchCV` doesn't sample from lists, it samples from *distributions*, and SciPy supplies them:

- `randint(200, 600)` — uniform integers, for things like `n_estimators`.
- `uniform(0.7, 0.3)` — a float in [0.70, 1.00]. Note SciPy's convention: the arguments are *(loc, scale)*, i.e. start and width, not start and end. Easy to misread.
- `loguniform(1e-2, 1e7)` — the interesting one. For Ridge's `alpha`, each **order of magnitude** gets equal sampling weight, so 0.1 is as likely as 100,000. A plain uniform draw over that range would put virtually every sample above 10⁵. The range is deliberately wide: a degree-2 expansion of standardised features produces terms whose squares are large, so the alpha that actually controls this model can be very high, and truncating at 1e4 would handicap it and make the comparison against the unregularised version unfair.

**2. Residual diagnostics.** `stats.shapiro` tests residuals for normality ([train.py:590](train.py#L590)); `stats.probplot` generates the Q–Q plot ([src/plots.py:528](src/plots.py#L528)) that shows *visually* where residuals depart from normal — information a single p-value can't convey.

**Alternative.** Python's `random` module could produce the integers, but not `loguniform`, and it wouldn't give `RandomizedSearchCV` the `.rvs()` interface it expects. The stats functions have no convenient replacement short of `statsmodels`, a much heavier dependency for two calls.

---

### 3.6 Matplotlib 3.11.2

**What it is.** Python's foundational plotting library. Every figure in `reports/figures/` — all 18 — is a Matplotlib figure, even the ones drawn by Seaborn.

**How we use it.** [src/plots.py](src/plots.py) centralises all figure generation, with two shared helpers worth calling out:

- **[`_inr_axis`](src/plots.py#L55)** attaches a `FuncFormatter` so rupee axes use **Indian digit grouping** (₹1,20,000 — lakh-style, not ₹120,000). A small thing that makes the charts read correctly to the intended audience.
- **[`save`](src/plots.py#L61)** writes to `reports/figures/` at 130 DPI with `bbox="tight"` and closes the figure. Closing matters: in a loop generating 18 figures, unclosed figures accumulate in memory and Matplotlib starts warning.

**One genuinely non-obvious fix.** The style block pins `font.family` to **DejaVu Sans** ([src/plots.py:33-38](src/plots.py#L33-L38)). Seaborn's default theme resolves to Arial on macOS, and **Arial has no glyph for U+20B9, the Indian rupee sign** — every rupee label renders as a hollow box. DejaVu Sans is Matplotlib's bundled default and does contain it. This is the kind of bug that's invisible until someone opens the figures on a different machine.

**Alternatives.** **Plotly** gives interactive charts and would integrate nicely with Streamlit, but produces heavier output and doesn't export cleanly to static PNGs for a report. **Altair** has an elegant declarative grammar but less control over fine details like custom axis formatters. For figures that must work identically in a notebook, a Markdown report and a web app, static Matplotlib PNGs are the pragmatic answer.

---

### 3.7 Seaborn 0.13.2

**What it is.** A statistical plotting layer on top of Matplotlib. Not a replacement — it *returns* Matplotlib objects you can keep customising.

**Why both.** Seaborn gives us chart types that would each be 20+ lines of raw Matplotlib — box plots by city, correlation heatmaps, regression scatter plots — and a coherent visual theme across all 18 figures from one `set_theme()` call. We then drop to Matplotlib for the custom bits (rupee formatters, annotations).

**How we use it.** `sns.set_theme(style="whitegrid", palette="colorblind")`. The palette choice is deliberate — **colour-blind-safe**, so the six-city charts remain readable for the ~8% of men with red-green colour vision deficiency. Specific chart types: `boxplot` for rent-by-city, `heatmap` for the correlation matrix, `scatterplot`/`regplot` for rent-vs-size, `barplot` for the category comparisons.

**Alternative.** Raw Matplotlib for everything — achievable, roughly 3× the code, and keeping 18 figures visually consistent by hand is a losing battle.

---

### 3.8 joblib 1.6.0

**What it is.** Serialisation — turning a live Python object into bytes on disk and back — optimised for objects containing large NumPy arrays.

**How it works, and why not plain `pickle`.** joblib *uses* pickle underneath but special-cases NumPy arrays, storing them in an efficient binary form instead of pickle's generic protocol. For a Random Forest holding hundreds of trees' worth of array data, that's meaningfully smaller and faster. scikit-learn's own documentation recommends joblib for exactly this.

**How we use it.** Three artifacts saved at [train.py:755](train.py#L755) and [train.py:778](train.py#L778): `pipeline.joblib` (the full selected pipeline), plus `quantile_low.joblib` and `quantile_high.joblib`. The app loads them with `joblib.load`.

**The deployment gotcha we handle.** A pickle stores a *reference* to the class, not the class itself — `models/pipeline.joblib` contains the string `src.preprocess.FeatureEngineer`. On load, Python must be able to import that path. If it can't, you get `ModuleNotFoundError`. That's why [app/app.py:22-24](app/app.py#L22-L24) inserts the project root into `sys.path` **before** the `joblib` import. The comment notes this is the single most common reason a Streamlit Community Cloud deployment fails while working fine locally.

**Alternatives.** **`pickle`** — works, slower on arrays. **ONNX** — a framework-neutral format that would eliminate the version-pinning problem entirely and allow serving from non-Python runtimes, but converting a pipeline with a custom transformer is substantial work. **PMML** — same idea, older, less support. For a Python-to-Python handoff, joblib is correct.

---

### 3.9 Streamlit 1.64.0

**What it is.** A framework that turns a Python script into a web app. No HTML, CSS, JavaScript, routing or REST endpoints.

**How it works.** Streamlit **re-runs the entire script top to bottom on every interaction.** Move a slider, the whole script executes again. That sounds wasteful, and it is — which is why the caching decorators are not optional:

- **`@st.cache_resource`** on [`load_pipeline`](app/app.py#L42) and [`load_quantile_models`](app/app.py#L49). For unhashable, expensive-to-create objects like a deserialised model. Without it, every slider move would re-read and re-unpickle three `.joblib` files. With it, the model is deserialised once per session.
- **`@st.cache_data`** on [`load_json`](app/app.py#L60). For serialisable data; returns a copy, so a mutation in one place can't corrupt the cache for another.

**How we use it.** Three tabs — Predict, Analysis, Limitations. The Predict tab builds a one-row DataFrame from the widget values and calls `.predict()`. That's the entire inference path; no preprocessing, because the pipeline carries it.

Two things the app does that are worth noting as *design* rather than mechanics:

- It shows the **Expected range** alongside the point estimate, and labels the source (`interval_source`) so users can tell whether they're seeing a true quantile interval or the ±MAE fallback.
- Picking a locality not in the training data triggers an explicit **warning** ([app/app.py:196-201](app/app.py#L196-L201)) that the locality effect has fallen back to the global average. The model would happily return a confident-looking number; the app refuses to let that pass unremarked.

There's also a **`missing_artifacts_screen`** ([app/app.py:68](app/app.py#L68)) — if `models/` is empty, the app explains how to run `train.py` instead of crashing with a stack trace.

**Alternatives.** **Flask/FastAPI + a frontend** — full control, and perhaps a week of work instead of a day. **Gradio** — similar philosophy, more ML-demo-focused, less flexible for multi-tab dashboards. **Dash** — more powerful for complex dashboards, considerably steeper learning curve. For an internal analytical tool where the audience is small and the point is the model, Streamlit's trade-off is right.

---

### 3.10 The Jupyter stack (dev-only)

Four packages in `requirements-dev.txt`, none needed to train or serve.

| Package | Role |
|---|---|
| **nbformat** 5.11.1 | The `.ipynb` file format as a Python API. `build_notebook.py` constructs the notebook cell by cell. |
| **nbconvert** 7.17.1 | Executes the notebook and embeds outputs; can convert to HTML/PDF. |
| **ipykernel** 7.4.0 | The execution engine that actually runs the cells. |
| **jupyter** 1.1.1 | The metapackage pulling in the interface. |

**The unusual choice: the notebook is *generated*, not hand-written.** [notebooks/build_notebook.py](notebooks/build_notebook.py) is 71 KB of Python that emits a 2.9 MB `.ipynb`. The reasoning ([build_notebook.py:4-8](notebooks/build_notebook.py#L4-L8)): a hand-edited notebook drifts. Someone retrains, the metrics change, and the narrative text still quotes the old numbers. Here, conclusions that quote figures render from `metrics.json` at execution time, so they *cannot* fall out of step. Run `python notebooks/build_notebook.py --exec` and the notebook rebuilds itself against the current model.

The same philosophy drives [make_readme.py](make_readme.py), which generates `README.md` from the same JSON.

**Alternatives.** **Quarto** or **Jupyter Book** for publishing. **papermill** for parameterised execution. **marimo** — reactive notebooks stored as plain `.py`, which would get the same anti-drift property more directly. The generate-from-script approach is unusual but solves the real problem without adding a dependency.

---

### 3.11 Python standard library

No installation, but load-bearing:

| Module | Use |
|---|---|
| **`pathlib`** | Every path in [src/config.py](src/config.py) is a `Path`. `PROJECT_ROOT = Path(__file__).resolve().parents[1]` makes all paths relative to the repo, so the code runs from any working directory and on any OS. |
| **`json`** | `metrics.json` and `feature_meta.json` — the contract between training and every consumer. |
| **`dataclasses`** | `ModelSpec` and `Setting` — declarative config objects, no boilerplate. |
| **`argparse`** | `train.py`'s CLI: `--fast` (1-iteration smoke test), `--n-jobs`, `--perm-repeats`. |
| **`re`** | Floor-string parsing and the `floor_pattern` collapse that reduced ~480 distinct raw strings to a handful of structural shapes. |
| **`collections.Counter`** | Frequency counts during profiling. |
| **`typing`** | Type hints throughout. |
| **`warnings`** | One targeted `filterwarnings` in `train.py` to keep the training log readable. |
| **`sys`**, **`os`**, **`time`**, **`subprocess`** | Path manipulation, environment, timing, notebook execution. |

---

## Part 4 — Modules We Wrote

Six modules under [src/](src/), plus three top-level scripts. The organising principle, stated in [src/\_\_init\_\_.py](src/__init__.py): *every* piece of logic lives here once, so `train.py`, the notebook and the app share one implementation and nothing is duplicated.

### 4.1 [src/config.py](src/config.py) — single source of truth

Paths, column names, feature groups, thresholds, the random seed. Nothing is hard-coded anywhere else.

The most valuable thing in it is **`RANDOM_STATE = 42`**, threaded through every split, every search, every CV shuffle and every model that accepts a seed. Re-running `train.py` reproduces identical numbers. Without a single shared seed, "the metrics changed" becomes unanswerable — you can't tell a real improvement from random variation.

Also here: the **area-type ratios** (carpet 1.00, built-up 0.85, super built-up 0.70), which convert three incompatible measurement conventions into a comparable carpet-area equivalent. `MIN_VALID_SIZE = 50.0`, the plausibility floor. `TEST_SIZE = 0.20`, `N_SPLITS = 5`, `STRATIFY_ON = "City"`. The quantile levels 0.10/0.90. And a comment explaining why `Posted On` is dropped: the listings span only 2022-04-13 to 2022-07-11, about 90 days — too short a window to carry inflation or market drift, so any date-derived feature would be fitting noise.

### 4.2 [src/preprocess.py](src/preprocess.py) — feature engineering

Home to `FeatureEngineer`, the custom transformer, and `build_preprocessor`.

**Why `FeatureEngineer` is a stateful class and not a `FunctionTransformer`.** This is the key design decision in the module ([src/preprocess.py:4-9](src/preprocess.py#L4-L9)). The `Size` and `total_floors` imputations need **group medians**, and those medians must be learned from the training fold *only*. A stateless function would recompute them from whatever frame it was handed — including validation and test rows — leaking information inside cross-validation. So the class learns its medians in `fit` and applies them in `transform`, exactly like any built-in scikit-learn transformer.

It produces: `floor_num` and `total_floors` (parsed out of the `Floor` string), `floor_ratio` (relative height in the building), `carpet_equiv_size` (the area-convention correction), and `size_x_<area type>` interaction terms.

**Why the interaction terms exist, and only for linear models.** A square foot of "super built-up area" means something different from a square foot of "carpet area". A tree can discover that split by itself. A linear model structurally cannot express it without an explicit product term, and would be unfairly penalised in the comparison without one ([src/config.py:113-117](src/config.py#L113-L117)).

**One accommodation for the app.** The transformer accepts *either* a raw frame containing `Floor`, *or* a frame that already carries `floor_num`/`total_floors` as separate columns — because the Streamlit app collects those as two separate inputs. Both callers work against the same fitted pipeline.

**A deliberate trap, documented.** `rent_per_sqft` is computed by `add_eda_features` and used for exploratory analysis and outlier detection **only**. It's derived from the target; using it as a feature would leak the answer completely. The function carries a warning saying so.

### 4.3 [src/data.py](src/data.py) — loading and profiling

`load_data()` validates the CSV header exactly and refuses to guess. There is **deliberately no synthetic fallback** — the comment explains that quietly substituting generated data would invalidate every number in the report. If the file is missing, the error message includes the Kaggle URL and the exact path to put it.

`profile_data()` returns the whole section-4 audit as a dictionary, so `train.py` can print it and the notebook can render it as tables without the two implementations diverging.

### 4.4 [src/models.py](src/models.py) — model definitions

The six `ModelSpec` objects, their search spaces, and the pipeline assembly functions. Also `prefix_params`, which handles a real annoyance: `TransformedTargetRegressor` nests the pipeline under `regressor__`, so `model__reg__alpha` must be re-keyed to `regressor__model__reg__alpha` before `RandomizedSearchCV` will recognise it.

### 4.5 [src/evaluate.py](src/evaluate.py) — metrics

Everything in rupees. Shared CV splitter so models are compared on identical folds. Sign-flipping for scikit-learn's negated scorers. Per-city breakdowns. `format_inr` for Indian digit grouping, imported by both the plots and the app.

### 4.6 [src/plots.py](src/plots.py) — all 18 figures

One module, one style, one `save()`. Centralising this is what keeps the figures consistent across the notebook, the README and the app.

### 4.7 [train.py](train.py) — the orchestrator

~900 lines running the full study end to end: profile → EDA → split → train 6 models × 3 outlier settings → select → diagnose → ablate → fit quantile models → save. It prints a sectioned log (captured in `reports/train_log.txt`) and writes `metrics.json`, which everything downstream reads.

**Model selection is more careful than "lowest error wins"** ([train.py:388-430](train.py#L388-L430)). The deployed configuration is chosen on **test MAE**, not RMSE — because, as we established, RMSE on this data is decided by a handful of extreme listings and ranking on it is unstable. Then, among models within **10% of the best MAE**, four criteria are ranked together:

1. `test_mae`
2. `test_rmse`
3. **`abs_cv_gap`** — the absolute value of the CV overfit gap. Positive means overfitting, but a large *negative* gap isn't bonus generalisation; it means the training folds happened to contain the extreme listings, which is an artifact. Zero in either direction is ideal, hence the absolute value.
4. **`cv_mae_cv`** — the coefficient of variation (std ÷ mean) rather than raw std, so a model isn't rewarded merely for having smaller errors overall.

Gradient Boosting wins on test MAE at **₹10,135** — ahead of Random Forest (₹10,245) and Linear Regression (₹10,760) — even though Linear Regression has the lower RMSE.

*(This corrects something I said earlier in our conversation: selection runs on test MAE with this multi-criteria shortlist, not on CV RMSE.)*

### 4.8 [make_readme.py](make_readme.py) and [notebooks/build_notebook.py](notebooks/build_notebook.py)

Document generators, both reading `metrics.json`. See §3.10.

---

## Part 5 — Technologies Deliberately Not Used

Worth recording, because the absences are choices:

| Not used | Why not |
|---|---|
| **XGBoost / LightGBM / CatBoost** | Almost certainly a point or two of accuracy better than scikit-learn's `GradientBoostingRegressor`. The case study specifies five named algorithms, and staying inside one library keeps the pipeline, the serialisation and the version pin uniform. CatBoost in particular would have handled the 2,235-level locality natively. |
| **A deep learning framework** (PyTorch/TensorFlow) | Neural networks lose to gradient-boosted trees on small tabular data, consistently and by a wide margin. 3,796 training rows is far below where they become competitive. |
| **A database** (PostgreSQL, SQLite, DuckDB) | The input is one 4,746-row CSV. A database would add operational weight and buy nothing. |
| **An experiment tracker** (MLflow, Weights & Biases) | `metrics.json` plus a seeded, deterministic `train.py` gives reproducibility without a service. Worth revisiting if the project grows to many runs. |
| **Docker** | A pinned `requirements.txt` on a known Python version covers reproducibility for a project of this size. |
| **A test suite** (pytest) | The clearest genuine gap. `parse_floor` and `FeatureEngineer` have enough edge-case logic to deserve unit tests. The header validation in `load_data` and the `--fast` smoke-test path partially compensate, but they aren't a substitute. |
| **Type checking** (mypy) | Type hints are present throughout but never machine-verified. |

---

## Appendix — Reproducing the Environment

```bash
# Runtime only (train + app)
pip install -r requirements.txt

# Plus notebook tooling
pip install -r requirements-dev.txt
```

Then:

```bash
python train.py                        # full run; writes models/ + reports/
python train.py --fast                 # 1-iteration smoke test of the structure
python make_readme.py                  # regenerate README.md from metrics.json
python notebooks/build_notebook.py --exec   # rebuild + execute the notebook
streamlit run app/app.py               # launch the UI
```

`train.py --fast` shrinks every search space to a single iteration. It verifies the whole script's structure in seconds and is **never** used for reported numbers.
