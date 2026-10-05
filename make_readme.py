#!/usr/bin/env python3
"""Generate README.md from models/metrics.json.

The README quotes a lot of numbers. Generating it from the same JSON the notebook
and the Streamlit app read means those numbers can never drift out of step with
the model that is actually saved in `models/`.

    python make_readme.py        # after python train.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src import config as cfg
from src.evaluate import format_inr


def load() -> dict:
    if not cfg.METRICS_PATH.exists():
        raise FileNotFoundError(
            f"{cfg.METRICS_PATH} not found -- run `python train.py` first.")
    with open(cfg.METRICS_PATH) as f:
        return json.load(f)


def md_table(df: pd.DataFrame, floatfmt: dict | None = None) -> str:
    d = df.copy()
    for col, fmt in (floatfmt or {}).items():
        if col in d.columns:
            d[col] = d[col].map(lambda v: fmt.format(v) if pd.notna(v) else "")
    header = "| " + " | ".join(str(c) for c in d.columns) + " |"
    sep = "|" + "|".join("---" for _ in d.columns) + "|"
    rows = ["| " + " | ".join(str(v) for v in r) + " |"
            for r in d.itertuples(index=False)]
    return "\n".join([header, sep, *rows])


def build() -> str:
    M = load()
    b = M["best_model"]
    prof = M["profile"]
    split = M["split"]
    rs = M["residual_stats"]
    pi = M["prediction_interval"]
    conc = M["error_concentration"]
    loc = prof["locality"]

    comp = pd.DataFrame(M["model_comparison"]).sort_values("test_rmse")
    exp = pd.DataFrame(M["outlier_experiment"])
    pc = pd.DataFrame(M["per_city"])
    ab = pd.DataFrame(M["ablation"])
    perm = pd.DataFrame(M["feature_importance_permutation"])

    k = b["key"]
    rows_by_setting = {s: exp[(exp.setting_key == s) & (exp.model_key == k)].iloc[0]
                       for s in exp.setting_key.unique()}
    best_city = pc.sort_values("rmse_pct_of_mean").iloc[0]
    worst_city = pc.sort_values("rmse_pct_of_mean").iloc[-1]
    rps = M["rent_per_sqft_by_city"]
    lowest_rmse = comp.iloc[0]

    # ---- comparison table ------------------------------------------------ #
    comp_tbl = pd.DataFrame({
        "Model": comp["model"],
        "CV RMSE (mean ± std)": [f"{m:,.0f} ± {s:,.0f}" for m, s
                                 in zip(comp.cv_rmse_mean, comp.cv_rmse_std)],
        "CV MAE (mean ± std)": [f"{m:,.0f} ± {s:,.0f}" for m, s
                                in zip(comp.cv_mae_mean, comp.cv_mae_std)],
        "Test R²": comp.test_r2.map("{:.4f}".format),
        "Test MSE": comp.test_mse.map("{:,.0f}".format),
        "Test RMSE": comp.test_rmse.map("{:,.0f}".format),
        "Test MAE": comp.test_mae.map("{:,.0f}".format),
        "Test MAPE %": comp.test_mape.map("{:.1f}".format),
    })

    exp_tbl = exp.pivot(index="model", columns="setting",
                        values="test_rmse").round(0).reset_index()
    exp_mae_tbl = exp.pivot(index="model", columns="setting",
                            values="test_mae").round(0).reset_index()

    city_tbl = pd.DataFrame({
        "City": pc.City,
        "Test n": pc.n_test,
        "Mean rent": pc.mean_rent.map("{:,.0f}".format),
        "RMSE": pc.rmse.map("{:,.0f}".format),
        "MAE": pc.mae.map("{:,.0f}".format),
        "R²": pc.r2.map("{:.4f}".format),
        "MAPE %": pc.mape.map("{:.1f}".format),
        "RMSE % of mean": pc.rmse_pct_of_mean.map("{:.0f}".format),
        "Unseen localities %": pc.unseen_locality_pct.map("{:.1f}".format),
    })

    perm_tbl = pd.DataFrame({
        "Feature": perm.feature,
        "MAE increase when shuffled": perm.importance_mean.map("{:,.0f}".format),
        "R² drop when shuffled": perm.r2_drop_mean.map("{:.4f}".format),
    })

    ab_tbl = pd.DataFrame({
        "Model": ab.label,
        "Test RMSE": ab.test_rmse.map("{:,.0f}".format),
        "Test MAE": ab.test_mae.map("{:,.0f}".format),
        "Test R²": ab.test_r2.map("{:.4f}".format),
    })

    rps_line = " · ".join(f"**{c}** {v:.1f}" for c, v in
                          sorted(rps.items(), key=lambda kv: -kv[1]))
    rps_ratio = max(rps.values()) / min(rps.values())

    return f"""# Residential Rent Prediction

**Case Study 36 · Semester V Machine Learning**

Predicting the monthly rent of a residential unit from its attributes — size,
bedrooms, floor, furnishing, locality and tenant preference — across six Indian
cities, with a Streamlit app for interactive estimation.

| | |
|---|---|
| **Selected model** | {b['name']} |
| **Trained under** | {b['setting_label']} |
| **Test R²** | {b['test_r2']:.4f} |
| **Test RMSE** | {format_inr(b['test_rmse'])} |
| **Test MAE** | {format_inr(b['test_mae'])} |
| **Test MAPE** | {b['test_mape']:.1f}% |
| **Dataset** | {prof['n_rows']:,} listings · {split['n_train']:,} train / {split['n_test']:,} held-out test |
| **Reproducibility** | `RANDOM_STATE = {M['run']['random_state']}` everywhere |

---

## Quick start

```bash
# 1. Create and activate a virtual environment (Python 3.12 recommended)
python3.12 -m venv .venv
source .venv/bin/activate          # Windows: .venv\\Scripts\\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train everything (~5-10 minutes)
python train.py

# 4. Launch the app
streamlit run app/app.py
```

To rebuild the analysis notebook as well:

```bash
pip install -r requirements-dev.txt
python notebooks/build_notebook.py --exec
python make_readme.py
```

> **scikit-learn is pinned exactly** in `requirements.txt`. The saved pipeline is
> serialised by that exact version; loading it under a different release can break
> unpickling of the `ColumnTransformer` / `TargetEncoder` internals.

### Where the dataset lives

`data/House_Rent_Dataset.csv` — the Kaggle *House Rent Prediction Dataset*
({prof['n_rows']:,} rental listings across six Indian cities), with
`data/Dataset Glossary.txt` alongside it. The raw CSV is gitignored; download it
from
[Kaggle](https://www.kaggle.com/datasets/iamsouravbanerjee/house-rent-prediction-dataset)
and place it at that path. `src/data.py` raises a clear error with these
instructions if it is missing — there is deliberately no synthetic fallback, so
every number in this README refers to the real data.

---

## Project layout

```
.
├── data/                      raw CSV + glossary (CSV gitignored)
├── src/                       all reusable logic, shared by notebook AND app
│   ├── config.py              paths, column schema, seed, thresholds
│   ├── data.py                load_data() + the profiling pass
│   ├── preprocess.py          FeatureEngineer, outlier logic, ColumnTransformer
│   ├── models.py              the six model variants and their search spaces
│   ├── evaluate.py            metrics, CV reporting, diagnostic tables
│   └── plots.py               every figure
├── notebooks/
│   ├── build_notebook.py      generates the notebook
│   └── rent_prediction_analysis.ipynb
├── models/                    pipeline.joblib, quantile_*.joblib, metrics.json,
│                              feature_meta.json
├── reports/
│   ├── figures/               every plot
│   └── *.csv                  every comparison table
├── app/app.py                 Streamlit app
├── train.py                   runs the full training pipeline
├── make_readme.py             regenerates this file from metrics.json
└── requirements.txt
```

All preprocessing and feature logic lives in `src/` and is imported by the
notebook, `train.py` and the app. Nothing is duplicated between them.

---

## Method

**Target.** Raw rent has a skew of **{prof['rent']['skew']:.1f}** (median
{format_inr(prof['rent']['median'])}, maximum
{format_inr(prof['rent']['max'])} — **{prof['rent']['max'] / prof['rent']['median']:.0f}×
the median**). Every model is trained on `log1p(Rent)`, which brings the skew to
{prof['rent']['skew_log1p']:.2f}. The estimator is wrapped in a
`TransformedTargetRegressor`, so predictions are inverted with `expm1` and **all
reported metrics are in rupees**.

**Feature engineering** (`src/preprocess.py`, a stateful transformer so group
medians are learned on training folds only):

* `Floor` — the string `"Ground out of 2"` is split into `floor_num` and
  `total_floors`, with `Ground` → 0, `Upper Basement` → −0.5,
  `Lower Basement` → −1. {prof['floor']['missing_total_floors']} rows have no
  "out of" part and so have a genuinely missing `total_floors`, imputed with the
  median for that city. **This is the "handle missing floor values" objective —
  the missing values were hidden inside the string, not stored as NaN.**
* `Size` — {prof['size']['below_min_valid']} listings report under
  {prof['size']['min_valid_threshold']:.0f} sq ft (the smallest is
  {prof['size']['min']:.0f}). These are treated as missing and imputed with the
  median for the same `(City, BHK)` group.
* `carpet_equiv_size` — `Size` converted to a carpet-area equivalent, with raw
  `Size` and `Area Type` deliberately **kept as features too**.
* `Size × Area Type` interaction terms, added **for the linear and polynomial
  models only** — trees learn that split themselves.

**Encoding.** One-hot for `City`, `Furnishing Status`, `Tenant Preferred`,
`Area Type`, `Point of Contact`, with `handle_unknown="ignore"` (required: two
levels have only 1–2 rows in the whole dataset). `Area Locality` has
{loc['n_unique']:,} levels of which {loc['appearing_once']:,} appear exactly once,
so it is **target-encoded with smoothing** rather than one-hot.

> **Leakage guards.** (1) The `TargetEncoder` sits **inside** the pipeline, so it
> is refitted within every CV fold and never sees the validation fold's target.
> (2) `rent_per_sqft` is derived from the target and is used for EDA and outlier
> detection **only** — it is never a model feature.

**Outliers.** Detected with a **per-city** IQR rule on `Rent` plus an extreme
`rent_per_sqft` flag. Per-city matters: a single global IQR bound of
{format_inr(prof['rent']['p75'] + 1.5 * (prof['rent']['p75'] - prof['rent']['p25']))}
sits barely above Mumbai's median rent and would flag over a third of that city's
listings while flagging almost nothing in Hyderabad.

**Validation.** A 20% test set, split once, stratified by city, **never
outlier-treated and never touched during tuning**. 5-fold `RandomizedSearchCV` on
the remaining 80%.

### Why models are tuned on MAE

The 5-fold CV **RMSE** on this data has a standard deviation of
{min(M['metric_stability']['cv_rmse_noise_ratio'].values()):.0%}–{max(M['metric_stability']['cv_rmse_noise_ratio'].values()):.0%}
of its own mean — it is decided by whichever few extreme listings land in a given
fold. CV **MAE** on the same folds has a std/mean of
{min(M['metric_stability']['cv_mae_noise_ratio'].values()):.0%}–{max(M['metric_stability']['cv_mae_noise_ratio'].values()):.0%}.
Hyperparameters are therefore selected on CV MAE. **R², MSE, RMSE and MAE are all
still reported for every model** — only the tuning criterion changes.

The same effect shows up on the test set: the single worst prediction of
{conc['n_rows']} accounts for **{conc['top_1_share_of_mse']:.0%}** of total MSE,
and the worst 10 for **{conc['top_10_share_of_mse']:.0%}**. Excluding just the
worst 1% of rows moves RMSE from {format_inr(conc['rmse_all'])} to
{format_inr(conc['rmse_excluding_worst_1pct'])}.

---

## Results

### Model comparison — {b['setting_label']}

{md_table(comp_tbl)}

All values in rupees per month except R² and MAPE. CV columns are 5-fold means ±
standard deviation across folds.

> Polynomial Regression is reported as **two rows** — plain degree 2, and degree 2
> with Ridge. The case study names five algorithms; a plain degree-2 expansion
> over ~24 one-hot encoded features produces a few hundred correlated terms and
> overfits badly, so regularisation was added and both are shown to make the
> improvement visible rather than merely asserted.

![Model comparison](reports/figures/model_rmse_comparison.png)

### Residual plots — all models

The case study lists *Residual Plot* as a comparison metric, not just a
best-model diagnostic.

![Residuals for all models](reports/figures/residuals_all_models.png)

### The outlier experiment

Every model under four settings. **All of them are scored on the same untreated
test set** — treating the test set too would make the metrics improve simply
because the test set got easier.

**Test RMSE**

{md_table(exp_tbl)}

**Test MAE**

{md_table(exp_mae_tbl)}

![Outlier experiment](reports/figures/outlier_experiment.png)
![Premium trade-off](reports/figures/outlier_premium_tradeoff.png)

> The CV columns in `reports/outlier_experiment.csv` are **not comparable across
> settings** — each setting cross-validates a different training set, so `remove`
> shows a flatteringly low CV RMSE purely because the hard rows were deleted from
> its own folds. Only the `test_*` columns are comparable.

### Per-city performance

{md_table(city_tbl)}

![Per-city performance](reports/figures/per_city_performance.png)

Best: **{best_city['City']}** at {best_city['rmse_pct_of_mean']:.0f}% RMSE-to-mean
(R² {best_city['r2']:.3f}). Worst: **{worst_city['City']}** at
{worst_city['rmse_pct_of_mean']:.0f}% (R² {worst_city['r2']:.3f}). The drivers are
sample size, rent variance, and the share of test listings sitting in a locality
the model never saw.

### Feature importance

{md_table(perm_tbl)}

![Feature importance](reports/figures/feature_importance.png)

### Locality ablation

{md_table(ab_tbl)}

{M['ablation_conclusion']}

---

## Final Analysis

### 1. Can monthly rent be predicted from property attributes?

**Yes, with useful but clearly bounded accuracy.** {b['name']} reaches
**R² = {b['test_r2']:.4f}** on {split['n_test']:,} held-out listings — about
**{b['test_r2'] * 100:.0f}% of the variance in monthly rent** explained by recorded
attributes alone. Typical error is **{format_inr(b['test_mae'])} MAE**
({b['test_mape']:.1f}% of actual rent), and **{rs['pct_within_25pct']:.0f}%** of
listings are predicted within 25% of their true rent.

The remaining variance is not noise — it is **missing features**. Amenities,
building age, parking and metro proximity all drive rent and none are recorded.

### 2. Which attribute most influences rent?

**Location and size — which ranks first depends on the metric.** By MAE (every
listing weighted equally) **{perm.iloc[0]['feature']}** is strongest: shuffling it
costs {format_inr(perm.iloc[0]['importance_mean'])} of average error. By R² drop
(dominated by expensive listings) `Size` ranks first.

Both are correct, and the disagreement is the finding. For an ordinary flat, which
city and locality it sits in matters more than a few hundred square feet; for a
premium flat, floor area is what separates price tiers. Median rent per square
foot: {rps_line} — a **{rps_ratio:.1f}× spread** across the six cities.

### 3. How does outlier treatment affect the model?

**The log transform helps unambiguously; outlier treatment is a trade-off, not an
improvement.** For {b['name']}, on the identical untreated test set:

| Setting | Test RMSE | Test MAE | Test R² |
|---|---|---|---|
{chr(10).join(f"| {r['setting']} | {format_inr(r['test_rmse'])} | {format_inr(r['test_mae'])} | {r['test_r2']:.4f} |" for r in rows_by_setting.values())}

Removing or capping extreme rents improves the error on ordinary listings and
worsens it on expensive ones — a model trained with premium listings deleted has
never seen a flat of that kind and regresses them towards the ordinary market.

**The trap this experiment exists to expose:** under `remove` the *cross-validated*
RMSE looks dramatically better, because those folds no longer contain the hard
rows. Scoring everything on one untreated test set is what makes the real effect
visible. The case study's premise is confirmed — a few premium properties distort
the regression badly on a raw target — but the right remedy is the **log
transform**, not deletion.

### 4. Does locality dominate the other predictors?

**No — it is the strongest single block, but it does not dominate.** Removing
`Area Locality` and `City` costs
**{(ab.iloc[1]['test_rmse'] - ab.iloc[0]['test_rmse']) / ab.iloc[0]['test_rmse'] * 100:.0f}% of RMSE**
(R² {ab.iloc[0]['test_r2']:.3f} → {ab.iloc[1]['test_r2']:.3f}). But location *alone*
reaches R² **{ab.iloc[2]['test_r2']:.3f}**, only
**{ab.iloc[2]['test_r2'] / ab.iloc[0]['test_r2'] * 100:.0f}%** of the full model's
{ab.iloc[0]['test_r2']:.3f}. Knowing where a flat is tells you more than any other
single fact — but you still need to know how big it is.

### 5. Which algorithm achieves the lowest RMSE?

**{lowest_rmse['model']}**, at **{format_inr(lowest_rmse['test_rmse'])}** test RMSE
(MAE {format_inr(lowest_rmse['test_mae'])}, R² {lowest_rmse['test_r2']:.4f}). Full
ranking in the comparison table above.

Note the caveat established earlier: RMSE on this data is decided by a handful of
listings, so the RMSE ranking is not robust. The deployed model was chosen on
several criteria rather than on RMSE alone — see below.

{("**Why the lowest-RMSE model was not the one deployed.** " + M["selection_tradeoff_note"]) if M.get("selection_tradeoff_note") else ""}

### 6. How well does the model perform on new listings?

On the {split['n_test']:,}-listing held-out test set, never seen during training
or tuning:

* **MAE {format_inr(b['test_mae'])}**, MAPE {b['test_mape']:.1f}%
* **{rs['pct_within_10pct']:.0f}%** predicted within 10% of actual rent,
  **{rs['pct_within_25pct']:.0f}%** within 25%
* Best city **{best_city['City']}** ({best_city['rmse_pct_of_mean']:.0f}% RMSE-to-mean),
  worst **{worst_city['City']}** ({worst_city['rmse_pct_of_mean']:.0f}%)
* Mean residual {format_inr(rs['mean'])} — no systematic over- or under-prediction
* 80% prediction interval achieves **{pi['empirical_coverage']:.1%}** empirical
  coverage, median width {pi['median_width_pct_of_rent']:.0f}% of actual rent

Error grows with rent: the model controls *proportional* error, so a constant
percentage becomes a larger rupee error at the top of the market.

### 7. Can the model be deployed as a rent estimation tool?

**Yes — as a decision aid with an explicit uncertainty band, not as a valuation
engine.**

Supporting it: the whole preprocessing chain is inside one pipeline, so the app
loads a single artifact and does **no preprocessing of its own** (no
training/serving skew); single-row prediction takes
**{b['predict_latency_ms']:.1f} ms**; it ships a real 80% prediction interval
rather than a bare point estimate; and it is reproducible end to end.

Against it: six cities only; a {prof['posted_on']['span_days']}-day snapshot of
2022; asking price ≠ negotiated rent; premium properties systematically
under-predicted; unseen localities silently degrade to a city-level estimate.

**Verdict:** deployable as a suggested-rent field with the range shown and limits
disclosed. **Not** deployable as an automated pricing engine.

### Rubric section 7 — the five determinations

| Determination | Answer |
|---|---|
| Attributes with the greatest effect on rent | Location (`City`, `Area Locality`) and `Size`; rank depends on MAE vs R² weighting |
| Best-performing algorithm | {b['name']} (selected on multiple criteria; lowest RMSE was {lowest_rmse['model']}) |
| Effect of outlier treatment | Log transform helps unambiguously; removal/capping trades typical-listing accuracy for premium accuracy |
| Does locality dominate all other attributes | No — strongest single block, ~{ab.iloc[2]['test_r2'] / ab.iloc[0]['test_r2'] * 100:.0f}% of explained variance on its own |
| Limitations across different cities | Error ranges from {best_city['rmse_pct_of_mean']:.0f}% to {worst_city['rmse_pct_of_mean']:.0f}% of mean rent by city |

---

## Model selection justification

{md_table(pd.DataFrame(M['selection_table']).round(2))}

{md_table(pd.DataFrame(M['selection_rank_detail']))}

Shortlisted within 10% of the best test MAE: {', '.join(M['shortlist'])}.
**Selected: {b['name']}.**
{"**Runner-up: " + M['runner_up']['name'] + "** — test RMSE " + format_inr(M['runner_up']['test_rmse']) + ", MAE " + format_inr(M['runner_up']['test_mae']) + ", R² " + f"{M['runner_up']['test_r2']:.4f}" + "." if M.get('runner_up') else ""}

**Two details in the ranking.** The CV gap is ranked on its **absolute** value —
a large *negative* gap is not extra generalisation, it means the training folds
happened to hold the extreme listings, and the ideal is a gap near zero either
way. CV stability is ranked on std ÷ mean rather than raw standard deviation, so
a model is not rewarded merely for having smaller errors overall.

**On the train-vs-test gap.** The raw gap is misleading here: the single
{format_inr(prof['rent']['max'])} listing falls in the *training* split, inflating
train RMSE by one row so the gap can come out negative for a model that is not
underfitting. The **CV gap** column — 5-fold validation RMSE minus 5-fold training
RMSE on the same data — is the sound generalisation check and is what the
selection ranks on.

**Interpretability.** Linear and polynomial models expose signed coefficients that
can be read directly, which matters when explaining an estimate to a landlord.
Tree ensembles need permutation importance or SHAP to be explained at all.

**Prediction speed.** Every candidate predicts a single row in single-digit to
low-tens of milliseconds — comfortably interactive, so speed does not discriminate
between them here. It would matter for bulk scoring of a whole listings database.

---

## Limitations

1. **Small dataset** — {prof['n_rows']:,} listings, {split['n_train']:,} used for
   training. Thin evidence for a problem with this much local variation.
2. **Thin per-locality coverage** — {loc['n_unique']:,} distinct localities, of
   which {loc['appearing_once']:,} appear exactly once
   ({loc['share_of_rows_in_singleton_localities']:.0%} of all rows).
3. **Only six cities** — Mumbai, Delhi, Bangalore, Chennai, Hyderabad, Kolkata.
   Nothing here transfers to Pune, Ahmedabad or any tier-2 city.
4. **A {prof['posted_on']['span_days']}-day window**
   ({prof['posted_on']['min']} to {prof['posted_on']['max']}) — no inflation, no
   seasonality, no market drift, and the rupee levels are from 2022.
5. **Missing features** — no amenities, building age, parking, metro proximity,
   security, society quality or view. The main reason two similar-looking flats
   get the same estimate when their real rents differ.
6. **Listing price ≠ negotiated rent** — the target is the asking price,
   systematically above what tenants actually pay.
7. **Unseen localities fall back to the global average**, and the prediction
   interval does not currently widen to signal it.
8. **Outlier treatment makes premium predictions unreliable** — quantified in the
   banded MAE table.
9. **`Area Type` is reported inconsistently** — the observed super-to-carpet size
   ratio is ≈1.0, not the conventional 0.70, so sizes are not measured on a
   consistent basis across listings.
10. **Accuracy varies sharply by city** — a single headline number overstates
    performance in the weaker cities.

---

## Real-world applicability

**Who would use it.** Landlords sanity-checking an asking price; tenants judging
whether a listing is fairly priced; rental platforms pre-filling a suggested rent,
flagging listings far outside the model's range, or detecting bait listings.

**How it would be integrated.** As a suggested-rent field in the listing creation
flow — showing the *range*, always overridable, labelled as an estimate from
comparable listings. Server-side it is one joblib artifact behind a thin endpoint;
the pipeline design means serving code builds a DataFrame and calls `predict`,
with no preprocessing to keep in sync.

**Retraining cadence.** Monthly is appropriate, quarterly the absolute minimum.
The full run takes {M['elapsed_minutes']:.0f} minutes, so cost is not the
constraint. Trigger on calendar schedule *and* on drift detection — monitor
prediction-versus-actual error on new listings and retrain early if it shifts.

**What data would most improve it.**
1. **Geocoordinates instead of free-text locality names** — this alone would fix
   the unseen-locality problem, since an unknown point still has neighbours.
2. **Amenities and building age** — the largest block of missing signal.
3. **Actual transacted rent** rather than asking price — removes a systematic
   bias rather than just reducing variance.
4. **Time-to-let** as a demand signal.
5. **More history**, so seasonality and inflation become learnable.

**What could go wrong.** *Anchoring* — if landlords price to the model's
suggestion, next year's training data reflects the model's own predictions and it
ends up measuring itself. *Entrenched inequity* — thin-data localities are pulled
towards the global average, so systematic under-estimation in underserved areas
could suppress prices there while the model looks accurate on aggregate. *False
precision* — a single rupee figure implies confidence the model does not have,
which is why the app shows an interval. *Premium-segment failure* — the segment
where the most money is at stake is the one predicted worst. *Silent staleness* —
nothing in a deployed model announces that its training data is years old.

---

## Deploying on Streamlit Community Cloud

1. Push the repository to GitHub. **The raw CSV is gitignored**, so either commit
   it deliberately (remove the `data/House_Rent_Dataset.csv` line from
   `.gitignore`) or have the app read it from a URL — the app needs
   `models/*.joblib` and `models/*.json`, which are produced by `train.py`.
2. Commit the trained artifacts: `models/pipeline.joblib`,
   `models/quantile_low.joblib`, `models/quantile_high.joblib`,
   `models/metrics.json`, `models/feature_meta.json`, and `reports/figures/`.
   `models/*.joblib` is gitignored by default — force-add it with
   `git add -f models/*.joblib`, or run `train.py` as part of the build.
3. At [share.streamlit.io](https://share.streamlit.io) create a new app pointing
   at `app/app.py` on your branch.
4. Set the Python version to **3.12** and let it install `requirements.txt`.

> **The one deployment gotcha.** `models/pipeline.joblib` references
> `src.preprocess.FeatureEngineer`, so the project root must be importable before
> `joblib.load` runs. `app/app.py` already inserts it into `sys.path` at the top —
> keep that, or the app fails with `ModuleNotFoundError: No module named 'src'`
> while working perfectly on your machine.

---

## Reproducibility

`RANDOM_STATE = {M['run']['random_state']}` is threaded through the train/test
split, every `RandomizedSearchCV`, the CV splitters, the `TargetEncoder`'s
internal cross-fitting and every model that accepts a seed. Re-running
`python train.py` reproduces these numbers exactly, on
scikit-learn {json.load(open(cfg.FEATURE_META_PATH))['sklearn_version'] if cfg.FEATURE_META_PATH.exists() else 'the pinned version'}.
"""


if __name__ == "__main__":
    text = build()
    (cfg.PROJECT_ROOT / "README.md").write_text(text)
    print(f"wrote README.md ({len(text.splitlines())} lines)")
