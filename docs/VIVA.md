# VIVA Preparation — Residential Rent Prediction

> **Written for:** you, to be drilled before the viva.
> Every answer is backed by a real number from this project's `models/metrics.json` or `reports/`.
> Companion document: [Project_Details.md](Project_Details.md)

---

## How to use this document

- **Sections A–I** are the standard WHY / WHAT / HOW / WHERE questions, grouped by topic.
- **Section J** is the complete formula bank — memorise it.
- **Section K** is a "where is the code for X?" lookup table.
- **Section L** is graph-reading.
- **Section M** is scenario questions ("if this happened, what would you do?").
- **Section N** is **trap questions** designed to derail you, with the correct response.
- **Section O** is **alternative framings** — the same questions worded differently to confuse you.
- **Section P** is rapid-fire one-liners.
- **Section Q** is what *you* should volunteer to steer the viva onto your strong ground.

**The golden rule:** every claim you make should be followed by a number. "I used the log transform
because the data was skewed" is a weak answer. "I used the log transform because the raw skew was
21.41 and it dropped to 0.909" is an unassailable one.

---

## SECTION A — The Absolute Basics

### A1. Is this project based on supervised or unsupervised learning?
**Supervised learning — specifically, regression.**

Three reasons:
1. Every training row has a **known label**: the `Rent` column.
2. The target is **continuous** (₹1,200 to ₹35,00,000), not a category.
3. The objective is to learn a mapping `f: property attributes → rent` and apply it to new,
   unlabelled listings.

### A2. Why not unsupervised? What would unsupervised learning give you here?
Unsupervised learning finds structure in **unlabelled** data. Here the label already exists and is
precisely what we want to predict — so throwing it away would be throwing away the entire signal.

Clustering could group similar listings into market segments, but it could **never output a number
in rupees**, which is what the business question demands.

**Where unsupervised *would* be legitimate:**
- **Clustering localities into tiers** to reduce the 2,235-level cardinality — a genuine
  alternative to target encoding.
- **PCA** on the encoded features for dimensionality reduction (unnecessary here: only ~24
  features).
- **Anomaly detection** (Isolation Forest, DBSCAN) as an alternative outlier detector to the IQR
  rule.

### A3. Is this regression or classification? Why?
**Regression**, because the target is a continuous real number. Classification predicts a
**discrete class label**.

### A4. Could you reframe this as classification?
Yes — by binning rent into bands. The project already defines them in `evaluate.rent_band()`:
`< ₹15,000`, `₹15,000–40,000`, `₹40,000–1,00,000`, `> ₹1,00,000 (premium)`.

**But it would be worse**, for three reasons:
1. **Information loss.** ₹14,999 and ₹15,001 become different classes, while ₹15,001 and ₹39,999
   become identical.
2. **Arbitrary boundaries.** Who decides the cut points?
3. **A landlord wants a number**, not a band.

**When would classification be the right call?** If the deliverable were "flag suspiciously
over-priced listings" — that is genuinely a binary problem.

### A5. What are the independent and dependent variables?
- **Dependent variable (target, response, label, y):** `Rent` — monthly rent in ₹.
- **Independent variables (features, predictors, regressors, X):** 10 raw attributes — `BHK`,
  `Size`, `Floor`, `Area Type`, `Area Locality`, `City`, `Furnishing Status`, `Tenant Preferred`,
  `Bathroom`, `Point of Contact` — plus 5 engineered: `floor_num`, `total_floors`, `floor_ratio`,
  `carpet_equiv_size`, and the 3 `size_x_*` interaction terms (linear models only).

### A6. How many records and features does your dataset have?
**4,746 rows × 12 columns.** After dropping `Posted On` and separating the target: **10 input
columns**, expanding to **~24 encoded features** after one-hot and target encoding.
Split: **3,796 training / 950 test**.

### A7. What is a model, in one sentence?
A function learned from data that maps inputs to an output, parameterised by values estimated by
minimising a loss function on the training set.

### A8. What is the difference between a parameter and a hyperparameter?

| | **Parameter** | **Hyperparameter** |
|---|---|---|
| Learned from data? | **Yes**, during `fit()` | **No**, set before training |
| Examples here | Linear regression `coef_`, tree split thresholds, leaf values | `learning_rate=0.0121`, `max_depth=5`, `n_estimators=535`, Ridge `alpha` |
| Chosen by | The optimisation algorithm | `RandomizedSearchCV` with 5-fold CV |

### A9. What is training, validation and test data? Which do you have?
| Set | Purpose | In this project |
|---|---|---|
| **Training** | Fit model parameters | 3,796 rows (80 %) |
| **Validation** | Tune hyperparameters, compare models | **Created inside the training set by 5-fold CV** — not a separate physical split |
| **Test** | Final unbiased estimate, touched once | 950 rows (20 %), stratified by city, **never treated, never used in tuning** |

**Why CV instead of a fixed validation split?** With only 3,796 training rows, carving off another
20 % would leave ~3,000 to train on and give a single noisy validation estimate. 5-fold CV uses
every row for validation exactly once and averages 5 estimates.

### A10. What is overfitting and underfitting? Show me an example from YOUR project.

| | Underfitting | Overfitting |
|---|---|---|
| Train error | High | **Low** |
| Test error | High | **High** |
| Cause | Model too simple | Model too complex / too little data |

**My live example of overfitting — Polynomial Regression (degree 2):**
```
Train RMSE  ₹62,695      <- looks fine
Test RMSE   ₹9,17,495    <- catastrophic
Test R²     -223.59      <- 224x worse than predicting the mean
```
~24 encoded features expand to **~324 correlated terms** on 3,796 rows. The model memorised the
training set.

**My example of underfitting:** Linear Regression on **raw** rent (setting 1) reaches R² = 0.582
against 0.792 on the log target. The linear-in-rupees form is too rigid for a target with skew 21.4.

### A11. What is the bias–variance trade-off?
```
Expected error  =  Bias²  +  Variance  +  Irreducible error
```
- **Bias** — error from wrong assumptions (the model is too simple). High bias → underfitting.
- **Variance** — sensitivity to the particular training sample. High variance → overfitting.
- **Irreducible error** — noise in the data. Cannot be reduced by any model.

Increasing model complexity lowers bias and raises variance. The optimum is in between.

**In my project:** Random Forest attacks **variance** (averaging de-correlated trees); Gradient
Boosting attacks **bias** (each tree corrects the previous one's errors); Ridge's `alpha` is a
direct dial on the trade-off.

### A12. What is a loss function? Which did you use?
A function measuring how wrong a prediction is, which training minimises.

- **Squared loss** `L = (y − ŷ)²` — the default for `LinearRegression`, `Ridge`,
  `DecisionTreeRegressor`, `RandomForestRegressor` and `GradientBoostingRegressor`.
- **Pinball / quantile loss** — used by the two prediction-interval models
  (`loss="quantile", alpha=0.1` and `alpha=0.9`).

**Loss vs metric:** the loss is what the optimiser minimises *during training*; the metric is what
*you* report. Here the loss is squared error on `log1p(Rent)`, while the reported metrics are
R²/MSE/RMSE/MAE/MAPE in rupees — they are deliberately different.

### A13. What is a feature? What is feature engineering? What did you engineer?
A feature is an input variable. Feature engineering is creating new features from existing ones.

**Five engineered here:**

| Feature | Formula | Why |
|---|---|---|
| `floor_num` | parsed from the `Floor` string | Extracts a hidden numeric fact |
| `total_floors` | parsed from the `Floor` string | Extracts the second hidden fact |
| `floor_ratio` | `floor_num / total_floors` | **Relative** height — floor 10 of 12 is a penthouse; floor 10 of 40 is mid-rise |
| `carpet_equiv_size` | `Size × ratio(Area Type)` | Makes the three measurement conventions comparable |
| `size_x_<area_type>` ×3 | `Size if type matches else 0` | Lets a linear model learn a different slope per area type |

### A14. What is the difference between a Pipeline and just calling functions?
A `Pipeline` is a **single estimator object** that contains all preprocessing plus the model.

**Three concrete benefits here:**
1. **No leakage.** `cross_validate` clones it per fold, so every learned statistic — group medians,
   one-hot categories, target encodings — is re-fit on that fold's training rows only.
2. **No training/serving skew.** The app loads one file and calls `.predict()`; there is only one
   implementation of the preprocessing, so it cannot drift.
3. **Hyperparameter tuning across the whole graph.** `RandomizedSearchCV` can tune any step.

### A15. Why did you set `random_state=42` everywhere?
**Reproducibility.** It seeds the pseudo-random number generator so the train/test split, the CV
fold assignment, the hyperparameter sampling and the trees' internal randomness are identical on
every run. Without it, every rerun would report different metrics and no number in my report would
be verifiable.

42 has no mathematical significance — it is a convention (*The Hitchhiker's Guide to the Galaxy*).
Any fixed integer works.

### A16. Does your result depend on the seed? Isn't that a weakness?
Yes, slightly — any finite-sample result does. Honest mitigations in this project:
- **5-fold cross-validation** averages over 5 different splits, and the fold **standard deviation
  is reported** (CV RMSE = ₹49,039 ± ₹39,836).
- The split is **stratified by city**, which removes the largest source of split-to-split variation.
- The large CV std is itself reported as a finding, not hidden.

**The honest extension:** repeat the whole experiment over several seeds and report a distribution.
That was not done here because 24 model fits already take 8.45 minutes, so a 10-seed study would be
~85 minutes — feasible, and the right next step.

### A17. What is the train–test split ratio and why 80/20?
**80/20** → 3,796 train / 950 test.

**Why:** with only 4,746 rows, you need enough training data for the model to learn and enough test
data for the metric to be stable. 950 test rows gives a standard error on the mean small enough to
distinguish models. 90/10 would leave only 475 test rows (too noisy); 70/30 would waste 475
training rows the model badly needs.

### A18. Why did you stratify the split by `City`?
City counts are very uneven — Mumbai 972, Kolkata 524. A purely random split could badly
under-represent Kolkata in the test set, and the per-city performance table would then be computed
on too few rows to mean anything. Stratifying preserves each city's proportion in both halves.

### A19. What does `n_jobs=-1` do?
Uses **all available CPU cores** via joblib's parallel backend. Applied to `RandomizedSearchCV`,
`cross_validate` and `permutation_importance`.

**Note the deliberate exception:** `RandomForestRegressor(n_jobs=1)`. The surrounding search already
uses all cores; letting the forest parallelise too creates **nested thread pools that
oversubscribe the CPU and run slower, not faster.**

### A20. How long does training take?
**8.45 minutes** for the complete run: 24 model fits (6 models × 4 settings), 3 ablation fits, 2
quantile models, permutation importance with 10 repeats, and 19 figures.

`python train.py --fast` runs the same structure with 1 search iteration per model, in seconds —
used to smoke-test the script before committing to the full run.

---

## SECTION B — Dataset and EDA

### B1. Where did your data come from?
Kaggle — **"House Rent Prediction Dataset"** by Sourav Banerjee. Real rental listings from six
Indian cities, stored at `data/House_Rent_Dataset.csv`. The column definitions are in
`data/Dataset Glossary.txt`.

### B2. Did you use synthetic data?
**No.** The case study permits it as a fallback, but `src/data.py` deliberately has **no synthetic
generator**. Its docstring states the reason: *"quietly substituting generated data would invalidate
every number in the report."* If the CSV is missing, it raises `FileNotFoundError` with download
instructions.

### B3. How many missing values does your dataset have?
**Zero explicit NaNs and zero duplicate rows** — but that number is misleading, and finding out why
is the point of the profiling step. Three genuine problems are hidden:

1. **`Size` has impossible values.** Minimum is **10 sq ft**; 16 rows are under 50 sq ft. These are
   data-entry errors, blanked to NaN by `MIN_VALID_SIZE = 50.0` and imputed.
2. **`Floor` hides missingness inside a string.** 4 rows read `"3"` or `"Ground"` with no
   `"out of N"` part, so `total_floors` is genuinely unknown.
3. **`Area Locality` is effectively missing information** for 1,465 localities that appear exactly
   once.

> **This is the strongest single answer in the whole viva.** "Zero missing values" sounds like
> nothing to do; demonstrating that you found three hidden problems anyway shows real diligence.

### B4. What is the distribution of your target variable?
Severely **right-skewed** (positively skewed):
```
min      ₹1,200          median   ₹16,000       mean     ₹34,993
p75      ₹33,000         p99      ₹3,00,000     max      ₹35,00,000
std      ₹78,106         SKEW     21.41         skew after log1p: 0.909
```
**Mean is 2.2× the median** and the **max is 219× the median** — classic right skew.

### B5. What is skewness? What does 21.41 mean?
```
            1    n   ( x_i - x̄ )³
Skewness = ---  Σ   ------------
            n   i=1       s³
```
- **Skew = 0** → symmetric (e.g. normal)
- **Skew > 0** → right/positive skew — long tail to the right, **mean > median**
- **Skew < 0** → left/negative skew — **mean < median**

Anything beyond |2| is considered severely skewed. **21.41 is extreme.** A handful of ₹10-lakh+
listings drag the mean far above the median. After `log1p` it is **0.909** — nearly symmetric.

### B6. What is kurtosis?
A measure of tail heaviness.
```
            1    n   ( x_i - x̄ )⁴
Kurtosis = ---  Σ   ------------        (excess kurtosis = this - 3)
            n   i=1       s⁴
```
Normal distribution → excess kurtosis 0. **My residuals have kurtosis 185.09** — extraordinarily
heavy tails, i.e. a few enormous errors.

### B7. Which EDA plots did you make and what did each show?
| Figure | Insight |
|---|---|
| `eda_rent_distribution.png` | Skew 21.41 raw vs 0.909 logged — **justifies the log transform** |
| `eda_rent_by_city.png` | Mumbai's distribution barely overlaps Kolkata's |
| `eda_rent_vs_size.png` | Rent rises with size, but the **slope is city-specific** |
| `eda_avg_rent_by_category.png` | Furnished > Semi > Unfurnished; Agent-listed > Owner-listed |
| `eda_correlation_heatmap.png` | Size, BHK and Bathroom are mutually correlated |
| `eda_listings_per_city.png` | Mumbai 972 … Kolkata 524 |
| `eda_outlier_per_city_vs_global.png` | **A global IQR rule flags 37.8 % of Mumbai** |

### B8. What was your single most important EDA finding?
> **Rent scales with size, but the scaling factor is city-specific.**

Median rent per square foot:
```
Mumbai 71.4 | Delhi 28.6 | Bangalore 17.5 | Chennai 16.2 | Hyderabad 14.5 | Kolkata 13.3
```
**Mumbai is 5.4× Kolkata per square foot.** Three consequences follow directly:
1. `City` is the most important feature (permutation importance ₹7,290).
2. Outlier bounds must be **per-city**, not global.
3. Size and City **interact**, which is why the ablation shows A ≠ B + C.

### B9. Why did you drop the `Posted On` column?
I profiled it first:
```
range: 2022-04-13 -> 2022-07-11    span: 89 days    distinct dates: 81
```
**89 days is far too short to carry inflation, seasonality or market drift.** Any date-derived
feature would fit mid-2022 noise rather than signal. I profiled it, stated the reason, and dropped
it — rather than silently discarding it or blindly keeping it.

### B10. Why is `Area Locality` a problem?
```
2,235 distinct localities
1,465 appear exactly ONCE  (30.9 % of all rows)
1,794 appear twice or fewer
```
On 3,796 training rows, one-hot encoding would create **more columns for that one feature than
there are rows**. It would overfit catastrophically and blow up memory. Hence target encoding.

### B11. What correlation did you find between features?
`Size`, `BHK` and `Bathroom` are strongly mutually correlated — bigger flats have more bedrooms and
more bathrooms. This is visible in `eda_correlation_heatmap.png`.

**It matters for interpretation:** permutation importance **under-credits correlated features**,
because shuffling `BHK` leaves most of its information available through `Size` and `Bathroom`.
That is why `BHK` scores only ₹279 — not because bedroom count is irrelevant to rent, but because
the model can recover it elsewhere.

### B12. What is Pearson correlation? What is its formula and range?
```
              Σ (x_i - x̄)(y_i - ȳ)              Cov(X, Y)
    r   =  ----------------------------   =   -------------
           √[Σ(x_i - x̄)²]  √[Σ(y_i - ȳ)²]        σ_X σ_Y
```
Range **[−1, +1]**. +1 = perfect positive linear relationship, 0 = no *linear* relationship,
−1 = perfect negative.

**Critical caveat:** Pearson measures **linear** association only. r ≈ 0 does **not** mean "no
relationship" — a perfect parabola has r = 0. And **correlation is not causation**.

**Spearman** is the rank-based alternative; it captures any **monotone** relationship and is robust
to outliers. On a target with skew 21.4, Spearman would arguably be more informative than Pearson.

---

## SECTION C — Preprocessing

### C1. How did you handle missing values?
Three mechanisms, each matched to the problem:

**1. `FeatureEngineer` — group-wise median imputation (the main one).**
```
Size:          (City, BHK) median  ->  City median  ->  global median
total_floors:  City median         ->  global median
floor_num:     global median
```
The cascade uses the **most specific evidence available**. A missing 2 BHK size in Mumbai gets the
Mumbai-2BHK median, not a nationwide average.

**2. `SimpleImputer` inside the `ColumnTransformer` — the safety net.**
| Strategy | Columns | Why |
|---|---|---|
| `median` | numeric | Robust to the skew |
| `most_frequent` | one-hot categoricals | The mode is the only sensible category fill |
| `constant, "__missing__"` | `Area Locality` | A missing locality is **informative** — it becomes its own category |

**3. Blanking implausible values.** `Size < 50 sq ft` → set to NaN → imputed.

### C2. Why median and not mean for imputation?
**Median is robust to the skew.** The mean `Size` is dragged upward by 8,000 sq ft penthouses; the
median is not. With a target skewed at 21.41 and sizes ranging 10–8,000 sq ft, mean imputation
would systematically inject values that are too high.

### C3. Why impute instead of dropping rows?
Dropping is only acceptable when very few rows are affected **and** the missingness is random.
Here:
- Dropping any row with an implausible `Size` discards 16–91 rows of a 4,746-row dataset — valuable
  data when you have so little.
- The missingness is **not** random: rows with a bare `"Ground"` floor string may differ
  systematically from the rest.
- Most importantly, **the deployed app must handle missing inputs at prediction time.** If the
  pipeline could only drop rows, the app would have nothing to return.

### C4. How did you handle the `Floor` column?
It stores **two facts in one string**. I parsed it with a regex split on `\s*out\s+of\s*`:
```
"3 out of 5"               -> floor_num =  3.0, total_floors = 5.0
"Ground out of 2"          -> floor_num =  0.0, total_floors = 2.0
"Upper Basement out of 4"  -> floor_num = -0.5, total_floors = 4.0
"Lower Basement out of 3"  -> floor_num = -1.0, total_floors = 3.0
"3"  /  "Ground"           -> total_floors = NaN -> imputed  (4 rows)
```

### C5. Why those specific numbers for Ground / Upper Basement / Lower Basement?
They preserve the **physical ordering** on a number line:
```
Lower Basement (-1.0) < Upper Basement (-0.5) < Ground (0.0) < Floor 1 (1.0) < ...
```
A linear model reads these as real numbers, so a monotone mapping is essential. The gap of 0.5
between the basements reflects that they are adjacent sub-ground levels rather than full storeys.

### C6. Why did you create `floor_ratio`?
```
floor_ratio = floor_num / total_floors
```
Absolute floor number is ambiguous: **floor 10 of 12 is a penthouse; floor 10 of 40 is mid-rise.**
The ratio captures *relative height in the building*, which is what actually drives a price
premium. A tree would need many splits to express this interaction; one ratio gives it directly.

### C7. How did you encode categorical variables?
Two different strategies, chosen by **cardinality**:

| Method | Columns | Cardinality | Why this method |
|---|---|---|---|
| **One-Hot** | City, Furnishing Status, Tenant Preferred, Area Type, Point of Contact | 3–6 levels | Low cardinality, nominal (no natural order) |
| **Target Encoding** | Area Locality | **2,235 levels** | One-hot would add more columns than rows |

### C8. Explain target encoding and its formula.
Replace each category with a **smoothed estimate of the mean target** for that category:
```
                n_c · ȳ_c  +  m · ȳ
  encoding_c = ─────────────────────
                     n_c  +  m
```
- `ȳ_c` = mean target in category c, `n_c` = count in c, `ȳ` = global mean, `m` = smoothing strength
- Many listings (`n_c` large) → encoding ≈ the locality's own mean
- One listing (`n_c = 1`) → encoding pulled almost entirely to the **global mean**

**Example (global mean ₹34,993):**
```
Bandra West, n=37, own mean ₹1,50,000, m=10  ->  (37·150000 + 10·34993)/47  ≈  ₹1,25,530
A one-off locality, n=1, own mean ₹1,80,000  ->  (1·180000 + 10·34993)/11   ≈  ₹48,176
```
Exactly the behaviour you want for 1,465 singleton localities.

### C9. Target encoding uses the target. Isn't that data leakage?
**It is leakage waiting to happen, and I guarded against it twice:**

1. **The `TargetEncoder` is placed INSIDE the `Pipeline`** (`src/preprocess.py`,
   `build_preprocessor`). So `cross_validate` and `RandomizedSearchCV` clone and re-fit it on every
   fold, and it never sees the validation fold's target. Fitting it once on the full training set
   before cross-validating would leak into **every** CV score.

2. **sklearn's `TargetEncoder` additionally cross-fits internally during `fit_transform`.** It
   splits the training data into k folds, and the encoding written into fold *i* is computed from
   folds *≠ i*. So even the **training-fold** encodings are out-of-fold. That is why I pass an
   explicit `cv=KFold(n_splits=5, shuffle=True, random_state=42)` — to make that internal
   cross-fitting reproducible.

The code carries a `CRITICAL -- LEAKAGE` comment saying *"Do not hoist this out of the pipeline."*

### C10. Why `drop="first"` on one-hot for linear models but not for trees?
To avoid the **dummy variable trap**. With all *k* indicator columns present, they sum to 1 for
every row, which is **perfectly collinear with the intercept**. Then `XᵀX` is singular, the normal
equation has no unique solution, and coefficients become unstable or infinite.

**Trees do not care** — they split on one column at a time and never invert a matrix — so they keep
all *k* columns, which also makes the feature-importance table easier to read.

### C11. Why `handle_unknown="ignore"`?
**Because it is mandatory on this dataset, not defensive tidiness.** `Built Area` has **2 rows** and
`Contact Builder` has **1 row** in the whole dataset. After an 80/20 split, such a level can appear
in the test set having never been seen in training. Without `handle_unknown="ignore"` the encoder
raises `ValueError` at predict time. With it, the unseen level produces an all-zero indicator row.

The same applies in the deployed app whenever a user picks an unusual combination.

### C12. Why scale features for linear models but not for trees?
**Linear models need it for three reasons:**
1. Numerical conditioning — `Size` spans 10–8,000 while `BHK` spans 1–6.
2. **Ridge's penalty `α Σ βⱼ²` is scale-dependent.** Unscaled, the penalty punishes a small-scale
   feature far more than a large-scale one. **Ridge without scaling is a bug.**
3. Degree-2 expansion of unscaled features gives terms like `Size² ≈ 64,000,000`, destroying
   conditioning.

**Trees do not need it** because a tree splits on `feature <= threshold`. Any **monotone**
rescaling maps the threshold along with the data and produces the **identical** tree. Scaling a
tree's inputs is pure wasted computation.

### C13. What is the difference between StandardScaler, MinMaxScaler and RobustScaler?
| Scaler | Formula | Range | Best when |
|---|---|---|---|
| **StandardScaler** | `(x − μ) / σ` | unbounded, mean 0 std 1 | Roughly normal data; **used here** |
| **MinMaxScaler** | `(x − min)/(max − min)` | [0, 1] | Bounded data, neural nets; **destroyed by a single outlier** |
| **RobustScaler** | `(x − median) / IQR` | unbounded | **Heavy outliers** |
| **Normalizer** | scales each *row* to unit norm | — | Text/TF-IDF vectors |

**Would RobustScaler have been better here?** Arguably yes for the raw-target setting, given the
outliers. But the *features* (Size, BHK, Bathroom) are far less extreme than the *target*, and the
target is handled by `log1p` instead. A fair thing to concede if asked.

### C14. What is the `Size × Area Type` interaction and why did you add it?
`Size` is reported under three conventions that are **not directly comparable**. Three columns:
```
size_x_carpet_area = Size if Area Type == "Carpet Area" else 0
size_x_built_area  = Size if Area Type == "Built Area"  else 0
size_x_super_area  = Size if Area Type == "Super Area"  else 0
```
Exactly one is non-zero per row, letting a linear model learn a **different slope per area type**.

**Added for linear models only**, because a tree given `Size` and the one-hot `Area Type` can learn
`if Area Type == Super and Size > 1200` by itself in two splits. **Giving each model family the
preprocessing it needs is what makes the comparison fair** — otherwise I would be comparing a
handicapped linear model against a well-served tree.

### C15. Did you verify the conventional area-conversion ratios?
Yes — `preprocess.empirical_area_ratios()` compares, within each (City, BHK) group, the median
reported `Size` of each area type against the median for Carpet Area listings. Two findings:

1. **`Built Area` has only 2 rows in the entire dataset**, so no group reaches the minimum count and
   its ratio **cannot be estimated empirically at all**. The conventional 0.85 is used as-is.
2. **For `Super Area` the observed ratio is close to 1.0, NOT the conventional 0.70.** Within
   comparable flats, super-area listings report roughly the same number as carpet-area listings.

**What I did about it:** applied the conversion as specified, **but kept raw `Size` and `Area Type`
as features** so the models can ignore it. State the convention, test it, report the disagreement,
let the model decide.

### C16. Why is `rent_per_sqft` not a feature?
**Because it is computed from the target:** `rent_per_sqft = Rent / carpet_equiv_size`.

Using it would hand the model the answer — `Rent = rent_per_sqft × size` reconstructs the target
exactly, giving R² ≈ 1.0 that collapses the instant it meets a real listing, where `Rent` is
unknown **by definition**.

It is used for **EDA and outlier detection only**, is absent from `config.NUMERIC_FEATURES`, and
the `ColumnTransformer`'s `remainder="drop"` makes it physically unreachable. The function
`add_eda_features` carries a `!! LEAKAGE WARNING !!` in its docstring.

### C17. Why is `FeatureEngineer` a custom class and not a simple function?
**Because it is stateful and must learn from the training fold only.**

The `Size` and `total_floors` imputations need **group medians**. A stateless `FunctionTransformer`
would recompute them from whatever frame it was handed — **including validation and test rows** —
which leaks information inside cross-validation.

By inheriting `BaseEstimator` and `TransformerMixin`:
- `fit()` learns the medians and stores them in trailing-underscore attributes
- `transform()` only *applies* them
- `BaseEstimator.get_params()` makes the class **cloneable**, which is what `cross_validate` needs
  to build a fresh unfitted copy per fold

### C18. How does the app provide `floor_num` and `total_floors` without a `Floor` string?
`FeatureEngineer._split_floor()` accepts **either** a raw frame containing `Floor`, **or** a frame
that already carries `floor_num` / `total_floors` as separate columns. Where both exist,
pre-supplied columns take precedence. The Streamlit app collects the two as separate widgets, so
both callers work against the same fitted pipeline.

### C19. What happens if a user enters "floor 10 of a 3-storey building"?
The pipeline silently corrects it:
```python
total_floors = np.maximum(total_floors, np.maximum(floor_num, 1.0))
```
A building cannot be shorter than the floor a flat sits on. Without this, `floor_ratio` would be
3.33, which is physically meaningless. The app also shows a warning explaining that the correction
will be applied.

---

## SECTION D — Outliers

### D1. What is an outlier?
An observation that lies an abnormal distance from the rest of the data. Critically, outliers come
in **two kinds** and they must be treated differently:

| Kind | Example here | Correct treatment |
|---|---|---|
| **Data-entry errors** | `Size = 10 sq ft`; 1 BHK listed as 8,000 sq ft | Remove or fix |
| **Genuine extreme values** | A ₹35,00,000 Mumbai penthouse | **KEEP** — it is real market data |

**Most of this project's extremes are the second kind**, which is exactly why removing them hurt.

### D2. How did you detect outliers?
**Two independent rules**, both computed **per city**:

**Rule 1 — IQR on `Rent`:**
```
Q1 = 25th percentile of Rent within the city
Q3 = 75th percentile of Rent within the city
IQR = Q3 - Q1
lower = Q1 - 1.5 x IQR        upper = Q3 + 1.5 x IQR
outlier if Rent < lower OR Rent > upper
```

**Rule 2 — extreme `rent_per_sqft`** beyond the per-city 0.5th / 99.5th percentile. This catches
listings whose rent is implausible **given their size** — which a rent-only rule misses entirely.

**Result on 3,796 training rows:** 383 flagged (351 by rent IQR, 44 by rent-per-sqft).

### D3. Why 1.5 × IQR?
**Tukey's convention.** For a normal distribution, Q1 − 1.5·IQR and Q3 + 1.5·IQR sit at roughly
±2.7σ, covering about 99.3 % of the data — so ~0.7 % of *normal* data gets flagged. The 3.0×IQR
variant is the "far out" convention reserved for extreme outliers.

### D4. Why per-city IQR instead of a global IQR? ★ KEY ANSWER
**Because I compared both and the global rule is demonstrably wrong on this data.**

| City | n | Per-city flagged | **Global flagged** | per-city % | **global %** |
|---|---:|---:|---:|---:|---:|
| Bangalore | 886 | 91 | 36 | 10.3 | 4.1 |
| Chennai | 891 | 87 | 46 | 9.8 | 5.2 |
| Delhi | 605 | 64 | 68 | 10.6 | 11.2 |
| Hyderabad | 868 | 85 | 26 | 9.8 | 3.0 |
| Kolkata | 524 | 36 | 16 | 6.9 | 3.1 |
| **Mumbai** | **972** | **104** | **367** | **10.7** | **37.8** |

**A single global rule would flag 367 of 972 Mumbai listings (37.8 %) as outliers**, because the
global upper bound sits barely above **Mumbai's median**. Kolkata's 99th percentile is around
₹40,000; Mumbai's median is around ₹50,000. A global rule would delete a large slice of perfectly
legitimate Mumbai listings and call it "cleaning".

The per-city rule flags a sane 7–11 % in every city. **Figure:**
`reports/figures/eda_outlier_per_city_vs_global.png`

### D5. What treatment strategies did you test?
| Strategy | Action | Effect on 3,796 training rows |
|---|---|---|
| `none` | leave untouched | 3,796 rows |
| `remove` | drop flagged rows | 3,796 → **3,413** (383 removed) |
| `cap` (winsorise) | clip `Rent` at per-city P1/P99 | 3,796 rows, **73 clipped** |

Combined with two target treatments (raw vs `log1p`) into **four settings**, each run on all six
models = **24 fits**.

### D6. What is winsorising?
Replacing extreme values with the nearest **percentile boundary** instead of deleting the row:
```
Rent_capped = clip(Rent, lower = P1_city, upper = P99_city)
```
**Advantage over removal:** you keep the row's features. A ₹5,00,000 Mumbai penthouse still
contributes its size, locality and bathroom information; only its rent is pulled inward.

### D7. Did you apply outlier treatment to the test set?
**No — absolutely not, and this is the point that makes the experiment valid.**

`treat_outliers` is called on the **training split only**. Treating the test set too would be
cheating: removing hard cases makes the metrics look better **without the model improving at all**
— the test set simply got easier.

Every one of the 24 model/setting combinations is scored against **the same untreated 950-row test
set**, which is the only reason the comparison across settings is meaningful.

### D8. What was the effect of outlier treatment? ★ THE CORE FINDING
**Test MAE (₹) for Gradient Boosting:**
```
1. Raw rent, no treatment    11,006
2. log1p only                10,135    <- BEST
3a. log1p + remove           12,504    <- WORSE
3b. log1p + cap              10,447
```

**And broken down by rent band — this is where the real story is:**

| Rent band | 1. Raw | **2. log1p** | 3a. remove | 3b. cap |
|---|---:|---:|---:|---:|
| < ₹15,000 | 3,539 | **2,528** | 2,482 | 2,524 |
| ₹15,000 – ₹40,000 | 5,843 | **5,345** | 5,067 | 5,279 |
| ₹40,000 – ₹1,00,000 | 18,036 | **14,798** | 17,360 | 14,531 |
| **> ₹1,00,000 (premium)** | 70,554 | **73,887** | **1,05,137** | 79,468 |

**Removing outliers raised premium-band error by 42 % (₹73,887 → ₹1,05,137) and gained ₹46 on
cheap flats.** That trade is clearly not worth it.

**Conclusion:** *the `log1p` transform alone already handles the skew (21.41 → 0.909). Removing or
capping on top of that trains a model that systematically under-predicts genuinely expensive
properties. The deployed model therefore uses `log1p` with **no** outlier removal — and the
experiment that justifies it is reported in full rather than hidden.*

### D9. Isn't it bad practice to keep outliers?
**No — it is bad practice to remove them without testing.** I ran all four settings and chose on
evidence.

The three reasons keeping them is right here:
1. **They are genuine market data**, not errors. ₹35,00,000/month for a Mumbai sea-facing penthouse
   is a real listing, not a typo.
2. **The `log1p` transform already handles their influence**, pulling skew from 21.41 to 0.909.
   Outlier removal on top is redundant.
3. **The app must serve premium properties too.** A model trained without them would confidently
   quote ₹1,00,000 for a ₹5,00,000 flat.

I *did* remove the genuine errors: 16 listings with `Size < 50 sq ft`.

### D10. How else could you detect outliers?
| Method | How | vs the IQR rule here |
|---|---|---|
| **Z-score** `\|z\| > 3` | `(x − μ)/σ` | **Bad here** — μ and σ are themselves distorted by the outliers |
| **Modified Z-score** | `0.6745(x − median)/MAD` | Robust; a good alternative |
| **IQR (Tukey)** | `Q1 − 1.5·IQR`, `Q3 + 1.5·IQR` | **Used** — robust, no distributional assumption |
| **Isolation Forest** | Isolates points with random splits | Multivariate; catches odd *combinations* |
| **Local Outlier Factor** | Local density deviation | Multivariate, density-based |
| **DBSCAN** | Points in no cluster are noise | Multivariate |
| **Domain rules** | `Size < 50 sq ft` | **Also used** — the most defensible of all |

**Why not Z-score:** it assumes approximate normality and uses the mean and std, both of which are
**themselves destroyed** by a skew of 21.41. IQR uses quartiles, which are robust.

---

## SECTION E — Models (WHY THIS OVER THAT)

### E1. Which algorithms did you use and why those five?
The case study specified them: **Linear Regression, Polynomial Regression, Decision Tree, Random
Forest, Gradient Boosting**. I report **six** variants because Polynomial Regression appears twice
— plain degree 2, and degree 2 with Ridge — since the plain version overfits catastrophically and
showing both demonstrates what regularisation buys.

They also form a deliberate **complexity ladder**:
```
Linear (parametric, high bias)
  -> Polynomial (parametric, more capacity)
    -> Ridge-Polynomial (regularised)
      -> Decision Tree (non-parametric, high variance)
        -> Random Forest (bagging: variance reduction)
          -> Gradient Boosting (boosting: bias reduction)
```

### E2. Which model did you select and why?
**Gradient Boosting Regressor**, trained on `log1p(Rent)` with **no** outlier treatment.

| Metric | Value |
|---|---|
| Test R² | 0.7144 |
| Test RMSE | ₹32,716 |
| **Test MAE** | **₹10,135** — lowest of all six |
| **Test MAPE** | **26.74 %** — lowest of all six |
| **Latency** | **6.53 ms** — fastest of all six |

**Selected by a four-criterion rank-sum**, not by one metric: test MAE, test RMSE, absolute CV
overfit gap, and CV MAE coefficient of variation.

### E3. Linear Regression has a better R² (0.792) and RMSE (₹27,897) than your chosen model. Why didn't you pick it? ★ THE HARDEST QUESTION — KNOW THIS COLD

**Because test RMSE on this dataset is not measuring general accuracy. It is measuring about ten
listings.**

**Error concentration on the 950-row test set:**
```
the  1 worst row  accounts for  42.2 % of total MSE
the 10 worst rows account for   76.9 % of total MSE
RMSE over all rows          : ₹32,716
RMSE excluding the worst 1 %: ₹16,234      <- HALVED
```

**The mechanism is extrapolation.** A linear model fitted in log space keeps extending its trend
beyond the range of rents it saw, so it lands closer on the handful of ₹3–12 lakh listings that
dominate squared error. A tree ensemble **cannot extrapolate at all** — its predictions are bounded
by the leaf values it learned — so it under-predicts those few properties and pays heavily in RMSE,
while fitting the other ~99 % of listings more accurately.

**Evidence that GB is better on the other 99 %:**
```
Test MAE   GB 10,135  vs  Linear 10,760
Test MAPE  GB 26.74 % vs  Linear 28.42 %
```

**And RMSE is too noisy to tune on anyway:** CV RMSE has std/mean of 0.60–0.81 across models, while
CV MAE sits at 0.10–0.15. Refitting hyperparameters on a signal that noisy selects configurations
essentially at random.

**My closing line:** *since the deployed tool estimates rent for ordinary listings, I selected the
model with the lower typical error, and I documented the RMSE trade-off rather than hiding it. The
code prints that trade-off automatically.*

### E4. Why Gradient Boosting over Random Forest?
They are close — GB is the winner, RF the runner-up:

| | **Gradient Boosting** | Random Forest |
|---|---:|---:|
| Test MAE | **₹10,135** | ₹10,245 |
| Test RMSE | **₹32,716** | ₹35,092 |
| Test R² | **0.7144** | 0.6714 |
| Test MAPE | **26.74 %** | 26.75 % |
| **Latency** | **6.53 ms** | 13.25 ms |

GB wins on **every accuracy metric** and is **2× faster** at prediction time, because its trees are
shallow (depth 5) while the forest's are deep. For an interactive web app, latency matters.

**Conceptually:** boosting reduces **bias** by fitting residuals sequentially; bagging reduces
**variance** by averaging. With only 3,796 training rows and a structured tabular problem, the bias
reduction is worth more here.

### E5. Why not XGBoost / LightGBM / CatBoost?
1. **The case study specified `GradientBoostingRegressor`.** Deliverable alignment comes first.
2. **They would likely be faster and marginally better** — XGBoost and LightGBM use histogram-based
   split finding and better regularisation. On 3,796 rows the speed gain is irrelevant.
3. **CatBoost would be genuinely interesting here** because it has built-in **ordered target
   statistics** for categorical features — essentially a principled version of the target encoding
   I implemented manually.
4. They are extra dependencies for a marginal gain on a small dataset.

**This is a great question to answer well — it shows you know the landscape beyond the syllabus.**

### E6. Why not a neural network?
| Reason | Detail |
|---|---|
| **Dataset size** | 3,796 training rows. Neural nets need far more; they would overfit badly. |
| **Tabular data** | Gradient-boosted trees remain state of the art on small/medium tabular problems; this is a well-established empirical result. |
| **Interpretability** | I can report feature importances and a decision path. A net is a black box. |
| **Training cost** | My full pipeline runs in 8.45 minutes on a laptop. |
| **Scope** | Not in the case study. |

**When would a net win?** With hundreds of thousands of listings, unstructured inputs (property
photos, free-text descriptions), or learned embeddings for the 2,235 localities.

### E7. Why not SVR / KNN?
- **SVR** scales poorly (roughly O(n²)–O(n³)), requires careful scaling and kernel choice, and
  gives no feature importances. It is also not in the specification.
- **KNN** is severely harmed by the curse of dimensionality after one-hot encoding, needs a
  meaningful distance metric across mixed numeric/categorical features, and has O(n) prediction
  cost with no interpretability.

### E8. Why Ridge over Lasso?
| | **Ridge (L2)** | **Lasso (L1)** |
|---|---|---|
| Penalty | `α Σ βⱼ²` | `α Σ abs(βⱼ)` |
| Sets coefficients to exactly 0 | **No** | **Yes** |
| Feature selection | No | Yes |
| With correlated features | **Shares weight among them** | Arbitrarily picks one |
| Geometry | Circular constraint region | Diamond with corners on the axes |

**My goal was to control collinearity among ~324 polynomial terms, not to select features.** With
highly correlated expanded terms, Ridge's behaviour of spreading weight across the group is the
right one; Lasso would arbitrarily keep one of each correlated pair and discard the rest.

**What about Elastic Net?** `α(ρ·L1 + (1−ρ)·L2)` — a genuinely reasonable third option that gets
both effects, at the cost of a second hyperparameter to tune. A fair thing to concede.

### E9. Why does Lasso set coefficients to exactly zero but Ridge does not?
**Geometry.** Constrained optimisation finds where the elliptical contours of the RSS first touch
the constraint region.
- **L1's region is a diamond** with **corners on the axes**. A corner means one coordinate is
  exactly zero, and contours are most likely to touch a corner.
- **L2's region is a circle/sphere** with **no corners**, so the touch point almost surely has all
  coordinates non-zero.

### E10. Explain Gradient Boosting to me.
**Build trees sequentially; each new tree is fitted to the errors of the ensemble so far.**

```
1. Start with a constant:  F_0(x) = mean(y)

2. For m = 1 .. M:
   a. Compute pseudo-residuals = negative gradient of the loss:
            r_im = -[ ∂L(y_i, F(x_i)) / ∂F(x_i) ]_{F = F_{m-1}}
      For squared loss this is simply  r_im = y_i - F_{m-1}(x_i)
   b. Fit a regression tree h_m(x) to those residuals
   c. Update with a learning rate ν:
            F_m(x) = F_{m-1}(x) + ν · h_m(x)

3. Final:  F_M(x) = F_0(x) + ν · Σ h_m(x)
```

**Why "gradient":** step (a) is **gradient descent in function space** rather than parameter space.
Each tree is one step in the direction that most reduces the loss.

**Worked example** (true rent ₹50,000, ν = 0.1):
```
F0 = 35,000         residual +15,000   tree1 predicts +12,000  -> F1 = 36,200
F1 = 36,200         residual +13,800   tree2 predicts +11,000  -> F2 = 37,300
F2 = 37,300         residual +12,700   tree3 predicts +10,500  -> F3 = 38,350
...converging slowly towards 50,000
```
**Small steps, many of them.** My tuned model uses learning rate **0.0121** with **535 trees**.

### E11. Explain Random Forest. Why is the "random" part essential?
**Train many de-correlated trees and average them:** `ŷ(x) = (1/B) Σ T_b(x)`.

**Two sources of randomness:**
1. **Bootstrap sampling (bagging)** — each tree trains on n rows drawn **with replacement**. Each
   bootstrap contains ~**63.2 %** of the distinct original rows (`1 − 1/e`); the other ~36.8 % are
   **out-of-bag**.
2. **Random feature subsetting (`max_features`)** — at **every split**, only a random subset of
   features is considered.

**Why the second is essential — the key insight:** bootstrapping alone still gives highly
**correlated** trees, because one dominant feature (here, `City`) would be the root split in nearly
every tree, and **averaging correlated predictors barely reduces variance**. Feature subsetting
forces different trees to explore different structures.

**The mathematics that proves it.** For B identically distributed predictors with variance σ² and
pairwise correlation ρ:
```
                      1 - ρ
Var(average) = ρσ² + ------- σ²
                        B
```
As B → ∞ the second term vanishes, **but `ρσ²` does not** — it is the floor set by tree
correlation. **Therefore lowering ρ matters as much as raising B.** That single formula explains
the entire design of Random Forest.

### E12. Bagging vs Boosting — give me the full comparison.
| | **Bagging (Random Forest)** | **Boosting (Gradient Boosting)** |
|---|---|---|
| Trees built | **In parallel**, independently | **Sequentially**, each depends on the last |
| Each tree sees | A bootstrap sample | Full data, reweighted by residuals |
| Primarily reduces | **Variance** | **Bias** |
| Base learners | Deep, low-bias, high-variance trees | Shallow weak learners (depth 2–7) |
| Combination | Simple average | Weighted additive sum with learning rate |
| Overfits if over-trained? | **No** — more trees never hurt | **Yes** |
| Parallelisable | Yes, trivially | No (inherently sequential) |
| Noise sensitivity | Lower | **Higher** (it chases residuals) |
| Tuning difficulty | Low | Higher (lr × depth × n_estimators) |

### E13. Why did Polynomial Regression fail so badly?
```
Train RMSE  ₹62,695        Test RMSE  ₹9,17,495        Test R² -223.59
```
A test R² of −223.59 means **224× worse than predicting the mean rent for every listing**.

**Three causes:**
1. **~324 correlated terms** from a degree-2 expansion of ~24 encoded features, fitted on 3,796
   rows. Products of one-hot columns are mostly zero and mutually redundant.
2. **Severe multicollinearity** → `XᵀX` near-singular → enormous, unstable coefficients.
3. **Unbounded extrapolation.** A quadratic in log space explodes once an input leaves the training
   range; after `expm1` that becomes an absurd rupee figure.

**My framing:** *the case study specified it, so I ran it and reported it honestly. Its failure is
not a bug — it is the correct, expected result of fitting 324 collinear terms to 3,796 rows, and it
is exactly the demonstration of the bias–variance trade-off that the comparison exists for.*

### E14. Why didn't Ridge fix the polynomial model?
It helped substantially — test R² improved from **−223.59 to −4.49** — but could not rescue it. A
degree-2 expansion of one-hot encoded data is simply the wrong hypothesis class for this dataset.
Shrinking bad features does not make them good features.

### E15. What is `TransformedTargetRegressor` and why use it?
It wraps an estimator so `y` is transformed before fitting and predictions are inverse-transformed:
```
fit(X, y)   ->  inner.fit(X, log1p(y))
predict(X)  ->  expm1(inner.predict(X))
```

**Three benefits over manually logging `y`:**
1. **`y` stays in rupees at the API boundary**, so ordinary scorers report R²/RMSE/MAE in rupees
   with no manual inversion to get wrong.
2. **The `TargetEncoder` inside the `ColumnTransformer` sees the LOG target**, which is far less
   distorted by skew 21.41.
3. **The saved pipeline predicts rupees directly**, so the Streamlit app needs zero
   post-processing.

### E16. Why `log1p` instead of `log`?
`log(0) = −∞`. `log1p(0) = 0`. Even though minimum rent here is ₹1,200, `log1p` is the safe habit
and is numerically more accurate for small x. Its exact inverse is `expm1`.

### E17. What does the log transform actually do to the model?
Three things:
1. **Fixes the skew:** 21.41 → 0.909.
2. **Converts a multiplicative relationship into an additive one:**
   ```
   log(Rent) = β0 + β1·Size   <=>   Rent = e^β0 · e^(β1·Size)
   ```
   Each extra square foot **multiplies** rent by a factor instead of adding a fixed number of
   rupees. That is how rent genuinely behaves.
3. **Changes the implicit loss to relative error.** Squared error in log space ≈ squared
   **relative** error in rupees. On a target spanning ₹1,200 to ₹35,00,000, that is the correct
   objective.

### E18. What is the downside of the log transform?
**Back-transforming a log-space mean introduces a downward bias**, because by Jensen's inequality
`exp(E[log X]) = geometric mean ≤ arithmetic mean = E[X]`.

**My two responses:**
- Empirically the bias is negligible here: **median residual = −₹37.70**.
- **The prediction intervals use quantiles, which ARE transform-exact**, so the uncertainty
  estimate is unaffected.

(The classical correction is Duan's smearing estimator; it was not needed given the measured
residual median.)

### E19. How did you tune hyperparameters?
`RandomizedSearchCV` with 5-fold CV, `refit="mae"`, `random_state=42`.

| Model | n_iter | Searched |
|---|---:|---|
| Linear | 1 | nothing to tune |
| Polynomial | 1 | nothing to tune |
| Polynomial + Ridge | 30 | `alpha ~ loguniform(1e-2, 1e7)` |
| Decision Tree | 40 | `max_depth`, `min_samples_leaf`, `min_samples_split`, `max_features` |
| Random Forest | 25 | `n_estimators`, `max_depth`, `max_features`, `min_samples_leaf` |
| Gradient Boosting | 30 | `learning_rate`, `n_estimators`, `max_depth`, `subsample`, `min_samples_leaf` |

### E20. Why RandomizedSearchCV over GridSearchCV?
**Compute.** A grid over 5 GB hyperparameters with 5 values each is 5⁵ = **3,125** configurations ×
5 folds = **15,625 fits**. Random search samples **30**.

**And it finds better configurations per unit compute.** Bergstra & Bengio (2012): in practice only
a few hyperparameters matter, and random search explores **more distinct values of each** of them.
Grid search wastes most of its budget varying parameters that do not matter.

**Random search also samples from continuous distributions** (`loguniform`, `uniform`), so it can
find `learning_rate = 0.012115` — a value no hand-specified grid would contain.

### E21. Why did you tune on MAE instead of RMSE? ★
**Because I measured the noise in both.**

| Model | CV RMSE std/mean | CV MAE std/mean |
|---|---:|---:|
| Linear Regression | 0.721 | 0.113 |
| Polynomial (deg 2) | 0.599 | 0.101 |
| Polynomial + Ridge | 0.669 | 0.107 |
| Decision Tree | 0.777 | 0.118 |
| Random Forest | 0.802 | 0.145 |
| Gradient Boosting | 0.812 | 0.147 |

**CV RMSE has a standard deviation 60–81 % of its own mean.** It is dominated by whichever handful
of extreme listings happen to fall in a given fold, so refitting on it selects hyperparameters
essentially **at random** — which is exactly how the Polynomial+Ridge model ended up with a
negative test R² on an early pass. **CV MAE sits at 10–15 %: a usable signal.**

**All four metrics are still reported for every model. Only the tuning criterion changed.**

### E22. What hyperparameters did your final model actually get?
```
learning_rate     0.012115      <- near the floor of loguniform(0.01, 0.3): slow, careful
n_estimators      535           <- near the ceiling of randint(150, 600), as expected with a small lr
max_depth         5
subsample         0.754547      <- Stochastic Gradient Boosting
min_samples_leaf  6
```
**The lr/n_estimators pairing is the textbook result:** a small learning rate needs many trees.
They must be tuned **together**, which is why both are in the search space.

### E23. What does `subsample=0.75` do?
Each tree is fitted on a random **75.45 %** of the training rows. This makes it **Stochastic
Gradient Boosting** (Friedman, 2002): the randomness decorrelates successive trees, reduces
variance and acts as a regulariser.

**Note the parameterisation:** `uniform(0.7, 0.3)` means `loc=0.7, scale=0.3` → the range
**[0.70, 1.00]**. The second argument is the **width**, not the upper bound. A very easy thing to
get wrong.

### E24. How do you prevent a Decision Tree from overfitting?
**Pre-pruning (early stopping)** — all four are in the search space:
| Parameter | Searched | Effect |
|---|---|---|
| `max_depth` | `[3,5,7,9,11,14,18,25,None]` | Caps tree depth |
| `min_samples_leaf` | `randint(1,40)` | No leaf smaller than this |
| `min_samples_split` | `randint(2,40)` | Do not split a small node |
| `max_features` | `[None,0.5,0.7,0.9,"sqrt"]` | Features considered per split |

**Post-pruning** (cost-complexity pruning, `ccp_alpha`) grows the full tree then prunes back. Not
used here — pre-pruning via randomised search was sufficient and cheaper.

Without any of these, a tree grows until every leaf is pure: perfect training accuracy, terrible
generalisation.

### E25. What splitting criterion does a regression tree use?
**MSE reduction (variance reduction).** At each node, try every (feature, threshold) pair and pick
the one maximising:
```
                       n_left               n_right
ΔMSE = MSE(parent) -  -------- MSE(left) - --------- MSE(right)
                      n_parent              n_parent

                1
where MSE(S) = ---  Σ (y_i - ȳ_S)²        = the variance within node S
               |S| i∈S
```
**Gini impurity and entropy/information gain are for CLASSIFICATION, not regression.** Know the
difference — examiners test this.

### E26. Why can't tree models extrapolate?
A tree's prediction is the **mean of the training targets in a leaf**. Every possible prediction
is therefore one of a finite set of leaf values learned during training, all of which lie within
`[min(y_train), max(y_train)]`.

Feed it a 20,000 sq ft mansion and it will fall into the "largest" leaf and return that leaf's
mean — it cannot produce a value above anything it has seen.

**This is the direct cause of this project's central finding:** 9 of my 10 worst predictions are
**under**-predictions of premium properties, and it is why Linear Regression (which *can*
extrapolate) beats Gradient Boosting on RMSE.

### E27. Can you say in one sentence why you have six models for five algorithms?
Polynomial Regression is reported twice — plain degree 2 and degree 2 with Ridge — because the
plain version overfits catastrophically (test R² −223.59) and reporting both keeps the deliverable
aligned with the specification while demonstrating what regularisation buys (−223.59 → −4.49).

---

## SECTION F — Evaluation Metrics

### F1. Which metrics did you use and why those?
| Metric | Why |
|---|---|
| **R²** | Required by the spec; "proportion of variance explained" |
| **MSE** | Required by the spec |
| **RMSE** | Required by the spec; same units as the target |
| **MAE** | Required by the spec; **also my primary selection metric** — robust |
| **MAPE** | Added by me; most intuitive answer to "how good is it on a new listing?" |
| **Residual plots** | Required by the spec; diagnoses model *assumptions*, not accuracy |

### F2. Explain R².
```
             SS_res        Σ (y_i - ŷ_i)²
R²  =  1 - --------  = 1 - ---------------
             SS_tot        Σ (y_i - ȳ)²
```
**The proportion of variance in rent that the model explains.**
- 1.0 = perfect · 0.0 = no better than always predicting the mean · **< 0 = worse than the mean**

**Mine is 0.7144** — the model explains 71.4 % of the variance in rent.

### F3. Can R² be negative? Yours showed −223.59 — explain.
**Yes.** `SS_tot` is fixed (it only depends on the data), but `SS_res` is unbounded above. If your
model's squared error exceeds that of the constant-mean baseline, R² goes negative.

My plain Polynomial model scored **−223.59**, meaning its squared error was 224× the baseline's.
**That is not a bug — it is the metric correctly reporting a catastrophically bad model.**

(Note: R² can only be negative on *out-of-sample* data. In-sample OLS R² is always in [0, 1]
because OLS minimises exactly the quantity in the numerator.)

### F4. R² vs Adjusted R²?
```
                   (1 - R²)(n - 1)
Adj R² = 1 - ----------------------------
                   n - p - 1
```
Adjusted R² penalises adding features that do not help. Plain R² **never decreases** when you add a
feature, even a random one.

**Why I report plain R²:** the feature count is essentially fixed across models, so plain R² is
already comparable. Adjusted R² matters when you are comparing models with *different* numbers of
predictors.

### F5. MSE vs RMSE vs MAE — the full comparison.
| | MSE | RMSE | MAE |
|---|---|---|---|
| Formula | `(1/n)Σ(y−ŷ)²` | `√MSE` | `(1/n)Σ\|y−ŷ\|` |
| Units | **₹²** (meaningless) | **₹** | **₹** |
| Outlier sensitivity | **Very high** | **High** | **Low** |
| Differentiable everywhere | Yes | Yes | **No** (kink at 0) |
| Minimiser estimates | conditional **mean** | conditional mean | conditional **median** |
| My value | 1,070,335,994 | 32,716 | **10,135** |

**Always true:** `RMSE ≥ MAE`, with equality only when all errors are identical in magnitude.

### F6. Your RMSE is ₹32,716 but your MAE is ₹10,135. Why the 3× gap? ★
**The ratio RMSE/MAE = 3.23 measures how unequal the errors are.** A ratio above ~1.5 signals a
heavy-tailed error distribution; 3.23 is very large.

**The cause, measured:**
```
the  1 worst test row accounts for 42.2 % of total MSE
the 10 worst test rows account for 76.9 % of total MSE
RMSE over all 950 rows          : ₹32,716
RMSE excluding the worst 1 %    : ₹16,234     <- HALVED
```
One listing out of 950 causes 42 % of the squared error. RMSE squares errors, so it is dominated by
that tail. MAE weights every error linearly and therefore describes the **typical** listing.

**This gap is not a defect in the model — it is the correct description of a heavy-tailed target,
and it is why I rank on MAE.**

### F7. When would you prefer RMSE over MAE?
When **large errors are disproportionately costly**. Examples:
- Structural engineering — a 10× error is more than 10× as bad
- Medical dosing
- A rental platform that **guarantees** its estimate and pays the difference — then a single huge
  miss is genuinely catastrophic and RMSE is the right loss

**For a suggestion tool for landlords**, being typically close matters more than never being badly
wrong, so MAE is the right criterion. **The metric must match the cost function of the
application.**

### F8. Explain MAPE and its weaknesses.
```
            100   n   | y_i - ŷ_i |
MAPE   =   -----  Σ   -------------
             n   i=1      | y_i |
```
**Mine: 26.74 %** — "typically within about a quarter of the true rent."

**Three weaknesses:**
1. **Undefined when `y = 0`.** (Safe here: minimum rent is ₹1,200.)
2. **Asymmetric** — over-prediction error is unbounded, under-prediction is capped at 100 %,
   because the denominator is the actual value.
3. **It therefore biases model selection towards under-prediction.** This is a known pathology and
   is why I report MAPE but do **not** select on it.

**Alternative:** SMAPE (symmetric MAPE) uses `(|y| + |ŷ|)/2` as the denominator.

### F9. What is a residual? What does a good residual plot look like?
```
residual  e_i = y_i - ŷ_i          (actual minus predicted)
```
A residual plot puts **predicted value on x** and **residual on y**.

**Good:** points scattered randomly about y = 0, constant vertical spread, no visible pattern,
roughly symmetric.

| Pattern | Diagnosis | Fix |
|---|---|---|
| **Funnel / cone** | **Heteroscedasticity** | Log-transform the target; tree model; weighted least squares |
| **Curve / U-shape** | Missing non-linearity | Polynomial terms, interactions, non-linear model |
| Systematic offset in a region | Bias there | Add a distinguishing feature |
| A few huge isolated points | Outliers / unmodelled factors | Investigate individually |

### F10. What do YOUR residuals look like?
```
mean                      ₹2,905      (well-calibrated models sit near 0)
median                   -₹37.70      <- essentially zero
std                      ₹32,604
skew                      9.71
kurtosis                185.09
corr(predicted, |error|)  0.642       -> CLEAR FUNNEL
Shapiro-Wilk p          1.9e-49       -> decisively NOT normal
within +/-10 %           25.5 % of test listings
within +/-25 %           59.9 % of test listings
```

**The mean-vs-median gap is the whole story in two numbers.** Median −₹37.70 says the model is
beautifully calibrated for a *typical* listing. Mean +₹2,905 is pulled up by a few large
under-predictions of premium flats. **That gap is the skew.**

### F11. Your residuals are heteroscedastic (funnel corr 0.642). Isn't that a problem?
**It is a known, measured, and defensible property here, for four reasons:**

1. **MAPE is roughly flat across cities** (20.7 % – 32.7 %), so the model is similarly accurate in
   *relative* terms everywhere. The funnel is in absolute rupees only.
2. **A ₹50,000 error on a ₹5,00,000 penthouse (10 %) is a better prediction than a ₹5,000 error on
   a ₹10,000 flat (50 %).** Constant absolute error would actually be the wrong target.
3. **The `log1p` transform means the model is implicitly optimising relative error**, which is
   correct for a target spanning ₹1,200 to ₹35,00,000.
4. **The prediction interval widens automatically for expensive properties**, so the uncertainty is
   communicated to the user rather than hidden.

Heteroscedasticity invalidates OLS **standard errors and p-values** — but I am not doing inference
on coefficients, and Gradient Boosting makes no such assumption.

### F12. Shapiro–Wilk gives p = 1.9e-49. Doesn't that invalidate your model?
**No.** Normality of residuals is required for **inference** in OLS — p-values, confidence intervals
on coefficients, t-tests. It is **not** required for point predictions to be useful, and
**Gradient Boosting makes no distributional assumption about residuals at all.**

I report it as a diagnostic, and it is one of the reasons I preferred a tree ensemble to the linear
model.

**Also note:** with n = 950, Shapiro–Wilk will reject normality for almost any real data. The
**effect size** (kurtosis 185, skew 9.71) is far more informative than the p-value.

### F13. What is the difference between a loss function and an evaluation metric?
- **Loss function** — what the optimiser minimises *during* training. Must usually be
  differentiable. Here: squared error on `log1p(Rent)`.
- **Evaluation metric** — what *you* report to judge the model. Need not be differentiable. Here:
  R², MSE, RMSE, MAE, MAPE on the **rupee** scale.

**They are deliberately different in this project:** I train on log-squared-error but select on
rupee MAE, because the training loss must be optimisable while the selection metric must match the
business cost.

### F14. Why are all your metrics in rupees rather than log units?
Because **"RMSE = 0.34 log-rupees" tells a landlord nothing.** Reporting log-space errors would
also be flattering: logging compresses the scale, so the error numbers would look artificially
small.

Because the estimator is a `TransformedTargetRegressor`, `predict()` already applies `expm1`, so
both `y_true` and `y_pred` arrive on the rupee scale and ordinary scikit-learn scorers are directly
interpretable.

### F15. Why do you clip predictions at zero?
```python
y_pred = np.clip(y_pred, 0.0, None)
```
A model trained on **raw** rent (outlier-experiment setting 1) can emit **negative** rent
predictions, which are nonsensical. Clipping before scoring means the comparison between settings
is not distorted by impossible predictions. The app applies the same clip.

---

## SECTION G — Cross-Validation and Leakage

### G1. What is cross-validation and why use it?
Split the training data into k folds; train on k−1 and validate on the held-out one; rotate; average.

**Why:** a single train/validation split gives **one** estimate whose value depends on which rows
happened to land where. With 3,796 rows and a heavy-tailed target, that is a lottery. CV averages
over k splits, reducing the variance of the estimate and using **every row for validation exactly
once**.

### G2. Why k = 5 and not 10 or Leave-One-Out?
| k | Bias of the estimate | Variance | Cost |
|---|---|---|---|
| 5 | Slightly higher | Lower | **5 fits** ✅ |
| 10 | Lower | Higher | 10 fits |
| n (LOO) | Nearly unbiased | **Highest** | n fits — infeasible |

**k = 5 is the standard compromise.** With 24 model fits already (6 models × 4 settings) and a
RandomizedSearch inside each, k = 10 would double an 8.45-minute run for a marginal bias
improvement.

### G3. Why `shuffle=True` in KFold?
Without it, folds are **contiguous slices of the CSV**. If the file has any ordering — by city, by
posting date — one fold could be almost entirely a single city, making the CV estimate
meaningless. Shuffling (with a fixed seed) removes that risk.

### G4. Why KFold and not StratifiedKFold?
`StratifiedKFold` preserves **class proportions** and is designed for **classification** targets.
My target is continuous, so there are no classes to stratify.

**The common workaround for regression** is to bin the target and stratify on the bins. I did not
need it, because the **train/test split is already stratified by `City`** (the dominant source of
variation) and `KFold(shuffle=True)` on that already-balanced training set is sufficient.

### G5. What is data leakage? Give me four kinds and how you prevented each.
Leakage is when information unavailable at prediction time influences training, making test scores
optimistically biased.

| Kind | Where it could occur here | Prevention |
|---|---|---|
| **Target leakage** | `rent_per_sqft = Rent / size` | Never in `NUMERIC_FEATURES`; `remainder="drop"` makes it unreachable |
| **Train–test contamination** | Imputing with full-dataset medians; fitting `TargetEncoder` on all data | Everything inside the `Pipeline`, re-fit per fold |
| **Preprocessing leakage** | Fitting `StandardScaler` before splitting | Inside the pipeline |
| **Evaluation leakage** | Choosing hyperparameters by test score | Tuning uses **CV on the training split only** |

**Plus a fifth, specific to this project:** treating outliers in the **test** set. `treat_outliers`
is called on the training split only.

### G6. How exactly does the Pipeline prevent leakage? Walk me through one fold.
```
cross_validate(pipeline, X_train, y_train, cv=KFold(5))

For fold 1:
  1. clone(pipeline)                       -> a FRESH, UNFITTED copy
  2. fit on folds 2-5 (3,037 rows):
       - FeatureEngineer.fit  learns (City,BHK) size medians from THOSE 3,037 rows
       - SimpleImputer.fit    learns medians/modes from THOSE 3,037 rows
       - OneHotEncoder.fit    learns the category list from THOSE 3,037 rows
       - TargetEncoder.fit    learns locality encodings from THOSE 3,037 rows' targets
                              (and internally cross-fits even among them)
       - GradientBoosting.fit trains on THOSE 3,037 rows
  3. score on fold 1 (759 rows) -- which contributed NOTHING to any fitted quantity
  4. discard the clone, repeat for fold 2
```
**The `clone()` in step 1 is what guarantees no state carries over between folds.**

### G7. What would happen if you fitted the TargetEncoder outside the pipeline?
The encoding for every locality would be computed using the target values of rows that later
appear in the validation fold. **Every CV score would be optimistically biased** — the model would
look far better than it is — and the bias would be largest for exactly the rare localities where
honest estimates matter most.

The source carries a comment: *"Do not hoist this out of the pipeline."*

### G8. Your CV RMSE is ₹49,039 ± ₹39,836 but your test RMSE is ₹32,716. Explain the discrepancy. ★
**Three things are going on, and none is an error:**

1. **The CV standard deviation (₹39,836) is enormous — larger than the test RMSE itself.** That is
   the heavy tail: folds that happen to contain the ₹35-lakh listing score terribly; folds that do
   not score well. The test split happens to be an easier draw. ₹32,716 sits comfortably within
   ₹49,039 ± ₹39,836.

2. **CV trains on only 80 % of the training data per fold** (3,037 of 3,796 rows), so each fold's
   model is slightly weaker than the final model fitted on all 3,796.

3. **The CV score of a tuned model is mildly OPTIMISTIC in a different sense:** it belongs to the
   configuration that was *selected using those same folds*. The held-out test set — never touched
   during tuning — is the unbiased estimate. **Both are reported side by side for exactly that
   reason**, and my code documents it.

### G9. What is nested cross-validation? Did you use it? Should you have?
**Nested CV** has an outer loop for performance estimation and an inner loop for hyperparameter
tuning, giving an unbiased estimate of the *whole tuning procedure*.

**I did not use it.** I used a **single held-out test set** instead: tuning happens entirely within
5-fold CV on the training split, and the test set is touched only for the final report. That
achieves the same goal — an unbiased estimate — at 1/5 the cost.

**Nested CV would be better** if the dataset were too small to hold out 950 rows, or if I needed a
confidence interval on the generalisation estimate itself. With 24 model fits already, nested CV
would multiply the runtime fivefold. **A fair and honest trade-off to state.**

### G10. How do you know your model isn't overfitting?
**Three diagnostics, and one of them is deliberately labelled misleading:**

| Diagnostic | Value | Reading |
|---|---|---|
| `overfit_gap_rmse` = test − train RMSE | **−₹25,323** | **MISLEADING — see below** |
| `cv_overfit_gap_rmse` = CV val − CV train RMSE | −₹5,753 | The sound measure; close to zero |
| CV MAE std/mean | 0.147 | Stable across folds |

**Why the plain gap is misleading:** it is **negative**, which naively reads as "generalises better
than it fits". It does not. The single ₹35,00,000 listing sits in the **training** split, so train
RMSE is inflated by one row.

`cv_overfit_gap_rmse` compares 5-fold validation RMSE against 5-fold training RMSE — **both
computed on the same treated training data** — and is the statistically sound generalisation gap.
It is what the automatic model selection ranks on.

**And it is ranked on its ABSOLUTE value**, because a large *negative* gap is not "extra
generalisation" — it is an artifact of extreme listings landing in the training folds. **The ideal
is a gap near zero in either direction.**

---

## SECTION H — Results and Interpretation

### H1. Which attribute most influences rent?
**`City`** — permutation importance of **₹7,290** (the increase in test MAE when the column is
shuffled), followed by **`Size`** at ₹5,506.

| Rank | Feature | MAE increase (₹) | R² drop |
|---:|---|---:|---:|
| 1 | **City** | 7,290 | 0.340 |
| 2 | **Size** | 5,506 | 0.330 |
| 3 | Bathroom | 2,804 | 0.219 |
| 4 | Point of Contact | 2,775 | 0.196 |
| 5 | Area Locality | 1,998 | 0.091 |
| 6 | Furnishing Status | 334 | 0.002 |
| 7 | BHK | 279 | 0.004 |
| 8 | Floor | 244 | −0.010 |
| 9 | Area Type | 146 | 0.007 |
| 10 | Tenant Preferred | 23 | −0.001 |

### H2. Why does `Bathroom` (₹2,804) beat `BHK` (₹279) by 10×? ★
**Two reasons, and both are good answers:**

1. **Bathroom count is a proxy for luxury tier.** A 3 BHK with 1 bathroom and a 3 BHK with 3
   bathrooms are very different properties. Bathroom count discriminates between them; BHK does
   not.
2. **BHK is strongly correlated with `Size` and `Bathroom`, so permutation importance
   under-credits it.** Shuffling `BHK` leaves most of its information recoverable from the other
   two, so the metric barely moves. **This is the known weakness of permutation importance with
   correlated features** — not evidence that bedroom count is irrelevant to rent.

### H3. `Point of Contact` ranks 4th. Isn't that suspicious?
**Yes, and I would raise it myself.** It is **not a property attribute** — it records whether you
contact the owner, an agent or a builder. Agent-listed properties are systematically more
expensive (agents handle higher-value inventory, and their commission is priced in).

**Is it leakage?** **No** — it is known at listing time, so it is legitimately available at
prediction time. But it encodes **market positioning rather than the physical property**, so if the
goal were a *valuation* model rather than a *listing-price* model, I would drop it.

**Raising this yourself is one of the strongest moves available in the viva.**

### H4. Does locality dominate all the other attributes?
**No.** I ran an explicit ablation — same algorithm, same setting, three feature sets:

| Model | Features | Test RMSE | Test MAE | **Test R²** |
|---|---|---:|---:|---:|
| **A** | All | **32,716** | **10,135** | **0.714** |
| **B** | Without `Area Locality` and `City` | 44,728 | 13,156 | 0.466 |
| **C** | Only `Area Locality` and `City` | 47,537 | 16,514 | 0.397 |

```
Dropping location worsens RMSE by 36.7 %  (R² 0.714 -> 0.466)
Location alone recovers R² 0.397 = 56 % of the full model's 0.714
```

**Conclusion:** *location is the single strongest block of predictors but it does NOT dominate on
its own. Location-only reaches R² 0.397 versus 0.714 for the full model, so the physical attributes
(size, bathrooms, BHK, floor) still contribute materially. Equally, removing location costs 36.7 %
RMSE, so neither block is redundant.*

### H5. Why doesn't A = B + C? (0.714 ≠ 0.466 + 0.397)
**Because the blocks interact.** 1,000 sq ft means something very different in Mumbai (₹71/sq ft)
than in Kolkata (₹13/sq ft). That **interaction** is only available to model A, which has both
blocks. Variance explained is not additive across correlated, interacting feature groups.

**This is an excellent point to make unprompted.**

### H6. Impurity importance and permutation importance disagree. Which do you trust?
**Permutation importance**, for three reasons:
1. It is measured on the **original input columns**, so it answers *"which attribute matters?"*
   rather than *"which encoded column did the trees split on?"*
2. It is measured on **held-out test data**, so it measures generalisation, not memorisation.
3. **Impurity importance is biased towards high-cardinality and continuous features**, which is
   exactly the kind of feature I have (the target-encoded locality).

**The disagreement itself is informative:** impurity importance puts
`Point of Contact_Contact Agent` first (0.334) because it is a clean binary split near the root,
while `City` is **split across 6 one-hot columns** and so its total is diluted. Permutation
importance aggregates the original column and correctly puts `City` first.

### H7. Why is your model so bad at Kolkata (R² = 0.10)?
| | Kolkata |
|---|---|
| Total listings | **524 — fewest of any city** |
| Test listings | 105 |
| Mean rent | ₹13,676 |
| Std rent | ₹18,343 — **larger than the mean** |
| Unseen localities in test | 35.2 % |
| R² | **0.10** |
| RMSE as % of mean | **126.5 % — worst** |

**Three compounding reasons:**
1. **Thinnest evidence** — 419 training rows.
2. **35.2 % of its test listings sit in a locality never seen in training**, so the target encoding
   falls back to the global mean.
3. **Low mean with high variance.** R² is `1 − SS_res/SS_tot`, and with `SS_tot` small (low rent
   level) but a few expensive outliers inflating it unevenly, R² is hard to achieve. Note that
   Kolkata's **MAE is the lowest of any city at ₹4,911** — it is only *relatively* bad.

### H8. Why does Mumbai have the biggest absolute error?
Mean rent ₹85,782 with std ₹1,09,970 — **the widest spread of any city** — plus **45.6 % unseen
localities, the worst of any city**. Its RMSE of ₹63,564 is the largest in absolute rupees, but as
a share of its own mean (74.1 %) it is only middling, better than Delhi (103.5 %) and Kolkata
(126.5 %).

**This is why `rmse_pct_of_mean` is the right column to compare across cities** — raw RMSE just
tells you which city has expensive flats.

### H9. What single factor best explains the per-city differences?
**`unseen_locality_pct`** — the share of test listings whose locality never appeared in training:
```
Mumbai 45.6 % | Bangalore 36.2 % | Kolkata 35.2 % | Delhi 31.4 % | Chennai 25.3 % | Hyderabad 22.4 %
```
A listing in an unseen locality gets a target encoding shrunk all the way to the global mean, so
the model is effectively predicting from city + physical attributes alone.

### H10. Tell me about your worst predictions.
| City | Locality | Actual | Predicted | % error | Times seen in train |
|---|---|---:|---:|---:|---:|
| Mumbai | Juhu | 12,00,000 | 5,44,973 | −55 % | 3 |
| Delhi | Green Park | 3,50,000 | 54,298 | −84 % | 1 |
| Mumbai | Lake Superior, Powai | 2,30,000 | 4,97,219 | **+116 %** | **0** |
| Kolkata | Avenue S, Santoshpur | 1,80,000 | 11,128 | **−94 %** | **0** |
| Hyderabad | Beeramguda (1 BHK, 8000 sq ft) | 2,00,000 | 56,062 | −72 % | 0 |

**Four patterns:**
1. **9 of 10 are under-predictions** — the tree ensemble cannot extrapolate.
2. **8 of 10 sit in a locality seen 0–2 times in training** — the dominant failure mode.
3. **7 of 10 are in Mumbai** — widest spread, most unseen localities.
4. **The Hyderabad row (1 BHK, 8,000 sq ft) is almost certainly a data-entry error**, not a model
   failure.

My code assigns each a diagnosis automatically based on locality frequency and rent level.

### H11. How well does the model perform on new listings?
```
25.5 % of test listings predicted within +/-10 % of actual rent
59.9 % within +/-25 %
MAPE 26.74 %
80 % prediction interval covers 77.8 % of test listings
```
**Honest framing:** *it is a useful guide price, not a valuation. Three listings in five land
within a quarter of the true rent, and the app shows a range rather than a single number so the
user is never misled about the precision.*

### H12. Is R² = 0.71 good?
**It depends entirely on the domain, and here it is reasonable.**

- Physics experiments: 0.71 would be terrible.
- Real-estate pricing from **10 attributes with no amenities, building age, parking, metro
  proximity, society quality, view or photographs**: 0.71 is respectable.
- The ablation shows location alone gives 0.397, so the physical attributes genuinely add
  substantial signal.
- **The honest statement is about the residual 29 %:** it is unmodelled factors, not noise.
  Amenities, building age, floor plan quality, and the gap between asking price and negotiated
  rent all live in there — and I list them as named limitations.

---

## SECTION I — Deployment

### I1. How did you deploy the model?
A **Streamlit** web app at `app/app.py`, run with `streamlit run app/app.py`. Three tabs:
**Estimate rent**, **Model comparison & analysis**, **Limitations**.

### I2. What exactly does the app do when the user presses "Estimate rent"?
```
1. Build a ONE-ROW pandas DataFrame with the 11 correctly-named columns
2. pipeline.predict(row)                -> point estimate in rupees
3. quantile_low.predict(row)            -> 10th percentile
   quantile_high.predict(row)           -> 90th percentile
4. Enforce three safety rails:
     low, high = min(low,high), max(low,high)       # ordering
     low, high = min(low,point), max(high,point)    # point inside its own interval
     clip at 0                                      # rent cannot be negative
5. Display: point estimate, 80 % range, rent per sq ft, and the model's test metrics
```

**The app performs NO preprocessing of its own.** That is the whole point of keeping every
transformation inside the sklearn `Pipeline`.

### I3. Why does the app do no preprocessing?
To eliminate **training/serving skew** — the single most common production ML bug, where the app's
preprocessing silently drifts out of step with the training code. Here there is **one
implementation**, serialised into `pipeline.joblib`. It is physically impossible for them to
disagree.

### I4. What is the `sys.path` insertion at the top of app.py for?
```python
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
```
Placed **before** `joblib.load`. The saved pipeline references `src.preprocess.FeatureEngineer`;
unpickling must be able to **import that class**. Without this, Streamlit Cloud fails with
`ModuleNotFoundError` while everything works fine locally. **It is the single most common
deployment failure for scikit-learn apps.**

### I5. Why `@st.cache_resource` for the model but `@st.cache_data` for the JSON?
Streamlit **re-runs the whole script on every interaction**. Without caching, every slider move
would deserialise a multi-megabyte model.

| Decorator | Returns | Right for |
|---|---|---|
| `@st.cache_resource` | the object **by reference** | Large, unserialisable, **read-only** resources — the models |
| `@st.cache_data` | a **copy** | Data — so one user's mutation cannot leak to another |

Using `cache_data` for the model would copy it on every access; using `cache_resource` for mutable
data would create cross-session bugs. **The choice is deliberate.**

### I6. Why joblib and not pickle?
joblib special-cases **NumPy arrays** and writes them as raw binary buffers instead of pickling
their element structure. For an ensemble of 535 trees this is dramatically smaller and faster.
joblib is the scikit-learn-recommended serialisation method.

**Both share the same caveats:** the class definitions must be importable at load time, and you
must never unpickle an untrusted file (it can execute arbitrary code).

### I7. What happens if the user picks a locality the model has never seen?
The dropdown has an explicit **"Other / not listed"** option, mapped to `"__unseen_locality__"`.
The `TargetEncoder` has never seen that string, so it returns the **global mean encoding** — the
estimate then relies on city, size and the other attributes alone.

**The app shows an explicit warning** telling the user exactly that, and advising them to treat it
as a wider guess than the displayed range suggests. **Communicating the fallback rather than hiding
it is the right design.**

### I8. How fast is a prediction?
**6.53 ms** for a single row, measured over 50 repeated calls in `train.py`. (Random Forest would
be 13.25 ms.) Comfortably fast enough for an interactive web app.

### I9. What does the app show besides the number?
```
Estimated monthly rent    ₹XX,XXX
Expected range            ₹LL,LLL - ₹HH,HHH      (80 % prediction interval)
Rent per sq ft            XX.X
```
Plus an expandable panel showing the model name, test RMSE / MAE / R², the training setting, a note
that the scores come from a 950-listing held-out test set, and a read-only echo of the exact input
row.

The case study asks for *"the selected model and prediction details where appropriate"* — this is
that.

### I10. How would you put this in production properly?
| Concern | Approach |
|---|---|
| **Serving** | Wrap the pipeline in a FastAPI endpoint; Streamlit is a demo UI, not a production service |
| **Containerisation** | Docker, with the exact pinned `requirements.txt` |
| **Versioning** | MLflow or DVC for model + data lineage |
| **Monitoring** | Log every prediction with inputs; track prediction-vs-actual drift and input-distribution drift |
| **Retraining** | **Monthly at minimum, quarterly at absolute minimum** — rental markets move faster than a 3-month snapshot |
| **A/B testing** | Shadow-deploy a new model before switching traffic |
| **Human override** | Always. The estimate is a suggestion, never an authority |
| **Fairness audit** | Check for systematic under-prediction in specific localities |

### I11. What could go wrong if a real platform deployed this?
> *A platform that showed these numbers as authoritative would **anchor the market to its own
> predictions**, creating a feedback loop where the model's errors become next year's training
> data. Under-prediction in underserved localities could also **entrench existing pricing
> inequities**. Any deployment needs a human override and monitoring of prediction-versus-actual
> drift.*

This is in the app's Limitations tab. **Raising the ethical/feedback-loop risk unprompted is a
strong move.**

### I12. How does the prediction interval work?
Two **additional** Gradient Boosting models with the **quantile loss**:
```
quantile_low.joblib   alpha = 0.10
quantile_high.joblib  alpha = 0.90
0.90 - 0.10 = 0.80  ->  an 80 % prediction interval
```

**Pinball loss:**
```
              { α(y - ŷ)        if y >= ŷ
L_α(y, ŷ) =   {
              { (α-1)(y - ŷ)    if y <  ŷ
```
For α = 0.9, under-prediction is penalised 9× more than over-prediction, pushing the model up until
only 10 % of true values lie above it — the 90th percentile.

**Validated:** nominal 80 %, **empirical coverage 77.8 %** on the held-out test set, median width
74.3 % of actual rent.

### I13. Why not just `prediction ± MAE`?
A fixed ±₹10,135 band would be **far too wide** for an ₹8,000 Kolkata studio (it would go negative)
and **far too narrow** for a ₹5,00,000 Mumbai penthouse. Quantile models learn a **conditional,
property-specific** width.

The `± MAE` approach is kept **only as a fallback** if the quantile models fail to load.

### I14. You trained the quantile models on log rent. Doesn't the transform break the quantiles? ★
**No — and this is the elegant part.** **Quantiles are invariant under monotone transformations:**
```
If g is strictly increasing:   Q_α( g(X) )  =  g( Q_α(X) )
```
Therefore `expm1( Q_α( log1p(Rent) ) ) = Q_α(Rent)` **exactly**. No approximation.

**Contrast with the mean, which is NOT transform-invariant:** `exp(mean(log x)) ≠ mean(x)` — it
gives the geometric mean, which is the source of the known downward bias when back-transforming
log-space mean predictions.

**This is a genuinely sophisticated point. Volunteer it.**

### I15. Is 77.8 % coverage against a nominal 80 % acceptable?
**Yes.** A 2.2-point shortfall on 950 test rows is well within sampling noise, and the shortfall is
in the direction you would expect given the heavy tail: the few extreme listings fall outside.

**An interval that OVER-covered would be a worse sign** — 95 % coverage on a nominal 80 % interval
would mean the intervals are uselessly wide.

---

## SECTION J — The Complete Formula Bank

> Examiners love formulas. Know every one of these, know what each symbol means, and know **one
> sentence about when you would use it**.

### J1. Measures of Central Tendency

**Mean (arithmetic average)**
```
         1    n              Σ x_i
   x̄  = ---  Σ  x_i    =    -------
         n   i=1               n
```
*Sensitive to outliers.* My rent mean is **₹34,993** — dragged far above the median by a few
₹10-lakh+ listings.

**Median (50th percentile)**
```
            { x_((n+1)/2)                        if n is ODD
   median = {
            { ( x_(n/2) + x_(n/2 + 1) ) / 2      if n is EVEN
```
(on the **sorted** data). *Robust to outliers.* My rent median is **₹16,000**.

**Mode** — the most frequently occurring value.
A distribution can be unimodal, bimodal or multimodal. My modal `BHK` is **2** (2,265 listings);
modal `City` is **Mumbai** (972); modal `Furnishing Status` is **Semi-Furnished** (2,251).

**The relationship that reveals skew:**
```
Right (positive) skew :   mode  <  median  <  mean      <- MY DATA (16,000 < 34,993)
Symmetric             :   mode  =  median  =  mean
Left (negative) skew  :   mean  <  median  <  mode
```

**Empirical (Pearson) mode approximation** for moderately skewed data:
```
mean - mode  ≈  3 (mean - median)
```

**Geometric mean** — `GM = (x₁·x₂·…·x_n)^(1/n) = exp( mean( log x ) )`.
*Relevant here:* back-transforming a log-space mean gives the **geometric mean**, which is ≤ the
arithmetic mean by Jensen's inequality. That is the source of the log-transform bias.

**Harmonic mean** — `HM = n / Σ(1/x_i)`. Used for rates; also the basis of the F1 score.

**Weighted mean** — `x̄_w = Σ(w_i·x_i) / Σ w_i`.

### J2. Measures of Dispersion

**Range** = `max − min`. My rent range: `35,00,000 − 1,200 = 34,98,800`.

**Variance**
```
                 1    n
   Population: σ² = --- Σ (x_i - μ)²
                    n  i=1

                 1      n
   Sample:   s² = ----- Σ (x_i - x̄)²       <- note the n-1  (Bessel's correction)
                  n - 1 i=1
```
**Why n−1?** Using `x̄` (estimated from the same data) instead of the true μ makes the sum of
squares systematically too small. Dividing by n−1 instead of n corrects the bias. We "used up" one
degree of freedom estimating the mean.

**Standard deviation** = `√variance`. Same units as the data. My rent std: **₹78,106**.

**Coefficient of Variation** `CV = σ / μ` — a *unitless* relative dispersion measure.
*Used in this project* to rank CV-stability: `cv_mae_cv = CV MAE std / CV MAE mean`, so a model is
not rewarded merely for having smaller errors overall.

**Mean Absolute Deviation** `MAD = (1/n) Σ |x_i − x̄|`.

**Median Absolute Deviation** `MedAD = median( |x_i − median(x)| )` — highly robust; the basis of
the modified Z-score.

**Interquartile Range** `IQR = Q3 − Q1` — the spread of the middle 50 %. **Robust.**

### J3. Position and Shape

**Percentile / Quantile** — `P_k` is the value below which k % of observations fall.
```
Q1 = P25      Q2 = P50 = median      Q3 = P75
```

**Five-number summary** — `min, Q1, median, Q3, max` — exactly what a **boxplot** draws.

**Z-score (standard score)**
```
        x - μ
   z = -------
          σ
```
How many standard deviations a value is from the mean. `|z| > 3` is the classic outlier rule —
**but it is unsuitable here** because μ and σ are themselves destroyed by skew 21.41.

**Modified Z-score (robust)**
```
         0.6745 ( x - median )
   M  = -----------------------          flag if |M| > 3.5
                MedAD
```
(0.6745 is the 75th percentile of the standard normal, making MedAD comparable to σ.)

**Skewness**
```
              1    n  ( x_i - x̄ )³
   Skew  =   ---  Σ   ------------
              n   i=1      s³
```
0 = symmetric; > 0 = right tail; < 0 = left tail.
**Mine: 21.41 raw → 0.909 after `log1p`.**

**Kurtosis**
```
              1    n  ( x_i - x̄ )⁴
   Kurt  =   ---  Σ   ------------            excess kurtosis = Kurt - 3
              n   i=1      s⁴
```
Normal → excess 0. **My residuals: 185.09** — extraordinarily heavy tails.

### J4. Association

**Covariance**
```
                  1      n
   Cov(X, Y) = ------- Σ (x_i - x̄)(y_i - ȳ)
                n - 1  i=1
```
Sign tells direction; magnitude is unit-dependent and therefore hard to interpret.

**Pearson correlation coefficient**
```
             Cov(X, Y)              Σ (x_i - x̄)(y_i - ȳ)
    r  =   -------------   =   --------------------------------
              σ_X σ_Y          √[Σ(x_i-x̄)²] · √[Σ(y_i-ȳ)²]
```
Range **[−1, +1]**. Measures **linear** association only. `r = 0` does **not** mean independence —
a perfect parabola has r = 0.

**Coefficient of determination** for simple linear regression: `R² = r²`.
*(This identity holds only for simple OLS with an intercept; in general R² is defined by the
SS ratio.)*

**Spearman rank correlation** — Pearson's r applied to the **ranks**. Captures any **monotone**
relationship and is robust to outliers. *Arguably more appropriate than Pearson on a target with
skew 21.41.*

**Variance Inflation Factor** (multicollinearity diagnostic)
```
              1
   VIF_j = --------       where R²_j is from regressing feature j on all the others
           1 - R²_j
```
VIF > 5 (some say 10) signals problematic multicollinearity. *Relevant to the ~324 polynomial
terms.*

### J5. Linear Models

**Simple linear regression**
```
    ŷ = β₀ + β₁x

              Σ (x_i - x̄)(y_i - ȳ)        Cov(X,Y)
       β₁  = ------------------------  =  ----------
                   Σ (x_i - x̄)²             Var(X)

       β₀  =  ȳ - β₁ x̄
```

**Multiple linear regression** — `ŷ = β₀ + β₁x₁ + … + β_p x_p`, or in matrix form `ŷ = Xβ`.

**Cost function (RSS / OLS objective)**
```
              n
   RSS(β) =  Σ  (y_i - ŷ_i)²   =   ||y - Xβ||²
             i=1
```

**Normal equation (closed-form OLS solution)**
```
   β̂ = (XᵀX)⁻¹ Xᵀ y
```
Requires `XᵀX` to be invertible — i.e. **no perfect multicollinearity**. This is exactly why
`drop="first"` is needed on one-hot columns for linear models.

**Gradient descent (the iterative alternative)**
```
   β := β - η · ∇J(β)          where for MSE:  ∇J(β) = -(2/n) Xᵀ (y - Xβ)
```
η is the learning rate. Variants: Batch (all rows), Stochastic (one row), Mini-batch.

**Ridge regression (L2)**
```
   J(β) = ||y - Xβ||²  +  α ||β||₂²  =  Σ(y_i - ŷ_i)² + α Σ βⱼ²

   β̂_ridge = (XᵀX + αI)⁻¹ Xᵀ y
```
Adding `αI` lifts every eigenvalue of `XᵀX` away from zero → well-conditioned inverse. **That is
the mathematical reason regularisation fixes multicollinearity.**

**Lasso regression (L1)**
```
   J(β) = ||y - Xβ||²  +  α Σ |βⱼ|
```
No closed form (the L1 term is not differentiable at 0); solved by coordinate descent. **Produces
exactly-zero coefficients → feature selection.**

**Elastic Net**
```
   J(β) = ||y - Xβ||²  +  α [ ρ Σ|βⱼ|  +  (1-ρ)/2 · Σ βⱼ² ]
```

**Polynomial regression (degree 2, two features)**
```
   ŷ = β₀ + β₁a + β₂b + β₃a² + β₄ab + β₅b²

   number of terms (degree 2, no bias) = p + p(p+1)/2
```
**Linear in β, non-linear in x.** That is why it is still called a *linear model*.

### J6. Trees and Ensembles

**Regression tree split criterion — variance / MSE reduction**
```
                        n_L              n_R
   ΔMSE = MSE(parent) - --- MSE(left) - --- MSE(right)
                         n                n

                   1
   where MSE(S) = ---  Σ (y_i - ȳ_S)²
                  |S| i∈S
```

**Classification criteria (know these for contrast — NOT used here)**
```
   Gini impurity  :  G = 1 - Σ p_k²                   (max 0.5 for 2 balanced classes)
   Entropy        :  H = - Σ p_k log₂ p_k             (max 1 bit for 2 balanced classes)
   Information Gain = H(parent) - Σ (n_child/n) H(child)
```

**Tree prediction (regression)**
```
   ŷ(x) = mean{ y_i : x_i lands in the same leaf as x }
```

**Random Forest prediction**
```
             1   B
   ŷ(x) =  ---  Σ  T_b(x)
             B  b=1
```

**Bootstrap coverage** — probability a given row appears in a bootstrap sample of size n:
```
   1 - (1 - 1/n)^n   ->   1 - 1/e   ≈  0.632      (63.2 %)
```
The remaining **~36.8 %** are **out-of-bag** for that tree.

**Variance of an average of correlated predictors** ★
```
                            1 - ρ
   Var(average)  =  ρ σ²  + ------- σ²
                              B
```
As B → ∞ only `ρσ²` remains. **Lowering ρ (random feature subsetting) matters as much as raising
B.** This single formula explains the entire design of Random Forest.

**Gradient Boosting**
```
   F₀(x)   = argmin_γ Σ L(y_i, γ)                 (= ȳ for squared loss)

   r_im    = -[ ∂L(y_i, F(x_i)) / ∂F(x_i) ]_{F=F_{m-1}}      (pseudo-residual)
           =  y_i - F_{m-1}(x_i)                              for squared loss

   F_m(x)  = F_{m-1}(x) + ν · h_m(x)              ν = learning rate

   F_M(x)  = F₀(x) + ν Σ_{m=1}^{M} h_m(x)
```

**Pinball / quantile loss**
```
                 { α (y - ŷ)          if y >= ŷ
   L_α(y, ŷ) =   {
                 { (α - 1)(y - ŷ)     if y <  ŷ
```
At α = 0.5 this is symmetric and recovers MAE / the median.

**AdaBoost weight update (for contrast)**
```
   α_m = ½ ln( (1 - err_m) / err_m )
   w_i := w_i · exp( α_m · 1[y_i ≠ h_m(x_i)] )     then normalise
```

### J7. Regression Metrics

```
                 Σ (y_i - ŷ_i)²            SS_res
   R²    =  1 - ----------------  =  1 - ---------
                 Σ (y_i -  ȳ )²            SS_tot

                      (1 - R²)(n - 1)
   Adj R² =  1  -  ----------------------
                      n - p - 1

              1
   MSE   =   ---  Σ (y_i - ŷ_i)²
              n

   RMSE  =  √MSE

              1
   MAE   =   ---  Σ | y_i - ŷ_i |
              n

             100        | y_i - ŷ_i |
   MAPE  =  -----   Σ   -------------
              n             | y_i |

             100        | y_i - ŷ_i |
   SMAPE =  -----   Σ   ---------------------
              n          (|y_i| + |ŷ_i|) / 2

   Residual :  e_i = y_i - ŷ_i
```

**Always true:** `RMSE ≥ MAE`. The ratio measures error inequality (**mine: 3.23**).

### J8. Classification Metrics (know them even though this is regression)

```
                      Predicted
                    Pos      Neg
   Actual  Pos      TP       FN
           Neg      FP       TN

   Accuracy  = (TP + TN) / (TP + TN + FP + FN)
   Precision = TP / (TP + FP)                 "of those I flagged, how many were right?"
   Recall    = TP / (TP + FN)                 "of those that existed, how many did I catch?"
   Specificity = TN / (TN + FP)
   F1        = 2 · (Precision · Recall) / (Precision + Recall)      <- harmonic mean
```
**If asked "why not accuracy?" in a regression context:** accuracy is undefined for a continuous
target. There is no notion of a "correct" prediction when the target is a real number — which is
precisely why regression uses distance-based metrics.

### J9. Preprocessing

```
   Standardisation   :  z = (x - μ) / σ
   Min-Max scaling   :  x' = (x - min) / (max - min)
   Robust scaling    :  x' = (x - median) / IQR
   Log transform     :  log1p(x) = log(1 + x)      inverse: expm1(x) = exp(x) - 1

                         n_c · ȳ_c  +  m · ȳ
   Target encoding   :  ----------------------
                              n_c  +  m

   IQR outlier bounds:  lower = Q1 - 1.5·IQR      upper = Q3 + 1.5·IQR
   Winsorising       :  x' = clip(x, P1, P99)
```

### J10. Theory

**Bias–Variance decomposition**
```
   E[(y - f̂(x))²]  =  Bias[f̂(x)]²  +  Var[f̂(x)]  +  σ²_noise

   where  Bias[f̂(x)] = E[f̂(x)] - f(x)
          Var[f̂(x)]  = E[ (f̂(x) - E[f̂(x)])² ]
          σ²_noise   = irreducible error
```

**Jensen's inequality** (why back-transforming a log-mean under-predicts)
```
   For a CONCAVE g:   E[ g(X) ]  ≤  g( E[X] )
   With g = log:      E[log X]   ≤  log E[X]
   Hence              exp(E[log X])  ≤  E[X]          i.e. geometric mean ≤ arithmetic mean
```

**Quantile invariance under monotone transforms** (why the prediction interval is exact)
```
   For strictly increasing g:   Q_α( g(X) )  =  g( Q_α(X) )
```

**Expected value and linearity**
```
   E[aX + bY] = a·E[X] + b·E[Y]
   Var(aX)    = a² Var(X)
   Var(X + Y) = Var(X) + Var(Y) + 2Cov(X, Y)
```

**Law of Large Numbers** — the sample mean converges to the true mean as n → ∞.
**Central Limit Theorem** — the sampling distribution of the mean approaches normal as n → ∞,
regardless of the population distribution. *Relevant here:* it is why my CV **mean** is more
trustworthy than any single fold score.

---

## SECTION K — "Where Is the Code For…?" Lookup Table

| The examiner asks about… | File | Function / location |
|---|---|---|
| Random seed, all paths, column names, thresholds | [src/config.py](src/config.py) | module-level constants |
| Loading the CSV + header validation | [src/data.py](src/data.py) | `load_data()` |
| The section-4 data profile | [src/data.py](src/data.py) | `profile_data()`, `print_profile()` |
| Collapsing floor strings to patterns | [src/data.py](src/data.py) | `floor_pattern()` |
| **Parsing the `Floor` column** | [src/preprocess.py](src/preprocess.py) | `parse_floor()` |
| **Missing-value imputation (group medians)** | [src/preprocess.py](src/preprocess.py) | `FeatureEngineer.fit()` / `.transform()` |
| `floor_ratio`, `carpet_equiv_size`, interactions | [src/preprocess.py](src/preprocess.py) | `FeatureEngineer.transform()` |
| The `Size < 50 sq ft` rule | [src/preprocess.py](src/preprocess.py) | `FeatureEngineer._blank_implausible_size()` |
| `rent_per_sqft` (EDA only) + leakage warning | [src/preprocess.py](src/preprocess.py) | `add_eda_features()` |
| Testing the area-conversion ratios | [src/preprocess.py](src/preprocess.py) | `empirical_area_ratios()` |
| **Outlier detection (per-city IQR)** | [src/preprocess.py](src/preprocess.py) | `detect_outliers()` |
| **Outlier treatment (none/remove/cap)** | [src/preprocess.py](src/preprocess.py) | `treat_outliers()` |
| **One-hot, scaling, target encoding** | [src/preprocess.py](src/preprocess.py) | `build_preprocessor()` |
| The locality-ablation feature sets | [src/preprocess.py](src/preprocess.py) | `build_preprocessor(feature_set=...)` |
| **The six model definitions + search spaces** | [src/models.py](src/models.py) | `model_specs()` |
| **Pipeline assembly + `log1p` wrapper** | [src/models.py](src/models.py) | `build_pipeline()` |
| Polynomial expansion | [src/models.py](src/models.py) | `_poly()` |
| Quantile models for the interval | [src/models.py](src/models.py) | `build_quantile_pipeline()` |
| Reading feature names out of a fitted pipeline | [src/models.py](src/models.py) | `get_feature_names()` |
| **R², MSE, RMSE, MAE, MAPE** | [src/evaluate.py](src/evaluate.py) | `compute_metrics()` |
| The shared 5-fold splitter | [src/evaluate.py](src/evaluate.py) | `cv_splitter()` |
| Cross-validation reporting | [src/evaluate.py](src/evaluate.py) | `cv_report()`, `cv_from_search()` |
| **Per-city performance** | [src/evaluate.py](src/evaluate.py) | `per_city_metrics()` |
| **The 10 worst predictions** | [src/evaluate.py](src/evaluate.py) | `worst_predictions()` |
| **Error concentration (42 % from one row)** | [src/evaluate.py](src/evaluate.py) | `error_concentration()` |
| Rent bands for the premium analysis | [src/evaluate.py](src/evaluate.py) | `rent_band()` |
| Indian rupee formatting | [src/evaluate.py](src/evaluate.py) | `format_inr()` |
| **Every figure** | [src/plots.py](src/plots.py) | `plot_*()` functions |
| **The train/test split** | [train.py](train.py) | `train_test_split(..., stratify=City)` |
| **Hyperparameter tuning (`refit="mae"`)** | [train.py](train.py) | `tune_and_evaluate()` |
| The four experimental settings | [train.py](train.py) | `SETTINGS` |
| **Multi-criteria model selection** | [train.py](train.py) | "SECTION 12 — MODEL SELECTION" |
| **Permutation importance** | [train.py](train.py) | "SECTION 11 — FEATURE IMPORTANCE" |
| **The locality ablation** | [train.py](train.py) | "SECTION 11 — LOCALITY ABLATION" |
| Residual diagnostics + Shapiro test | [train.py](train.py) | "SECTION 10" |
| Saving artifacts | [train.py](train.py) | "SECTION 15" |
| **The web app** | [app/app.py](app/app.py) | `render_predict_tab()` |
| The analysis tables in the app | [app/app.py](app/app.py) | `render_analysis_tab()` |
| **The limitations text** | [app/app.py](app/app.py) | `render_limitations_tab()` |
| The `sys.path` deployment fix | [app/app.py](app/app.py) | top of file, before `joblib.load` |
| README generation | [make_readme.py](make_readme.py) | — |
| Notebook generation | [notebooks/build_notebook.py](notebooks/build_notebook.py) | — |
| **All the numbers** | `models/metrics.json` | — |
| Comparison tables | `reports/*.csv` | — |

---

## SECTION L — Graph-Based Questions

### L1. Show me your rent distribution plot and explain it.
**`reports/figures/eda_rent_distribution.png`** — two panels, raw and `log1p`.

> *The left panel shows an extreme right skew: a tall spike near ₹10,000–20,000 and a very long
> thin tail stretching to ₹35,00,000. Skewness is 21.41. The right panel shows the same data after
> `log1p`, which is nearly symmetric — skewness 0.909. **This one figure is the justification for
> the entire target transform**, and is why mean (₹34,993) is more than twice the median
> (₹16,000).*

### L2. What does your residual plot show and is it good?
**`best_model_residual_diagnostics.png`**

> *Residuals are centred on zero for the bulk of listings — the median residual is −₹37.70, so the
> model is well calibrated for a typical property. But there is a **clear funnel**: the vertical
> spread widens as predicted rent increases, and `corr(predicted, |error|) = 0.642` quantifies
> that. A handful of very large positive residuals (under-predictions of premium flats) produce the
> skew of 9.71 and kurtosis of 185.*
>
> *Strictly, this is heteroscedasticity, which would invalidate OLS inference. It does not
> invalidate my model: Gradient Boosting makes no distributional assumption, MAPE is roughly flat
> across cities (20.7–32.7 %), and the prediction interval widens automatically for expensive
> properties, so the uncertainty is communicated rather than hidden.*

### L3. How do you read a Q-Q plot?
It plots the quantiles of your residuals against the quantiles of a theoretical normal
distribution.
- **Points on the 45° line** → residuals are normal
- **Both ends curving away from the line** → heavy tails
- **S-shape** → skew
- **One end only** → one-sided skew

**In mine the tails curve away sharply**, consistent with kurtosis 185 and Shapiro p = 1.9e-49.

### L4. How do you read a boxplot?
```
    |----[ Q1 |median| Q3 ]----|      o   o
   lower      BOX              upper  outliers
   whisker                    whisker
```
- Box spans **Q1 to Q3** (the IQR, the middle 50 %)
- Line inside the box is the **median**
- Whiskers extend to the furthest point within **1.5 × IQR** of the box
- Points beyond are plotted individually as **outliers**

**In `eda_rent_by_city.png`:** Mumbai's box sits almost entirely above Kolkata's whisker. That
visual non-overlap is the argument for per-city outlier bounds in one picture.

### L5. In your learning/complexity curve, train error is low and test error is high. What is that?
**Overfitting — high variance.** My live example:
```
Polynomial (deg 2):  train RMSE ₹62,695   test RMSE ₹9,17,495
```
**Fixes:** more data, fewer features, more regularisation (Ridge took R² from −223.59 to −4.49),
ensembling, early stopping.

### L6. What if BOTH train and test error are high?
**Underfitting — high bias.** The model is too simple.
**Fixes:** more/better features, a more complex model, less regularisation, more training
iterations.
**My example:** Linear Regression on **raw** rent reaches R² 0.582 vs 0.792 on the log target — the
linear-in-rupees form is too rigid.

### L7. What if the learning curve shows train and test converging to a high error?
Both curves flat and high, with a small gap → **high bias**. Adding more data will not help;
you need a better model or better features.

**If they converge to a low error with a small gap** → you are in a good place and more data has
diminishing returns.

**If there is still a large gap at the right edge** → more data *will* help.

### L8. Your feature importance plot shows two panels that disagree. Explain.
> *The left panel is **impurity-based importance** on the **encoded** columns, computed on training
> data during fitting. The right panel is **permutation importance** on the **original** input
> columns, computed on the **held-out test set**.*
>
> *They disagree because impurity importance puts `Point of Contact_Contact Agent` first (0.334) —
> a clean binary split near the root — while `City` is split across six one-hot columns and its
> total is diluted. Permutation importance aggregates the original `City` column and correctly puts
> it first at ₹7,290.*
>
> ***I trust permutation importance**, because it is measured on original columns and on held-out
> data, and because impurity importance is known to be biased towards high-cardinality features
> like my target-encoded locality.*

### L9. Explain your model comparison bar chart.
**`model_rmse_comparison.png`** shows CV RMSE with error bars alongside test RMSE for all six
models.

> *Two things stand out. First, **the error bars are enormous** — CV RMSE ± std is ₹49,039 ±
> ₹39,836 for my selected model — which is the visual evidence for why I tune on MAE rather than
> RMSE. Second, **the two polynomial variants are off the scale** (test RMSE ₹9,17,495 and
> ₹1,43,423), which is the overfitting demonstration.*

### L10. Explain the premium trade-off chart.
**`outlier_premium_tradeoff.png`** plots MAE by rent band for each of the four settings.

> *The three cheap bands are nearly flat across all four settings — treatment barely changes them.
> The premium band (> ₹1,00,000) is where the settings separate dramatically: ₹73,887 with no
> treatment versus **₹1,05,137 with outliers removed**. **That single bar is my entire argument for
> keeping the outliers.***

### L11. Explain the prediction interval coverage figure.
**`prediction_interval_coverage.png`** sorts test listings by actual rent and plots each one's
80 % interval.

> *The band widens as rent rises, which is correct — the model is genuinely less certain about
> expensive properties. **77.8 % of actual values fall inside, against a nominal 80 %**, which is
> well within sampling noise on 950 rows. The points that fall outside are concentrated in the
> premium tail, exactly as expected.*

### L12. What does the correlation heatmap tell you?
> *`Size`, `BHK` and `Bathroom` are strongly mutually correlated — bigger flats have more bedrooms
> and more bathrooms. **This matters for interpretation, not for the model:** tree ensembles handle
> correlated features fine, but permutation importance under-credits them, which is why `BHK`
> scores only ₹279. It is not evidence that bedroom count is irrelevant.*

### L13. If I showed you a residual plot with a clear U-shape, what would you conclude?
**Missing non-linearity.** The model is systematically under-predicting at both extremes and
over-predicting in the middle (or vice versa), which means the true relationship has curvature the
model cannot express.

**Course of action:** add polynomial terms or interactions, use a non-linear model (tree
ensemble), or transform the offending feature.

### L14. If residuals were perfectly random but large, what would that mean?
The model has extracted **all the systematic signal available in these features** — the remaining
error is either **irreducible noise** or driven by **features I do not have**.

**In my case, I can name those missing features:** amenities, building age, parking, metro
proximity, society quality, security, view, and the gap between asking price and negotiated rent.
That is the honest explanation for the residual 29 % of variance.

---

## SECTION M — Scenario Questions ("If this happened, what would you do?")

### M1. What if you got 10× more data (50,000 listings)?
1. **Re-run the whole pipeline** — every threshold was derived from data and should be re-derived.
2. **Revisit the locality encoding.** With 50,000 rows, many localities would cross the threshold
   where their own mean is reliable; I would reduce the smoothing strength `m`, or consider
   one-hot for the top-N localities with target encoding for the tail.
3. **Try gradient-boosting libraries built for scale** — LightGBM/XGBoost with histogram splits.
4. **Reconsider a neural network with learned locality embeddings** — now plausible.
5. **Expect premium-band accuracy to improve most**, because the tail would finally be populated.
6. **Use a larger validation set** and possibly switch from 5-fold CV to a fixed validation split
   to save compute.

### M2. What if a new city (Pune) was added?
**The model would not generalise to it at all**, and I would say so rather than let it guess.
- `City` is one-hot encoded with `handle_unknown="ignore"`, so "Pune" produces an **all-zero City
  block** — the model falls back to the global baseline plus physical attributes.
- Every Pune locality would also be unseen → target encoding falls back to the global mean.
- **Correct action: retrain with Pune data.** In the interim, the app should refuse to serve
  unsupported cities rather than return a confident-looking wrong number.

### M3. What if rents doubled due to inflation?
**The model would systematically under-predict by ~50 %, and nothing in it would detect that.**

- It is trained on an **89-day snapshot from mid-2022** and has no temporal feature (I dropped
  `Posted On` deliberately).
- **Short-term patch:** apply an external inflation index as a multiplier.
- **Correct fix:** retrain on current data. My stated retraining cadence is **monthly at minimum,
  quarterly at absolute minimum**.
- **In production** this would be caught by **drift monitoring** — comparing the distribution of
  predictions against realised rents over time.

### M4. What if the model gave a negative rent prediction?
**It is clipped at zero**, in both `compute_metrics` and the app:
```python
y_pred = np.clip(y_pred, 0.0, None)
```
**When can it happen?** Only in the **raw-target** setting (experiment setting 1), where a linear
model can extrapolate below zero. **It cannot happen in the deployed model**, because `expm1` of
any real number is > −1 and the point estimate is additionally floored at 0.

### M5. What if a user enters Size = 50,000 sq ft?
The app's `number_input` caps `Size` at **8,000**, which is the dataset maximum — so the input is
rejected at the UI layer.

**If it somehow got through:** the tree ensemble **cannot extrapolate**. It would fall into the
"largest size" leaf and return that leaf's mean, massively under-predicting. **That is the correct
conservative failure mode for a tree** — but the UI cap is the real defence, and that is why it is
there.

### M6. What if your model performs well in CV but badly on the test set?
**That is the signature of overfitting to the validation folds** — usually from tuning too
aggressively on CV, or from leakage inside CV.

**My checks:**
1. Is the `TargetEncoder` inside the pipeline? (Yes — if it were not, CV would be optimistic.)
2. Is `cv_overfit_gap_rmse` large and positive? (Mine is −₹5,753, i.e. near zero.)
3. Did I use the test set during tuning? (No — the test set is touched once, for reporting.)
4. Is the test set unrepresentative? (Stratified by city, so unlikely.)

**In my project the opposite happened** — CV RMSE ₹49,039 vs test RMSE ₹32,716 — which I explain by
the heavy tail and the enormous CV std of ₹39,836.

### M7. Your examiner says: "Your R² is only 0.71, that's not good enough. Fix it."
**My response, in order:**
1. **Name what the residual 29 % actually is:** unmodelled features — amenities, building age,
   parking, metro proximity, society quality, view — not random noise. The ablation shows location
   alone gives 0.397, so the physical attributes genuinely add real signal.
2. **Point out that I could trivially inflate R²** by using `rent_per_sqft` as a feature, which
   would give ~1.0 — and that doing so would be **leakage**, because it is computed from the
   target.
3. **Point out that R² is the wrong lever here anyway:** one test listing is 42 % of total squared
   error, so R² is largely measuring that one row. **MAE of ₹10,135 and MAPE of 26.7 % are the
   honest accuracy statement.**
4. **State what would actually improve it:** amenity and building-age features, geocoordinates
   instead of free-text locality (which would fix the unseen-locality problem entirely), actual
   transacted rent instead of asking price, and more data per locality.

### M8. What if 30 % of `Size` values were missing instead of 0 %?
My imputation cascade already handles it:
```
(City, BHK) median  ->  City median  ->  global median
```
**But I would add two things:**
1. **A missingness indicator feature** (`size_was_missing`), because missingness is often
   informative — listings that omit the size may differ systematically.
2. **Consider iterative/model-based imputation** (`IterativeImputer`, MICE), which predicts `Size`
   from the other columns rather than using a flat group median.

I would also **test** whether the imputation is helping, by comparing against dropping those rows.

### M9. What if you had to make predictions in real time at 10,000 requests/second?
| Concern | Action |
|---|---|
| **Latency** | 6.53 ms per row is already fine, but I would **batch** requests — vectorised prediction over 1,000 rows is far cheaper than 1,000 single calls |
| **Throughput** | Horizontal scaling behind a load balancer; the model is stateless |
| **Model size** | Consider reducing `n_estimators` from 535, or distilling to a smaller model, and measure the accuracy cost |
| **Caching** | Many requests repeat (same locality/size/BHK); an LRU cache on the input tuple would absorb a large share |
| **Framework** | FastAPI + Gunicorn/Uvicorn workers; Streamlit is a demo UI, not a serving layer |

### M10. The business says "under-predicting is worse than over-predicting." What changes?
**Switch to an asymmetric loss** — which I already have the machinery for:
1. **Train with the quantile loss at α > 0.5.** `GradientBoostingRegressor(loss="quantile",
   alpha=0.7)` penalises under-prediction 7:3 and shifts the whole estimate upward.
2. **Or just report the upper quantile** (`quantile_high`, α = 0.9) as the headline number instead
   of the mean estimate.
3. **Or define a custom cost function** with explicit rupee costs for each direction and select the
   model that minimises expected cost rather than MAE.

**And I would change the selection metric to match**, because the metric must mirror the business
cost function.

### M11. What if you discovered a data-entry error in the target after deployment?
1. **Quantify the blast radius** — how many rows, which cities, which rent bands?
2. **Re-run the experiment with the correction** and compare metrics against the deployed model.
3. **Check whether that row was in my worst-predictions list** — e.g. the Hyderabad 1 BHK at 8,000
   sq ft for ₹2,00,000 is almost certainly an error, and it is in my top-10 worst table.
4. **Version the data and the model** so the correction is traceable.
5. **Re-deploy only if the corrected model is better on the held-out test set.**

### M12. How would you detect that your deployed model has gone stale?
| Signal | How |
|---|---|
| **Prediction drift** | Monitor the distribution of predictions over time against the training distribution (PSI, KL divergence) |
| **Input drift** | Same, on each input column — a surge in unseen localities is a direct alarm |
| **Performance drift** | Log predictions, join against realised rents when available, track rolling MAE |
| **A tripwire** | Alert if rolling MAE exceeds the test MAE of ₹10,135 by more than, say, 25 % |

### M13. Your examiner says: "Just use the mean rent as the prediction. Why bother with ML?"
**Because that baseline is exactly what R² measures against, and I beat it substantially.**
```
Predicting the mean for everything  ->  R² = 0.000 by definition
My model                            ->  R² = 0.7144
```
In rupees: the constant-mean baseline would have RMSE equal to the test standard deviation of rent.
My model's RMSE is ₹32,716 and MAE ₹10,135 — a vast improvement.

**And a per-city median baseline is also beaten:** the ablation's "location only" model is
essentially a sophisticated version of that baseline, and it reaches only R² 0.397 against 0.714.

### M14. What if two models tie on every metric?
**My code already handles this explicitly.** The multi-criteria selection ranks on four criteria
and sums the ranks; if candidates tie on rank-sum, the code reports *"N candidates tie on rank-sum
— the four criteria do not separate them"* and breaks the tie on **test MAE**.

**Why MAE as the tiebreak:** it is the metric that is both **stable** on this data (unlike RMSE,
which is decided by ~10 test rows) and **aligned with the deployment use case** of estimating rent
for ordinary listings.

**If they still tied, I would prefer:** the simpler model (Occam's razor), then the faster one, then
the more interpretable one.

### M15. What if you had to explain a single prediction to a landlord?
Three levels, in increasing sophistication:
1. **The comparables story:** *"Flats of this size in this locality typically rent for ₹X, and the
   model adjusts for your bathroom count and floor."*
2. **Feature importance:** *"City and size drive most of the estimate; furnishing and tenant
   preference barely matter."*
3. **The right technical answer: SHAP values.** They decompose an individual prediction into
   per-feature contributions that sum exactly to the prediction. **I did not implement SHAP** — the
   project uses global importance (permutation + impurity) — and that is the honest gap to name if
   pushed on per-prediction explanations.

### M16. What if the dataset had a class imbalance?
**Class imbalance is a classification concept** — it does not apply to a continuous target.

**The regression analogue is target imbalance / rare-region scarcity**, and this dataset has it
badly: only a small fraction of listings exceed ₹1,00,000. **That is exactly why premium-band MAE
is ₹73,887 while the cheap-band MAE is ₹2,528.**

**Possible remedies:** sample weights that up-weight rare rent regions; stratifying the split on
binned rent as well as city; or training a separate specialist model for premium properties.

### M17. If you had two weeks more, what would you do?
In priority order:
1. **Geocode the localities.** Replacing free-text locality with latitude/longitude would fix the
   unseen-locality problem entirely — the dominant failure mode (8 of my 10 worst predictions).
2. **Add SHAP** for per-prediction explanations in the app.
3. **Multi-seed repetition** of the whole experiment to put confidence intervals on the model
   ranking.
4. **Try LightGBM/CatBoost** — CatBoost's ordered target statistics are a principled version of my
   manual target encoding.
5. **Conformal prediction** as a distribution-free alternative to quantile regression, with
   guaranteed marginal coverage.
6. **A specialist premium model** or a two-stage approach (predict the band, then the value).

---

## SECTION N — Trap Questions Designed to Derail You

> These are questions built on a **false premise**, or designed to make you contradict yourself.
> **The skill is spotting the premise and correcting it politely before answering.**

### N1. TRAP: "Your dataset has no missing values, so you didn't really do any data cleaning, did you?"
**The false premise:** that "0 NaNs" means "clean data".

**Your answer:**
> *The CSV reports zero NaNs, which is precisely why profiling first mattered. Three genuine
> problems were hidden: `Size` has values down to **10 sq ft** (16 rows below 50), which I blank
> and impute; the `Floor` column stores **two facts in one string** with 4 rows missing the
> building height entirely; and `Area Locality` has **2,235 levels with 1,465 appearing exactly
> once**. Finding problems that the null count cannot show you is the data cleaning.*

### N2. TRAP: "Linear Regression has a higher R² than your model. So your model is worse, isn't it?"
**The false premise:** that R² (or RMSE) is the right ranking metric on a heavy-tailed target.

**Your answer:** (this is question **E3** in Section E — the full version is there)
> *On the test set, **one listing out of 950 accounts for 42 % of total squared error** and ten
> account for 77 %. RMSE excluding the worst 1 % drops from ₹32,716 to ₹16,234. So R² and RMSE here
> are largely measuring about ten listings. Linear Regression wins them by **extrapolating** into
> the premium tail, which a tree ensemble structurally cannot do. On the other ~99 % of listings
> Gradient Boosting is better: MAE ₹10,135 vs ₹10,760 and MAPE 26.7 % vs 28.4 %. Since the tool
> estimates rent for ordinary listings, I selected on MAE and documented the trade-off — my code
> prints it automatically.*

### N3. TRAP: "You should have removed the outliers. That's standard practice."
**The false premise:** that outlier removal is unconditionally correct.

**Your answer:**
> *I tested it rather than assuming. Four settings, all six models, all scored on the same
> untreated test set. Removing outliers **raised** test MAE from ₹10,135 to ₹12,504, and in the
> premium band it raised MAE from **₹73,887 to ₹1,05,137 — a 42 % degradation** — while gaining ₹46
> on cheap flats. The `log1p` transform already handles the skew (21.41 → 0.909), so removal on top
> is redundant. **I did remove the genuine errors:** 16 listings with Size below 50 sq ft.*

### N4. TRAP: "Cross-validation already gives you a test score, so why keep a separate test set?"
**The false premise:** that CV scores are unbiased after tuning.

**Your answer:**
> *Because I **selected hyperparameters using those very folds**. The CV score of the winning
> configuration is the score of the configuration chosen to maximise it, so it is optimistically
> biased — a form of evaluation leakage. The held-out test set is touched only once, after
> everything is decided, which is the only way to get an unbiased number. My code documents this in
> `cv_from_search`, and both are reported side by side.*

### N5. TRAP: "You used the target to create the locality feature. That's leakage, your whole project is invalid."
**The false premise:** that all use of `y` during fitting is leakage.

**Your answer:**
> *Target encoding uses `y` during **fitting**, which is legitimate — so does every supervised
> model. Leakage would be using **validation-fold** targets, and I prevent that twice. First, the
> `TargetEncoder` is inside the `Pipeline`, so `cross_validate` clones and re-fits it per fold and
> it never sees the validation targets. Second, sklearn's `TargetEncoder` **cross-fits internally**
> during `fit_transform`, so even the training-fold encodings are out-of-fold. My code carries a
> comment saying "do not hoist this out of the pipeline."*
>
> *The feature that **would** be leakage is `rent_per_sqft = Rent / size`, and I explicitly keep it
> out of the model with a `!! LEAKAGE WARNING !!` in the source.*

### N6. TRAP: "Gradient Boosting is just a fancy Random Forest, right?"
**The false premise:** that they are the same family.

**Your answer:**
> *No — they are opposite strategies. Random Forest is **bagging**: independent trees built in
> parallel on bootstrap samples, averaged, which reduces **variance**. Gradient Boosting is
> **boosting**: trees built **sequentially**, each fitted to the previous ensemble's residuals,
> which reduces **bias**. RF uses deep low-bias trees; GB uses shallow weak learners — mine are
> depth 5. More trees never hurt a Random Forest; **too many trees will overfit a boosted model**.*

### N7. TRAP: "Why didn't you just use Deep Learning? It's better."
**The false premise:** that deep learning dominates on all problems.

**Your answer:**
> *With 3,796 training rows and structured tabular data, gradient-boosted trees are the better
> choice, and that is a well-established empirical result rather than a preference. A neural
> network would overfit badly, would give me no feature importances, and would take far longer to
> train for worse accuracy. My whole pipeline runs in 8.45 minutes on a laptop. **Deep learning
> would win** with hundreds of thousands of listings, or with unstructured inputs like property
> photos or free-text descriptions, or with learned embeddings for the 2,235 localities.*

### N8. TRAP: "Your accuracy is only 71 %." / "What's your model's accuracy?"
**The false premise:** that accuracy applies to regression.

**Your answer:**
> *Accuracy is a **classification** metric — the fraction of correctly classified instances. It is
> undefined for a continuous target, because there is no notion of a "correct" real number. My
> 0.7144 is **R², the proportion of variance explained** — a different quantity entirely. The
> closest thing to an accuracy statement here is: **25.5 % of test listings are predicted within
> ±10 % of actual rent and 59.9 % within ±25 %**, with a MAPE of 26.74 %.*

### N9. TRAP: "You only have 4,746 rows. That's not big data — is this even machine learning?"
**The false premise:** that ML requires big data.

**Your answer:**
> *Machine learning is defined by learning a function from data, not by data volume. 4,746 rows is
> a perfectly normal size for tabular regression, and it is exactly why I chose the methods I did:
> gradient-boosted trees over a neural network, 5-fold cross-validation rather than a single
> validation split, smoothed target encoding rather than one-hot for 2,235 localities, and
> randomised rather than grid search. **The dataset size drove every methodological choice**, and I
> list "small dataset" as my first named limitation.*

### N10. TRAP: "Your 80 % prediction interval only covers 77.8 %. It's broken."
**The false premise:** that nominal and empirical coverage must match exactly.

**Your answer:**
> *A 2.2-point shortfall on 950 test rows is well within sampling noise, and the shortfall is in
> the direction you would predict given the heavy tail — the few extreme listings fall outside.
> **Over-coverage would be the worse sign:** 95 % coverage on a nominal 80 % interval would mean
> the intervals are uselessly wide. The intervals are also validated on held-out data, which most
> projects do not bother to do at all.*

### N11. TRAP: "You dropped a column. Aren't you throwing away information?"
**The false premise:** that all columns carry signal.

**Your answer:**
> *I dropped `Posted On`, and I profiled it before deciding. The listings span **89 days**, from
> 2022-04-13 to 2022-07-11, with 81 distinct dates. That window is far too short to carry
> inflation, seasonality or market drift, so any date-derived feature would fit mid-2022 noise
> rather than signal. **Adding a feature that is pure noise does not help a model — it gives it
> something to overfit to.** The reasoning is recorded in the notebook, so the decision is auditable
> rather than silent.*

### N12. TRAP: "If City is the most important feature, just build six separate per-city models."
**A reasonable-sounding suggestion with real problems. Engage with it properly.**

**Your answer:**
> *It is a legitimate idea — it is essentially what adding full City interactions would do, and it
> would let each city have its own price structure. **But with this data it would hurt**, for three
> reasons. Kolkata would train on **419 rows** and Delhi on 484 — far too few for a 535-tree
> ensemble. Second, I would lose **cross-city information sharing**: the fact that bathroom count
> signals luxury is learned from all 3,796 rows, not 419. Third, it multiplies the maintenance and
> deployment surface by six.*
>
> *My evidence that one model already captures the city effect: `City` is one-hot encoded and is
> the top permutation feature, and the ablation shows dropping location costs 36.7 % RMSE. **With
> 10× the data, per-city models would be worth testing.***

### N13. TRAP: "Your model under-predicts expensive properties. Isn't that a fatal flaw?"
**Half-true premise. Concede the fact, reject the word "fatal".**

**Your answer:**
> *The fact is correct and I measured it: **9 of my 10 worst predictions are under-predictions**,
> because a tree ensemble structurally cannot extrapolate above the leaf values it learned, and
> because the `log1p` target deliberately down-weights the extreme tail.*
>
> *It is not fatal, for three reasons. The median residual is **−₹37.70**, so the model is well
> calibrated for a typical listing, which is the deployment use case. The **prediction interval
> widens automatically** for expensive properties, so the user is told. And the honest fix is more
> premium training data, not a different algorithm — **I showed that removing outliers makes this
> exact problem 42 % worse**, so the naive remedy is counterproductive.*

### N14. TRAP: "You said trees don't need scaling, but you scaled some features. Which is it?"
**Testing whether you actually understand your own code.**

**Your answer:**
> *Both, deliberately. `build_preprocessor` takes a `family` argument. For `family="linear"` it
> adds `StandardScaler` and the `size_x_*` interaction terms and uses `drop="first"` on one-hot.
> For `family="tree"` it does **none** of those. Linear models need scaling for conditioning and
> because **Ridge's penalty is scale-dependent**; trees split on `feature <= threshold`, so any
> monotone rescaling produces the identical tree. **Giving each family the preprocessing it needs
> is what makes the comparison fair** — the alternative is comparing a handicapped linear model
> against a well-served tree.*

### N15. TRAP: "Just add more features. More features always help."
**Your answer:**
> *They do not. Adding uninformative features increases variance and gives the model more
> opportunity to overfit — I have a live demonstration in this project: expanding ~24 features to
> **~324 polynomial terms** took test R² from 0.792 to **−223.59**. My permutation importance also
> shows `Tenant Preferred` contributes **₹23** of MAE and `Floor` has a slightly **negative** R²
> contribution, i.e. within noise. The features that would actually help are ones I do not have —
> amenities, building age, metro proximity, geocoordinates.*

### N16. TRAP: "Your CV standard deviation (₹39,836) is almost as large as the CV mean (₹49,039). That means your model is unreliable."
**The false premise:** that CV std measures model reliability rather than target heaviness.

**Your answer:**
> *That standard deviation measures the **variability of the metric across folds**, which on this
> target is driven by **which extreme listings land in which fold**. It is a property of the data,
> not of the model — and it is true of **every** model here: CV RMSE std/mean is 0.60–0.81 for all
> six. **The same folds give CV MAE std/mean of only 0.10–0.15**, which is why I tune and rank on
> MAE. I measured that noise rather than assuming it, and I report it as `metric_stability` in
> `metrics.json`.*

### N17. TRAP: "Why is your training RMSE higher than your test RMSE? That's impossible."
**The false premise:** that train error must always be lower.

**Your answer:**
> *It is unusual but entirely possible, and here the cause is identifiable: **the single
> ₹35,00,000 listing sits in the training split**, so train RMSE (₹58,039) is inflated by one row
> relative to test RMSE (₹32,716). That is why I report **two** overfitting diagnostics and flag
> the naive one as misleading. `cv_overfit_gap_rmse` compares 5-fold validation RMSE against 5-fold
> **training** RMSE on the same data, which is the statistically sound gap, and it is what my
> automatic selection ranks on — on its **absolute** value, because a large negative gap is an
> artifact, not extra generalisation.*

### N18. TRAP: "You could get R² = 0.99 if you tried harder."
**Your answer:**
> *I could get R² ≈ 1.0 right now by adding `rent_per_sqft` as a feature — and it would be
> **leakage**, because `rent_per_sqft = Rent / size` reconstructs the target exactly. My code
> explicitly keeps it out with a leakage warning. **A suspiciously perfect R² on this problem is a
> red flag, not an achievement.** The realistic ceiling here is set by the features I do not have:
> amenities, building age, parking, metro proximity, society quality, view — and the fact that the
> target is asking price, not negotiated rent.*

### N19. TRAP: "Point of Contact is one of your top features. That looks like leakage to me."
**Your answer:** (concede the shape of the concern, then resolve it)
> *It is a fair thing to question, and I flag it myself. It is **not** leakage: `Point of Contact`
> is known at listing time, so it is legitimately available at prediction time, and the app
> collects it as an input. But it is **not a property attribute** either — it encodes market
> positioning, because agent-listed properties are systematically more expensive. **If the goal
> were a valuation model rather than a listing-price model, I would drop it**, and I would expect
> MAE to rise by roughly the ₹2,775 its permutation importance implies.*

### N20. TRAP: "Your model is a black box. How can anyone trust it?"
**Your answer:**
> *Four layers of interpretability, three of which are in the app. **Global:** permutation
> importance on the original columns, measured on held-out data, says `City` ₹7,290 and `Size`
> ₹5,506. **Structural:** impurity importance on encoded columns. **Counterfactual:** the locality
> ablation isolates what each block of features contributes. **Per-prediction uncertainty:** an
> 80 % interval with validated 77.8 % coverage.*
>
> *What I do **not** have is per-prediction attribution — **SHAP values** would give that, and it
> is the honest gap. It is next on my list.*

### N21. TRAP: "You used `random_state=42` — you just picked the seed that gave the best results, didn't you?"
**Your answer:**
> *No. The seed is set **once** in `config.py` and threaded through every split, every search and
> every model, so the whole experiment is reproducible. I did not search over seeds. I also report
> the **5-fold CV standard deviation** alongside every mean, which is precisely the honest way to
> express split-dependent uncertainty.*
>
> *That said, a multi-seed study would be stronger and I would call it the right next step: 24
> model fits take 8.45 minutes, so ten seeds is about 85 minutes — entirely feasible.*

### N22. TRAP: "Everyone's rent prediction project gets R² above 0.9. Why is yours lower?"
**Your answer:**
> *Usually for one of three reasons, and all three are worth checking. **Leakage** — using
> `rent_per_sqft`, or any feature computed from rent, gives ~1.0 instantly. **Scoring in log
> space** — R² on `log(Rent)` is much higher than R² on rupees, because logging compresses the
> scale; I deliberately report **every metric in rupees**. **Removing the test-set outliers** —
> which makes the test set easier rather than the model better; I score every one of my 24
> configurations on the same **untreated** test set.*
>
> *My 0.7144 is computed on raw rupees, on an untreated held-out test set, with no target-derived
> features. **It is a lower number and a more honest one.***

### N23. TRAP: "Which is better, MAE or RMSE?"
**The false premise:** that one is universally better.

**Your answer:**
> *Neither — it depends on the cost function of the application. **RMSE** is right when large
> errors are disproportionately costly: structural engineering, medical dosing, or a platform that
> guarantees its estimate and pays the difference. **MAE** is right when you care about typical
> performance and the target is heavy-tailed.*
>
> *Here, **one test row causes 42 % of the squared error**, so RMSE measures that row rather than
> the model — and CV RMSE is too noisy to tune on (std/mean 0.60–0.81 vs 0.10–0.15 for MAE). So I
> rank on MAE and report RMSE. **If the business told me a single large miss was catastrophic, I
> would switch.***

### N24. TRAP: "What's your model's p-value? Is it statistically significant?"
**The false premise:** that predictive models are tested for significance.

**Your answer:**
> *p-values belong to **statistical inference** — testing whether a coefficient differs from zero —
> which requires assumptions my residuals violate (Shapiro–Wilk p = 1.9e-49, funnel correlation
> 0.642). **Gradient Boosting has no coefficients to test.** The predictive analogue of
> significance is **out-of-sample performance on data the model never saw**, which is exactly what
> my held-out test set provides, plus the 5-fold CV standard deviation as a measure of uncertainty.*
>
> *The one p-value I do report is **Shapiro–Wilk on the residuals**, as a diagnostic — and with
> n = 950 it would reject normality for almost any real data, so I read the **effect size** (skew
> 9.71, kurtosis 185) rather than the p-value.*

### N25. TRAP: "Your project is just calling scikit-learn functions. What did YOU actually do?"
**Your answer:**
> *The library calls are the easy part. The work is in the decisions, and each one is backed by a
> measurement I took:*
> - *Profiling the real data before writing any preprocessing — which is how I found 16 impossible
>   `Size` values and the missingness hidden inside the `Floor` string.*
> - *Writing a **custom stateful transformer** (`FeatureEngineer`) because a stateless function
>   would leak group medians across CV folds.*
> - *Testing the conventional area ratios empirically and finding that **Super Area's observed
>   ratio is ~1.0, not the conventional 0.70**.*
> - *Comparing per-city against global outlier rules and finding a **global rule would flag 37.8 %
>   of Mumbai**.*
> - *Running a **4-setting × 6-model experiment** on one frozen untreated test set.*
> - *Measuring that **CV RMSE is 60–81 % noise** and switching the tuning criterion to MAE.*
> - *Quantifying that **one test row is 42 % of total MSE** and justifying the selection metric
>   from it.*
> - *Running an **ablation** to answer "does locality dominate" with a number rather than an
>   opinion.*
> - *Building **validated quantile intervals** rather than a cosmetic ± MAE band.*
>
> *scikit-learn provides the algorithms. **The experimental design is mine.***

---

## SECTION O — Alternative Framings
### (the same questions, reworded to confuse you — map each back to the core answer)

> **Technique:** when a question feels unfamiliar, identify which of the ~15 core topics it
> actually belongs to, then give that answer. The wording changes; the answer does not.

### O1. The log-transform question
| Framing | Maps to |
|---|---|
| "Why did you log the target?" | **Core: skew 21.41 → 0.909** |
| "Why not model rent directly?" | same |
| "What would change if you removed the log1p?" | same + *"that is literally my experiment setting 1: Linear R² drops 0.792 → 0.582; GB MAE rises ₹10,135 → ₹11,006"* |
| "Your target is in log space — isn't that cheating on the metrics?" | *"No — `TransformedTargetRegressor` applies `expm1` inside `predict`, so every reported metric is in rupees."* |
| "What does a coefficient mean in a log model?" | *"It is multiplicative: `log(Rent) = β₀ + β₁·Size` means each extra sq ft multiplies rent by `e^β₁`."* |
| "Why `log1p` rather than `log`?" | *"`log(0) = −∞`; `log1p(0) = 0`. Safe habit, more accurate for small x."* |
| "Did you consider a Box–Cox or Yeo–Johnson transform?" | *"Legitimate alternatives that estimate the exponent from data. `log1p` is the λ = 0 special case, it is interpretable as a multiplicative model, and it already brought skew to 0.909 — so the extra fitted parameter was not worth it."* |

### O2. The model-selection question
| Framing | Maps to |
|---|---|
| "Which model is best?" | **Core: GB on MAE; Linear on RMSE; here is why MAE** |
| "Why not the model with the highest R²?" | same |
| "Linear Regression beats you on two metrics. Defend yourself." | same |
| "Is a simpler model not preferable?" | *"Occam's razor applies when performance ties. It does not here: GB wins MAE, MAPE and latency. And Linear's RMSE edge comes from extrapolating on ~10 listings."* |
| "If I forced you to deploy Linear Regression, what would happen?" | *"Typical error rises — MAE ₹10,760 vs ₹10,135, MAPE 28.4 % vs 26.7 % — and I would gain better behaviour on premium properties. If the business cared most about never badly missing an expensive flat, that would be the right trade."* |
| "What if your examiner's rubric says to rank on RMSE?" | *"Then Linear Regression wins at ₹27,897 and I would say so. My code reports the RMSE winner explicitly alongside the MAE winner, and prints the trade-off automatically."* |

### O3. The outlier question
| Framing | Maps to |
|---|---|
| "How did you treat outliers?" | **Core: detected per-city, tested 3 strategies, kept them** |
| "Why didn't you clean your data properly?" | same + *"I did remove the 16 genuine errors"* |
| "Isn't a ₹35-lakh rent obviously an error?" | *"No — it is a real Mumbai listing. Mumbai's median rent per sq ft is ₹71.4, 5.4× Kolkata's. Errors are things like Size = 10 sq ft, which I did remove."* |
| "What happens to your metrics if you remove outliers from the test set too?" | *"They improve, and the improvement is meaningless — the test set got easier, the model did not get better. That is exactly why I score all 24 configurations on one frozen untreated test set."* |
| "Why 1.5 × IQR and not 3?" | *"Tukey's convention; ≈ ±2.7σ under normality. 3.0 is the 'far out' rule. 1.5 flags a sane 7–11 % per city."* |
| "Would you use a Z-score instead?" | *"No — μ and σ are themselves destroyed by skew 21.41. IQR uses quartiles, which are robust. A modified Z-score using the median absolute deviation would be acceptable."* |

### O4. The encoding question
| Framing | Maps to |
|---|---|
| "How did you handle categorical variables?" | **Core: one-hot for low cardinality, target encoding for locality** |
| "Why not label-encode the cities?" | *"Label encoding imposes a false ordering — it would tell a linear model Bangalore is 'twice' Delhi. One-hot has no such implication."* |
| "Why not one-hot the localities too?" | *"2,235 levels on 3,796 training rows — more columns than rows for one feature, 1,465 of them containing a single 1."* |
| "Isn't target encoding dangerous?" | *"It is, which is why it is inside the Pipeline AND internally cross-fitted."* |
| "What if a locality has only one listing?" | *"The smoothing formula pulls it almost entirely to the global mean: (1·180,000 + 10·34,993)/11 ≈ ₹48,176 rather than ₹1,80,000."* |
| "What is `m` in your smoothing formula?" | *"The smoothing strength. `smooth='auto'` estimates it empirically from the within-category versus between-category variance — empirical-Bayes shrinkage."* |
| "Would frequency encoding work?" | *"It would reduce 2,235 columns to 1, but it encodes popularity rather than price level. Target encoding directly encodes what I care about. For a tree, a hybrid of both is often strong."* |

### O5. The cross-validation / leakage question
| Framing | Maps to |
|---|---|
| "Explain your validation strategy." | **Core: one stratified 80/20 split + 5-fold CV inside the training split** |
| "How do you know you're not leaking?" | **Core: everything inside the Pipeline, cloned per fold** |
| "Why does the preprocessing go inside the pipeline?" | same |
| "What is `clone()` doing?" | *"Building a fresh unfitted copy from the constructor parameters, which is why `FeatureEngineer` inherits `BaseEstimator`. Without it, a fitted transformer would be reused across folds — leakage."* |
| "Would your results change if you fitted the scaler on all the data first?" | *"CV scores would become optimistically biased. The test score would be subtly contaminated too, since the test rows would have influenced μ and σ."* |
| "Why 5 folds? Why not 10? Why not leave-one-out?" | *"Bias/variance/compute. 24 model fits already take 8.45 minutes; LOO would need 3,796 fits per model."* |
| "Is your test set big enough?" | *"950 rows. Enough that the MAE standard error is small relative to the differences between models, and enough to compute per-city metrics with 105–195 rows per city."* |

### O6. The metrics question
| Framing | Maps to |
|---|---|
| "What metrics did you use?" | **Core: R², MSE, RMSE, MAE + MAPE, all in rupees** |
| "What's your accuracy?" | **TRAP — accuracy is a classification metric.** See question **N8** |
| "How good is your model, in plain English?" | *"Typical error about ₹10,135, or 26.7 % of actual rent. Three listings in five land within a quarter of the true value."* |
| "Is R² = 0.71 good?" | *"For real-estate pricing from 10 attributes with no amenities, building age or photos — yes, reasonable. And the residual 29 % is unmodelled features, which I can name."* |
| "Why is your MSE such a huge number?" | *"Because its units are **rupees squared**. It is reported because the spec requires it; RMSE is its interpretable square root."* |
| "Why report five metrics instead of one?" | *"Because they disagree, and the disagreement is the finding: Linear wins R²/RMSE by extrapolating on ten listings, GB wins MAE/MAPE on the other 940."* |

### O7. The "what is this project" question
| Framing | Maps to |
|---|---|
| "Is this supervised or unsupervised?" | **Core: supervised regression** |
| "What kind of ML problem is this?" | same |
| "Why isn't this clustering?" | same + *"clustering cannot output a number in rupees"* |
| "Could you solve this with classification?" | *"Yes, by binning — I already define bands in `rent_band()` — but it throws away information and a landlord wants a number."* |
| "Is this a prediction or an inference problem?" | *"**Prediction.** I care about out-of-sample accuracy, not about unbiased estimates of individual coefficients. That is why I use a black-box ensemble and why violated OLS assumptions do not invalidate the model."* |
| "Is this parametric or non-parametric?" | *"My chosen model is **non-parametric** (tree ensemble). I also fit parametric baselines (Linear, Polynomial, Ridge). The distinction explains my central finding: only the parametric models extrapolate."* |

### O8. The feature-importance question
| Framing | Maps to |
|---|---|
| "Which feature matters most?" | **Core: City (₹7,290), then Size (₹5,506)** |
| "Does location dominate?" | **Core: the ablation — 0.397 vs 0.714, so no** |
| "What would happen if you dropped the city?" | *"R² falls 0.714 → 0.466, RMSE rises 36.7 %. That is ablation model B."* |
| "Why is BHK so unimportant?" | *"It is correlated with Size and Bathroom, so permutation importance under-credits it. Bathroom count is also a better luxury proxy."* |
| "Can I trust your importance numbers?" | *"Permutation importance yes — measured on original columns, on held-out data, 10 repeats with standard deviations reported. Impurity importance is biased towards high-cardinality features and I report it only as a contrast."* |
| "Which features would you drop?" | *"`Tenant Preferred` (₹23) and possibly `Area Type` (₹146) are within noise. I kept them because the case study names them and because dropping them changes nothing measurable."* |

### O9. The deployment question
| Framing | Maps to |
|---|---|
| "How is this deployed?" | **Core: Streamlit app loading one joblib pipeline** |
| "Could a company use this tomorrow?" | *"As a suggestion inside a listing form, with the range shown and a human override — yes. As an authoritative valuation — no, and the Limitations tab says so."* |
| "What happens at prediction time?" | *"The app builds a one-row DataFrame and calls `.predict()`. All preprocessing is inside the artifact — no training/serving skew."* |
| "Why Streamlit and not Flask?" | *"Zero HTML/CSS/JS for a model-demo UI. For production serving I would use FastAPI; Streamlit is a demo layer, not a serving layer."* |
| "What if two users hit it at once?" | *"`@st.cache_resource` shares the loaded model by reference, which is safe because prediction is read-only. `@st.cache_data` returns copies for the JSON, so no cross-session mutation."* |

### O10. The "how does X work" family
| Framing | Maps to |
|---|---|
| "Explain Random Forest." | **Core: bagging + feature subsetting; the `ρσ² + (1−ρ)σ²/B` formula** |
| "Why is it called 'random'?" | same — *two* sources of randomness |
| "What is bootstrapping?" | *"Sampling n rows with replacement; ~63.2 % distinct, ~36.8 % out-of-bag."* |
| "What is out-of-bag error?" | *"A free validation estimate from the ~36.8 % of rows each tree did not see."* |
| "Explain Gradient Boosting." | **Core: sequential, fits pseudo-residuals = negative gradient, scaled by ν** |
| "Why 'gradient'?" | *"It is gradient descent in function space, not parameter space."* |
| "What is a weak learner?" | *"A model slightly better than chance. Boosting combines many; mine are depth-5 trees."* |
| "What is the learning rate doing?" | *"Scaling each tree's contribution. Small ν needs more trees but generalises better — mine is 0.0121 with 535 trees."* |

---

## SECTION P — Rapid-Fire One-Liners

| Question | Answer |
|---|---|
| Rows? | 4,746 |
| Columns? | 12 raw, 10 used, ~24 after encoding |
| Train/test? | 3,796 / 950 (80/20, stratified by City) |
| Target? | `Rent`, monthly, in ₹ |
| Problem type? | Supervised regression |
| Missing values? | 0 explicit, 3 hidden problems |
| Duplicate rows? | 0 |
| Cities? | 6 — Mumbai, Delhi, Bangalore, Chennai, Hyderabad, Kolkata |
| Localities? | 2,235 (1,465 appear once) |
| Target skew? | 21.41 raw → 0.909 after `log1p` |
| Median rent? | ₹16,000 |
| Mean rent? | ₹34,993 |
| Max rent? | ₹35,00,000 |
| Models trained? | 6 variants × 4 settings = 24 fits |
| Best model? | Gradient Boosting Regressor |
| Best setting? | `log1p` target, **no** outlier treatment |
| Test R²? | 0.7144 |
| Test RMSE? | ₹32,716 |
| Test MAE? | ₹10,135 |
| Test MAPE? | 26.74 % |
| Test MSE? | 1,070,335,994 ₹² |
| CV RMSE? | ₹49,039 ± ₹39,836 |
| Runner-up? | Random Forest (MAE ₹10,245) |
| Lowest RMSE? | Linear Regression (₹27,897) |
| Learning rate? | 0.012115 |
| n_estimators? | 535 |
| max_depth? | 5 |
| subsample? | 0.7545 |
| min_samples_leaf? | 6 |
| Prediction latency? | 6.53 ms |
| Training runtime? | 8.45 minutes |
| CV folds? | 5, shuffled, seed 42 |
| Tuning method? | RandomizedSearchCV, `refit="mae"` |
| Most important feature? | `City` (₹7,290 MAE increase) |
| 2nd most important? | `Size` (₹5,506) |
| Least important? | `Tenant Preferred` (₹23) |
| Does locality dominate? | No — 0.397 vs 0.714 |
| Outliers flagged? | 383 of 3,796 training rows |
| Outlier rule? | Per-city 1.5 × IQR on Rent + per-city rent-per-sqft percentiles |
| Global IQR on Mumbai? | Would flag **37.8 %** — hence per-city |
| Effect of removing outliers? | Test MAE ₹10,135 → ₹12,504; premium MAE +42 % |
| Prediction interval? | 80 % nominal, **77.8 % empirical** |
| Interval method? | Two GB models with quantile loss, α = 0.10 / 0.90 |
| Interval width? | 74.3 % of actual rent (median) |
| Residual median? | −₹37.70 |
| Residual skew / kurtosis? | 9.71 / 185.09 |
| Funnel correlation? | 0.642 |
| Shapiro p? | 1.9e-49 (not normal) |
| Within ±10 % / ±25 %? | 25.5 % / 59.9 % |
| Error concentration? | 1 row = 42.2 % of MSE; 10 rows = 76.9 % |
| Why tune on MAE? | CV RMSE std/mean 0.60–0.81; CV MAE 0.10–0.15 |
| Worst city? | Kolkata (R² 0.10, RMSE 126.5 % of mean) |
| Best city? | Chennai (R² 0.79, RMSE 60.8 % of mean) |
| Why Kolkata worst? | 524 listings, 35.2 % unseen localities, std > mean |
| Deployment? | Streamlit, `streamlit run app/app.py` |
| Serialisation? | joblib |
| Random seed? | 42, set once in `config.py` |
| scikit-learn version? | 1.9.1 (pinned exactly) |
| Dropped column? | `Posted On` (89-day span) |
| Leakage feature avoided? | `rent_per_sqft` |
| Engineered features? | `floor_num`, `total_floors`, `floor_ratio`, `carpet_equiv_size`, 3 × `size_x_*` |
| Area ratios? | Carpet 1.00, Built 0.85, Super 0.70 |
| Did the ratios hold? | Super Area observed ≈ 1.0, **not 0.70** — reported, features kept |
| Figures produced? | 19 |

---

## SECTION Q — What YOU Should Volunteer

> Steer the viva onto your strongest ground. These are the points that make an examiner think
> *"this student actually understands what they built."* Work them in whenever the topic is close.

### Q1. "I profiled before I preprocessed."
> *Before writing a single line of preprocessing I profiled the real CSV. That is how I found 16
> listings with `Size` below 50 sq ft, missingness hidden inside the `Floor` string, and that 1,465
> of 2,235 localities appear exactly once. None of that shows up in a null count.*

### Q2. "I tested the outlier assumption instead of following the recipe."
> *The standard recipe says remove outliers. I ran four settings across all six models on one
> frozen untreated test set and found that removal **raises** premium-band MAE by 42 %. The log
> transform already does the job. So I kept them, and I report the experiment that proves it.*

### Q3. "I measured the noise in my metrics before choosing one."
> *CV RMSE has a standard deviation 60–81 % of its own mean on this target; CV MAE is 10–15 %.
> Tuning on RMSE was selecting hyperparameters essentially at random — it is how the Ridge
> polynomial got a negative test R² on an early pass. So I switched the tuning criterion to MAE and
> still report all four metrics.*

### Q4. "One test listing causes 42 % of my squared error."
> *That one diagnostic reframes the entire comparison. It is why Linear Regression's RMSE win is
> about ten listings rather than general accuracy, and why I select on MAE. RMSE excluding the
> worst 1 % halves, from ₹32,716 to ₹16,234.*

### Q5. "My prediction intervals are exact under the log transform."
> *Quantiles are invariant under monotone transforms, so `expm1(Q_α(log1p(Rent)))` is exactly the
> α-quantile of rent — no approximation. The mean is **not** transform-invariant, which is the
> known source of bias when back-transforming log predictions. And I validated the coverage:
> 77.8 % against a nominal 80 %.*

### Q6. "I flag `Point of Contact` myself."
> *It is my 4th most important feature at ₹2,775, but it is not a property attribute — it encodes
> market positioning, because agent-listed properties are systematically more expensive. It is not
> leakage, because it is known at listing time, but for a valuation model rather than a
> listing-price model I would drop it.*

### Q7. "I tested the domain assumption and it failed."
> *The conventional multiplier for super area to carpet area is 0.70. I tested it within comparable
> (City, BHK) groups and the **observed ratio is close to 1.0**. And `Built Area` has only 2 rows
> in the whole dataset, so its ratio cannot be estimated empirically at all. I applied the
> conversion as specified but **kept raw `Size` and `Area Type` as features** so the models can
> ignore it.*

### Q8. "I know what my model cannot do."
> *A tree ensemble cannot extrapolate. Nine of my ten worst predictions are under-predictions of
> premium properties, and that is structural, not a tuning problem. Removing outliers makes it 42 %
> worse. The real fix is more premium training data.*

### Q9. "I know the gap in my interpretability story."
> *I have global importance from two methods, a counterfactual ablation, and validated prediction
> intervals. What I do **not** have is per-prediction attribution — **SHAP values** would give
> that, and it is next on my list.*

### Q10. "My documentation cannot go stale."
> *`README.md` is generated from `models/metrics.json` by `make_readme.py`, and the notebook is
> generated by `build_notebook.py` importing the same `src/` functions `train.py` uses. The numbers
> in my report physically cannot drift out of step with the model saved in `models/`.*

---

## SECTION R — Things NOT to Say

| ❌ Never say | ✅ Say instead |
|---|---|
| "My accuracy is 71 %" | "My **R²** is 0.7144 — the proportion of variance explained. Accuracy is a classification metric." |
| "I cleaned the data" | "I profiled it, found 16 implausible `Size` values and missingness hidden in the `Floor` string, and handled each specifically." |
| "I removed the outliers" | "I **tested** removal, capping and keeping. Keeping won — removal raised premium MAE by 42 %." |
| "Gradient Boosting is the best algorithm" | "Gradient Boosting won **on MAE and MAPE for this dataset**. Linear Regression won on RMSE, and I can explain exactly why." |
| "The model is 71 % correct" | "The model explains 71 % of the variance. Typical error is ₹10,135, or 26.7 %." |
| "I used machine learning" | "I used **supervised regression** — six algorithms compared under four experimental settings." |
| "I split the data 80/20" | "I made **one** stratified 80/20 split, frozen for the whole experiment, and never treated or tuned on the test set." |
| "I tuned the hyperparameters" | "`RandomizedSearchCV`, 5-fold, `refit='mae'` — and I can tell you why MAE rather than RMSE." |
| "The polynomial model didn't work" | "The polynomial model reached test R² **−223.59** — a textbook overfitting demonstration, exactly what the comparison exists to show." |
| "I don't know" | "I did not test that, but here is what I would expect and how I would check it: …" |

---

## SECTION S — The 60-Second Project Pitch
### (memorise this; it opens most vivas)

> *This is **Case Study 36, Residential Rent Prediction** — a **supervised regression** problem.
> I predict the monthly rent of an Indian residential listing from ten property attributes, using
> 4,746 real Kaggle listings across six cities.*
>
> *The target is severely right-skewed — **skewness 21.41** — so every model is trained on
> **`log1p(Rent)`**, which brings the skew to 0.909. The biggest preprocessing challenge was
> `Area Locality`: **2,235 levels, 1,465 appearing exactly once**, which I handle with smoothed
> **target encoding** placed inside the pipeline so it is re-fitted per cross-validation fold.*
>
> *I compared **six model variants across four experimental settings — 24 fits — all scored on one
> frozen, untreated 950-listing test set**. The selected model is a **Gradient Boosting Regressor**
> with **test R² 0.7144, RMSE ₹32,716 and MAE ₹10,135**.*
>
> *Two findings are worth highlighting. First, **outlier treatment hurt**: removing outliers raised
> premium-band error by 42 %, so the deployed model keeps them. Second, **one test listing accounts
> for 42 % of total squared error**, which is why I select on MAE rather than RMSE and document the
> trade-off rather than hiding it.*
>
> *It is deployed as a **Streamlit app** that returns a point estimate plus a validated **80 %
> prediction interval** built from two quantile-loss models, with 77.8 % empirical coverage.*

---

## Final Checklist Before You Walk In

**Numbers you must know cold:**
```
4,746 rows | 3,796 train | 950 test
21.41 skew -> 0.909
2,235 localities | 1,465 singletons
R² 0.7144 | RMSE ₹32,716 | MAE ₹10,135 | MAPE 26.74 %
42 % of MSE from ONE test row
Removing outliers: premium MAE ₹73,887 -> ₹1,05,137
Ablation: 0.714 all / 0.466 no-location / 0.397 location-only
80 % interval -> 77.8 % empirical coverage
City ₹7,290 > Size ₹5,506 > Bathroom ₹2,804
```

**Five sentences you should be able to say without thinking:**
1. *"This is supervised regression — the target is continuous and labelled."*
2. *"Skew 21.41 → 0.909 is why I log-transformed the target."*
3. *"One test listing is 42 % of my squared error, which is why I rank on MAE."*
4. *"Removing outliers raised premium-band error by 42 %, so I kept them — and I have the table."*
5. *"Everything is inside the Pipeline, so cross-validation re-fits every learned statistic per
   fold and the app needs zero preprocessing."*

**If you are stuck:** say what you measured, then say what you would check next. Never guess a
number — if you cannot recall one, say *"it is in `models/metrics.json`; the order of magnitude is
X and I can show you."*

---

*Companion document: [Project_Details.md](Project_Details.md) — full technical reference.*
*All figures referenced are in `reports/figures/`. All tables are in `reports/*.csv` and
`models/metrics.json`.*
