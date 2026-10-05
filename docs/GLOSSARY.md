# Glossary — Residential Rent Prediction

Every abbreviation, acronym and technical concept used anywhere in this project,
with its full form, a plain-English meaning, and where it actually appears in
the code or report.

**Notation used below:** `y` = actual rent, `ŷ` = predicted rent, `n` = number of
listings, `ȳ` = mean of the actual rents.

---

## 1. Evaluation metrics

| Short | Full form | What it means | Where it appears |
|---|---|---|---|
| **R²** | **R-squared**, or the **Coefficient of Determination** | The fraction of the variation in rent that the model explains. `R² = 1 − SS_res/SS_tot`. 1.0 is perfect, 0 means no better than always predicting the mean, and it **can go negative** when a model is worse than that. Ours is 0.7144. | `src/evaluate.py`, every comparison table |
| **MSE** | **Mean Squared Error** | Average of the squared errors: `MSE = (1/n) Σ(y − ŷ)²`. Squaring punishes big mistakes much harder than small ones. Its units are rupees², so it is not directly interpretable. | `src/evaluate.py`, required by the case study |
| **RMSE** | **Root Mean Squared Error** | `RMSE = √MSE`. The square root brings it back to rupees so it is readable. Still dominated by the largest errors. Ours is ₹32,716. | Headline metric throughout |
| **MAE** | **Mean Absolute Error** | Average of the unsigned errors: `MAE = (1/n) Σ|y − ŷ|`. Treats every listing equally, so it describes the *typical* mistake rather than the worst ones. Ours is ₹10,135. | Used for tuning and model selection |
| **MAPE** | **Mean Absolute Percentage Error** | MAE expressed as a share of actual rent: `MAPE = (100/n) Σ|y − ŷ|/y`. The fair way to compare a ₹10,000 flat with a ₹3,00,000 one. Ours is 26.7%. | Added beyond the spec because it is the most intuitive answer to "how accurate is it?" |
| **Residual** | — | One prediction's error: `residual = y − ŷ`. Positive = the model under-predicted. A plot of residuals is the main diagnostic for whether a regression is well behaved. | Sections 8 and 10 |

### Why RMSE and MAE disagree in this project

RMSE squares the errors, so a handful of very expensive listings dominate it. In
our test set **the single worst prediction accounts for ~13% of total MSE, and
the worst 10 of 950 account for ~55%**. MAE has no such property. That is why
Linear Regression wins on RMSE while Gradient Boosting wins on MAE — and why
models are tuned and selected on MAE.

---

## 2. Statistics and distribution terms

| Term | Full form / meaning | Where it appears |
|---|---|---|
| **IQR** | **Interquartile Range** = Q3 − Q1, the spread of the middle 50% of the data. The classic outlier rule flags anything below `Q1 − 1.5×IQR` or above `Q3 + 1.5×IQR`. | Outlier detection, `src/preprocess.py` |
| **Q1 / Q3** | **First** and **third quartile** — the 25th and 75th percentile. | Same |
| **Percentile / quantile** | The value below which a given share of the data falls. The 99th percentile of rent is the figure 99% of listings fall below. | Winsorizing, quantile models |
| **Median** | The 50th percentile — the middle value. Unlike the mean it is not dragged by extremes. Our rent median is ₹16,000 while the mean is ₹34,993. | Imputation, EDA |
| **Skewness** | How lopsided a distribution is. 0 = symmetric; large positive = a long right tail. Raw rent has a skew of **21.41**; `log1p(rent)` has **0.91**. | The justification for the log transform |
| **Kurtosis** | How heavy the tails are — how common extreme values are compared with a bell curve. High kurtosis means rare huge errors. | Residual diagnostics, section 10 |
| **std / σ** | **Standard deviation** — typical distance from the mean. | CV spread, per-city tables |
| **Coefficient of variation** | `std ÷ mean`. A scale-free measure of noise. We use it to show CV RMSE is unusable (0.60–0.83) while CV MAE is fine (0.10–0.15). | Model selection |
| **Q-Q plot** | **Quantile-Quantile plot**. Plots your data's quantiles against a normal distribution's. Points on the straight line = normally distributed; curving away at the ends = heavy tails. | Section 10 residual diagnostics |
| **Heteroscedasticity** | Error variance that changes across the range — the "funnel shape" in a residual plot. Ours funnels because the model controls *percentage* error, so rupee error naturally grows with rent. | Section 10 |
| **Winsorizing** | Clipping extreme values to a percentile instead of deleting the rows. Our `cap` strategy winsorizes rent at each city's 1st and 99th percentile. | `treat_outliers()` |
| **Correlation** | How strongly two numeric variables move together, from −1 to +1. Shown as a heatmap in EDA. | EDA section |

---

## 3. Core machine-learning concepts

| Term | Full form / meaning |
|---|---|
| **ML** | **Machine Learning** |
| **EDA** | **Exploratory Data Analysis** — plotting and summarising the data before modelling, to find its shape, errors and surprises |
| **Regression** | Predicting a **continuous number** (rent in rupees), as opposed to classification, which predicts a category |
| **Feature** | An input column the model learns from (size, city, BHK). Also called a predictor or independent variable |
| **Target** | The thing being predicted — here `Rent`. Also called the label or dependent variable |
| **Feature engineering** | Creating new, more useful features from raw ones — e.g. splitting `"Ground out of 2"` into `floor_num = 0` and `total_floors = 2` |
| **Overfitting** | Memorising the training data, including its noise, so performance collapses on new data |
| **Underfitting** | The model is too simple to capture the real pattern, so it is bad on training *and* new data |
| **Generalisation** | How well a model performs on data it has never seen. The whole point |
| **Generalisation gap** | Validation error minus training error. Large positive = overfitting. See the warning in §11 below |
| **Bias–variance trade-off** | Simple models err by being systematically wrong (bias); complex ones by being unstable (variance). Tuning balances the two |
| **Data leakage** | When information that would not be available at prediction time sneaks into training, producing scores that look great and collapse in production. **Two guards in this project** — see §7 |
| **Extrapolation** | Predicting outside the range of values seen in training. Linear models extrapolate; **tree models cannot** — their predictions are capped by the leaf values they learned. This is exactly why Linear Regression beats the tree ensembles on RMSE here |
| **Baseline** | A deliberately simple reference model. Ours is plain Linear Regression with no tuning |
| **Regularisation** | Penalising large coefficients to stop a model fitting noise. **L2 / Ridge** adds a penalty proportional to the squared coefficients |
| **Multicollinearity** | Input features that are highly correlated with each other, making coefficients unstable and huge. Degree-2 polynomial expansion creates it in bulk |
| **Hyperparameter** | A setting you choose *before* training (tree depth, number of trees, Ridge's alpha), as opposed to a parameter the model learns |
| **Ensemble** | Combining many models into one. **Bagging** trains them in parallel on resampled data (Random Forest); **boosting** trains them in sequence, each fixing the last one's mistakes (Gradient Boosting) |
| **Ablation study** | Deliberately removing a group of features and re-measuring, to find out how much they were actually contributing. Ours answers "does locality dominate?" |

---

## 4. The six models

| Short | Full name | How it works, in one line |
|---|---|---|
| — | **Linear Regression** | Fits a straight-line relationship: rent = a weighted sum of the features |
| — | **Polynomial Regression (degree 2)** | Linear regression after adding every squared term and every pairwise product, letting it bend. Our ~24 features expand to ~324 terms |
| — | **Ridge Regression** | Linear regression with an **L2** penalty that shrinks coefficients towards zero, controlling the instability that the polynomial expansion creates |
| **DT** | **Decision Tree** | A flowchart of yes/no splits ("is Size > 900?") ending in a predicted value |
| **RF** | **Random Forest** | Hundreds of decision trees, each on a random subset of rows and columns, averaged together (**bagging**) |
| **GB / GBR / GBM** | **Gradient Boosting (Regressor / Machine)** | Trees built one after another, each trained to correct the running error of all the previous ones (**boosting**). This is the deployed model |
| **alpha (α)** | — | Two different meanings in this project: Ridge's **regularisation strength**, and the **quantile level** (0.1, 0.9) for the prediction-interval models |
| **SHAP** | **SHapley Additive exPlanations** | A method for explaining individual predictions of a black-box model. Mentioned as an alternative to permutation importance; not implemented here |

---

## 5. Preprocessing, encoding and scaling

| Term | Full form / meaning | Where |
|---|---|---|
| **Imputation** | Filling in missing values. We impute `Size` with the median of the same (City, BHK) group, and `total_floors` with the city median |
| **OHE** | **One-Hot Encoding** — turning a category into one 0/1 column per level. `City` becomes six columns. Suitable only when the number of levels is small |
| **Target encoding** | Replacing a category with the average target value for that category. Used for `Area Locality`, which has 2,235 levels — one-hot would add 2,235 near-empty columns |
| **Smoothing** (empirical-Bayes) | Pulling a rare category's target encoding towards the global average, in proportion to how little data supports it. A locality with one listing gets almost the global average; one with 37 keeps most of its own |
| **Cross-fitting / out-of-fold encoding** | Computing each row's target encoding from *other* rows only, so a row never contributes to its own encoding. This is what stops target encoding from leaking |
| **Standardisation / StandardScaler** | Rescaling each numeric feature to mean 0, standard deviation 1. Needed by linear and polynomial models; irrelevant to trees, which only care about ordering |
| **Interaction term** | The product of two features, e.g. `Size × Area Type`. Lets a linear model express "a square foot means something different under each measurement convention". Trees discover this themselves |
| **log1p** | `log(1 + x)` — the natural logarithm, shifted so that `x = 0` is safe. Applied to the target to tame its skew of 21.4 |
| **expm1** | `e^x − 1` — the exact inverse of `log1p`. Applied to predictions so every reported number is in rupees |
| **Monotone transformation** | A transformation that preserves ordering. `log1p` is one, which is why `expm1(quantile of log rent)` is *exactly* the quantile of rent — no approximation in our prediction intervals |
| **NaN** | **Not a Number** — pandas' marker for a missing value. Notably, this dataset has **zero** explicit NaNs; its missing values hide inside the `Floor` text |
| **dtype** | **Data type** of a column (`int64`, `float64`, `object`/string) |

---

## 6. Validation and model selection

| Term | Full form / meaning |
|---|---|
| **Train / test split** | Holding back a portion of the data (20% here) that the model never sees, so you can measure honest performance |
| **Held-out test set** | That untouched portion. Ours is 950 listings, never outlier-treated and never used during tuning |
| **CV** | **Cross-Validation** — repeatedly splitting the training data into training and validation parts to estimate performance without touching the test set |
| **k-fold CV** | Split the training data into *k* equal parts; train on k−1 and validate on the remaining one; repeat k times and average. We use **k = 5** |
| **Fold** | One of those *k* parts |
| **Validation set** | The fold held out in a given CV round — used to tune, unlike the test set which is used only once at the end |
| **Stratified split** | Splitting so the proportion of each group is preserved. We stratify by `City` so all six cities keep their share in both splits |
| **Hyperparameter tuning** | Searching for the settings that perform best under cross-validation |
| **GridSearchCV** | Tries every combination in a grid. Exhaustive and slow |
| **RandomizedSearchCV** | Samples a fixed number of random combinations from the search space. Much cheaper, and what we use |
| **n_iter** | How many random combinations RandomizedSearchCV tries |
| **refit** | Which metric the search uses to pick its winner, and then retrains on. **We set `refit="mae"`** |
| **random_state / seed** | A fixed number making every random operation repeatable. Ours is **42** everywhere |
| **Reproducibility** | Re-running the code gives byte-identical results |
| **Permutation importance** | Shuffle one input column, measure how much worse the model gets. The drop is that column's importance. Measured on the held-out set, so it reflects real predictive value |
| **Impurity importance** | Tree-based importance from how much each split reduces error. Fast, but **biased towards high-cardinality features** like our target-encoded locality — which is why we report permutation importance alongside it |
| **Rank-sum** | Our selection method: rank each candidate on several criteria and add the ranks, so no single metric decides alone |

---

## 7. The two leakage guards in this project

Worth knowing by name, because both are graded points.

1. **`rent_per_sqft` is never a model feature.** It is `Rent ÷ carpet_equiv_size` —
   computed *from the target*. We use it for EDA plots and outlier detection only.
   Feeding it to a model would hand over the answer and produce a suspiciously
   perfect R².
2. **The target encoder lives inside the pipeline.** It is therefore refitted on
   each training fold during cross-validation and never sees the validation
   fold's rents. Fitting it once on the full training set before cross-validating
   would leak the answer into every CV score.

---

## 8. Prediction intervals

| Term | Full form / meaning |
|---|---|
| **Point estimate** | A single predicted number, e.g. ₹56,177 |
| **Prediction interval** | A range expected to contain the **actual value** for a new listing. Ours is 80%, from ₹23,049 to ₹1,36,881 in the screenshot |
| **Confidence interval** | Different thing — a range for an estimated *parameter* (like a mean), not for a single new observation. We report a prediction interval, which is wider and is the right tool here |
| **Quantile regression** | Predicting a chosen percentile of the target instead of its mean. Training at α = 0.1 and α = 0.9 gives the two ends of an 80% interval |
| **Quantile loss / pinball loss** | The asymmetric loss function that makes a model predict a quantile rather than a mean |
| **Empirical coverage** | How often the interval *actually* contained the true rent on the test set. Ours is **77.8%** against a nominal 80% — slightly narrow, and stated as such |
| **Nominal coverage** | The coverage the interval is *supposed* to have (80%) |

---

## 9. Dataset columns and real-estate terms

| Short | Full form | Meaning |
|---|---|---|
| **BHK** | **Bedroom, Hall, Kitchen** | The standard Indian way of describing flat size. "2 BHK" = 2 bedrooms plus a hall (living room) and a kitchen |
| **Carpet Area** | — | The usable floor area inside the walls — literally where you could lay carpet. The smallest of the three measures |
| **Built-up Area** | — | Carpet area **plus** the walls and internal ducts. Roughly 10–20% larger than carpet area |
| **Super Area** / Super Built-up Area | — | Built-up area **plus** your share of common space (lobbies, stairs, lifts). The largest, and the one builders prefer to advertise |
| **sq ft** | **square feet** | The unit of `Size` |
| **Area Locality** | — | Free-text neighbourhood name. 2,235 distinct values, 1,465 of which appear exactly once |
| **Furnishing Status** | — | Furnished / Semi-Furnished / Unfurnished |
| **Tenant Preferred** | — | Who the owner will rent to: Bachelors, Family, or `Bachelors/Family` |
| **Point of Contact** | — | `Contact Owner` / `Contact Agent` / `Contact Builder` |
| **Floor** | — | Stores *two* facts in one string, e.g. `"Ground out of 2"` = ground floor of a 2-storey building |

> **Important finding.** The conventional ratios (carpet ≈ 0.70 × super area) are
> **not supported by this dataset** — the observed ratio is ≈ 1.00 across 23
> comparable (City, BHK) groups, and `Built Area` has only 2 rows in the whole
> dataset so its ratio cannot be estimated at all.

---

## 10. Indian number and currency conventions

| Term | Meaning |
|---|---|
| **₹ / INR** | **Indian Rupee** (Unicode U+20B9). The project uses a custom formatter because Python's default `,` grouping is wrong for Indian numbers |
| **Lakh** | **1,00,000** — one hundred thousand |
| **Crore** | **1,00,00,000** — ten million (100 lakh) |
| **Indian digit grouping** | Groups of 2 after the first group of 3: `35,00,000`, not `3,500,000`. Implemented in `format_inr()` in `src/evaluate.py` |

---

## 11. Project-specific terms you will be asked about

| Term | What it means here |
|---|---|
| **`carpet_equiv_size`** | `Size` converted to a carpet-area equivalent so the three measurement conventions are comparable. Raw `Size` and `Area Type` are **kept as features too**, so the model can ignore the conversion if it is unhelpful — which the data suggests it is |
| **`floor_num` / `total_floors`** | The two numbers parsed out of the `Floor` string. `Ground → 0`, `Upper Basement → −0.5`, `Lower Basement → −1` |
| **`floor_ratio`** | `floor_num ÷ total_floors` — relative height within the building |
| **The four settings** | The outlier experiment runs every model under: (1) raw rent, (2) `log1p` only, (3a) `log1p` + outliers removed, (3b) `log1p` + outliers capped |
| **"Untreated test set"** | The test split is **never** outlier-treated. All four settings are scored against it, so removing outliers cannot flatter the metrics by making the test set easier. This is the single most important design decision in the experiment |
| **"Primary setting"** | The configuration the deployed model is trained under — here, `log1p` target only |
| **"Unseen locality"** | A locality in the test set that never appeared in training. Between 22% and 46% of test listings per city. Their locality effect falls back to the global average |
| **"Premium band"** | Test listings above ₹1,00,000/month. Outlier treatment damages these badly: MAE rises from ₹73,887 to ₹1,05,137 under removal |
| **Shortlist** | Models within 10% of the best test MAE, which then go through multi-criteria ranking |

> ### ⚠️ One ambiguity to be careful about
> **"CV" is used in two different senses** in this project:
> * **Cross-Validation** — almost everywhere (`cv_rmse_mean`, "5-fold CV")
> * **Coefficient of Variation** (`std ÷ mean`) — only in the model-selection
>   column `cv_mae_cv`, where it measures fold-to-fold stability
>
> If you are asked about `cv_mae_cv`, it reads as "the **c**oefficient of
> **v**ariation of the **c**ross-**v**alidated MAE".

---

## 12. scikit-learn classes used

| Class | What it does |
|---|---|
| `Pipeline` | Chains preprocessing steps and the model into one object, so all of them are fitted together inside each CV fold |
| `ColumnTransformer` | Applies different transformations to different columns (scaling to numerics, one-hot to categories, target encoding to locality) |
| `TransformedTargetRegressor` | Wraps a model so the **target** is transformed before fitting and inverted after predicting. This is what lets us train on `log1p(Rent)` while every reported metric stays in rupees |
| `OneHotEncoder(handle_unknown="ignore")` | One-hot encoding that outputs all-zeros for an unseen category instead of crashing. **Required** here: `Built Area` has 2 rows and `Contact Builder` has 1, so such a level can appear in the test split having never been in training |
| `TargetEncoder` | Target encoding with built-in cross-fitting and smoothing |
| `SimpleImputer` | Fills missing values with a median or most-frequent value |
| `StandardScaler` | Standardises numeric features |
| `PolynomialFeatures` | Generates the squared and interaction terms for polynomial regression |
| `KFold` | The 5-fold cross-validation splitter |
| `train_test_split` | Makes the single stratified 80/20 split |
| `RandomizedSearchCV` | Hyperparameter search |
| `permutation_importance` | Computes permutation importance |
| `BaseEstimator`, `TransformerMixin` | The base classes our custom `FeatureEngineer` inherits, so it behaves like any other sklearn step |

---

## 13. Files, formats and tooling

| Short | Full form | Meaning |
|---|---|---|
| **CSV** | **Comma-Separated Values** | The raw dataset format |
| **JSON** | **JavaScript Object Notation** | `metrics.json` and `feature_meta.json` — structured results read by the notebook, README and app |
| **MD** | **Markdown** | This file, and `README.md` |
| **IPYNB** | **IPython Notebook** | The `.ipynb` analysis notebook format |
| **PNG** | **Portable Network Graphics** | Every saved figure |
| **joblib** | — | The library used to save and load trained models. Faster than plain `pickle` for large NumPy arrays |
| **Pickle / serialisation** | — | Converting a Python object into bytes on disk, and back. Why scikit-learn must be **pinned exactly** — a different version may fail to unpickle the saved pipeline |
| **venv** | **Virtual environment** | An isolated per-project Python installation (`.venv/`) |
| **pip** | **Pip Installs Packages** | Python's package installer |
| **uv** | — | A fast drop-in replacement for pip/venv, used to build this project's environment |
| **PEP 668** | **Python Enhancement Proposal 668** | The rule that makes Homebrew's Python "externally managed" and refuses direct `pip install` — the reason a venv is mandatory here |
| **CLI** | **Command-Line Interface** | `python train.py --fast` |
| **UI / UX** | **User Interface / User Experience** | The Streamlit app |
| **API** | **Application Programming Interface** | How code talks to a library or service |
| **Agg** | **Anti-Grain Geometry** | matplotlib's non-interactive backend — lets plots be generated headlessly, with no display attached |
| **DPI** | **Dots Per Inch** | Figure resolution (130 here) |
| **Apache Arrow** | — | The columnar data format Streamlit uses to send DataFrames to the browser. Source of the "Expected bytes, got a float" error fixed in the app |
| **PID** | **Process ID** | The number identifying the running Streamlit server |
| **HTTP** | **HyperText Transfer Protocol** | How the browser talks to the Streamlit server |
| **URL** | **Uniform Resource Locator** | A web address |
| **LAN** | **Local Area Network** | Your home/college network — the `192.168.x.x` address |
| **IP** | **Internet Protocol** (address) | A machine's numeric network address |
| **NAT** | **Network Address Translation** | What your router does; the reason your public IP is usually not reachable from outside |
| **Port** | — | A numbered channel on a machine. Streamlit defaults to **8501** |
| **MB** | **Megabyte** | File sizes |

---

## 14. One-line answers worth memorising

* **Why `log1p`?** Rent has a skew of 21.41 and a maximum 219× the median. Squared-error regression on that target chases a handful of listings. `log1p` brings the skew to 0.91.
* **Why per-city outlier detection?** A single global IQR bound of ₹67,500 sits barely above Mumbai's ₹52,000 median — it would flag **37.8% of Mumbai** listings while flagging 3% in Hyderabad.
* **Why `TargetEncoder` and not one-hot for locality?** 2,235 levels, 1,465 appearing exactly once, in 4,746 rows.
* **Why tune on MAE and not RMSE?** CV RMSE's std is 60–83% of its own mean on this data — it is noise. CV MAE's is 10–15%.
* **Why is the test set never outlier-treated?** Otherwise removing outliers would improve the score simply by deleting the hard cases.
* **Does locality dominate?** No. All features R² 0.714 → without location 0.466 → location only 0.397. Strongest single block, but not sufficient alone.
