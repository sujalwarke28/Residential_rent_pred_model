# App Guide — Residential Rent Prediction

A top-to-bottom walkthrough of the app, in the exact order you see things on
screen, in plain English, with the formulas and charts behind each piece.

**How to read this document.** It follows the three tabs:

1. **[Estimate rent](#tab-1--estimate-rent)** — every input field, what it does, and how much it actually moves the answer
2. **[Model comparison & analysis](#tab-2--model-comparison--analysis)** — every table and chart on that tab, explained
3. **[Limitations](#tab-3--limitations)** — what the numbers do not tell you

Plus two reference sections at the end: **[the formulas](#appendix-a--every-formula-used)**
and **[how the whole thing fits together](#appendix-b--the-pipeline-end-to-end)**.

---

## The 30-second version

You type in a flat's details. The app sends them through a saved machine-learning
model that was trained on **4,746 real rental listings** from six Indian cities,
and it returns three things:

| Output | Example | Meaning |
|---|---|---|
| **Estimated monthly rent** | ₹56,177 | The single best guess |
| **Expected range** | ₹23,049 – ₹1,36,881 | An 80% prediction interval — the true rent should land inside this about 8 times out of 10 |
| **Rent per sq ft** | 66.1 | The estimate divided by the size, for sanity-checking |

The model's typical error is **₹10,135 (about 27% of actual rent)**. It is a
sanity-check tool, not a valuation.

---

# Tab 1 — Estimate rent

## 1.1 The sidebar (visible on every tab)

Before the inputs, the panel on the left is the model's **report card**. It never
changes — it describes the model, not your flat.

| Line | Value | Plain meaning |
|---|---|---|
| Selected model | **Gradient Boosting** | Which of the six candidate algorithms won |
| (big number) | **₹32,716** | **This is the model's RMSE, not a rent.** Its error on a typical-to-bad prediction |
| Test R² | **0.7144** | It explains about **71%** of why rents differ between flats |
| Test MAE | **₹10,135** | Its average miss, in rupees |
| Test MAPE | **26.7%** | That same miss as a percentage |
| Trained under... | log1p target, 3,796 listings, scored on 950 | How it was built and tested |
| scikit-learn 1.9.1 · random_state 42 | | Version and seed, so results are reproducible |

> ⚠️ **Easy to misread.** That large ₹32,716 is the model's *error*, not a price.
> It sits in the sidebar so you always see how much to trust the estimate on the
> right.

---

## 1.2 The input fields — all eleven, in screen order

Each field below shows a **sensitivity test**: I held one baseline flat fixed and
changed only that one field, so you can see exactly what it is worth to the model.

> **Baseline flat used throughout:**
> Bangalore · Electronic City · 900 sq ft Carpet Area · 2 BHK · 2 bathrooms ·
> floor 3 of 6 · Semi-Furnished · Bachelors/Family · Contact Owner
> → **₹16,887**

---

### ① City — *the single most important input*

**What it is:** one of the six cities in the dataset — Mumbai, Delhi, Bangalore,
Chennai, Hyderabad, Kolkata.

**Why it matters so much:** rent per square foot differs **5.4× between Mumbai
and Kolkata**. The same flat is simply a different product in a different city.

| City | Prediction | vs baseline | Median ₹/sq ft in data |
|---|---|---|---|
| **Mumbai** | ₹40,536 | **+140%** | 71.4 |
| Delhi | ₹17,896 | +6% | 28.6 |
| Bangalore *(baseline)* | ₹16,887 | — | 17.5 |
| Chennai | ₹16,221 | −4% | 16.2 |
| Hyderabad | ₹14,875 | −12% | 14.5 |
| Kolkata | ₹14,475 | −14% | 13.3 |

**Measured importance:** #1 by permutation importance — shuffling this column
costs **₹7,290** of accuracy, more than any other input.

![Rent vs size by city](reports/figures/eda_rent_vs_size.png)

*Left: every listing, rent against size, coloured by city — Mumbai (blue) sits
visibly above everything else. Right: the per-square-foot gap stated directly.*

---

### ② Area Locality — *neighbourhood*

**What it is:** the specific neighbourhood, filtered to the city you picked (604
options for Mumbai). Plus an **"Other / not listed"** choice.

**The problem this field has:** the dataset contains **2,235 distinct
localities**, and **1,465 of them appear exactly once**. Most neighbourhoods have
almost no evidence behind them.

**How the model handles it:** a technique called **target encoding**. Instead of
treating the locality name as a label, it replaces it with the *average rent of
listings in that locality*, pulled towards the city average depending on how
little data supports it:

$$
\text{encoded value} = \frac{n_L \times \bar{y}_L + m \times \bar{y}}{n_L + m}
$$

where $n_L$ = listings in that locality, $\bar{y}_L$ = their average rent,
$\bar{y}$ = the global average, and $m$ = a smoothing weight.

In plain terms: **a locality with 37 listings keeps most of its own character; a
locality with 1 listing is mostly replaced by the city average.** This stops the
model memorising individual flats.

| Bangalore locality | Prediction | vs baseline | Listings in data |
|---|---|---|---|
| Indira Nagar | ₹21,867 | +30% | 5 |
| Marathahalli | ₹20,523 | +22% | 5 |
| Koramangala | ₹20,026 | +19% | 9 |
| Whitefield | ₹18,201 | +8% | 2 |
| Electronic City *(baseline)* | ₹16,887 | — | 24 |
| **"Other / not listed"** | ₹18,201 | +8% | 0 |

**Measured importance:** #5 by permutation (**₹1,998**) — lower than you'd
expect, precisely *because* so many localities are too rare to carry signal.

> ⚠️ **If you pick "Other / not listed"**, the model has no neighbourhood
> information at all and falls back to the city average. The app shows a warning.
> Treat that estimate as much rougher than the range suggests.

---

### ③ Area Type — *how the size was measured*

**What it is:** which convention the square footage uses.

| Term | What it counts |
|---|---|
| **Carpet Area** | The usable floor inside your walls — literally where you could lay carpet. Smallest |
| **Built-up Area** | Carpet area **+** the walls and ducts. ~10–20% larger |
| **Super Area** | Built-up **+** your share of lobbies, stairs, lifts. Largest — what builders advertise |

**Why it matters:** 900 sq ft of *carpet* is a much bigger flat than 900 sq ft of
*super area*. The number means different things.

| Area Type | Prediction | vs baseline |
|---|---|---|
| Carpet Area *(baseline)* | ₹16,887 | — |
| Built Area | ₹15,878 | −6% |
| Super Area | ₹14,000 | **−17%** |

Exactly the right direction: tell the model your 900 sq ft is *super* area and it
correctly infers a smaller real flat, so a lower rent.

> 🔍 **A real finding from this project.** The textbook ratio says carpet ≈ 0.70 ×
> super area. In this dataset the **observed ratio is ≈ 1.00** across 23
> comparable (City, BHK) groups — meaning listings report whichever number they
> like, inconsistently. `Built Area` has only **2 rows in the entire dataset**, so
> its ratio could not be estimated at all. The conversion is still applied as
> specified, but raw `Size` and `Area Type` are kept as separate features so the
> model can route around a conversion the data does not support.

---

### ④ Size (sq ft) — *the second most important input*

**What it is:** floor area, measured under the Area Type above.

| Size | Prediction | vs baseline |
|---|---|---|
| 400 | ₹12,130 | −28% |
| 600 | ₹12,481 | −26% |
| 800 | ₹15,442 | −9% |
| **900** *(baseline)* | ₹16,887 | — |
| 1,200 | ₹17,559 | +4% |
| 1,600 | ₹21,555 | +28% |
| 2,400 | ₹26,438 | +57% |
| 4,000 | ₹31,571 | +87% |

**Notice it is not a straight line.** Doubling from 900 → 1,800 does *not* double
the rent. Rent rises with size but with **diminishing returns** — a classic
pattern that linear regression struggles with and tree models capture naturally.

**Measured importance:** #2 by permutation (**₹5,506**), and **#1** when measured
by R² drop — because for expensive flats, floor area is what separates a ₹2 lakh
flat from a ₹5 lakh one.

---

### ⑤ BHK — *Bedroom, Hall, Kitchen*

**What it is:** the standard Indian flat descriptor. "2 BHK" = 2 bedrooms + a
hall (living room) + a kitchen.

| BHK | Prediction | vs baseline |
|---|---|---|
| 1 | ₹16,711 | −1% |
| 2 *(baseline)* | ₹16,887 | — |
| 3 | ₹19,024 | +13% |
| 4 | ₹19,766 | +17% |
| 5–6 | ₹20,635 | +22% |

**Why the effect looks small:** because you already told the model the size. Once
it knows the flat is 900 sq ft, whether that's cut into 2 or 3 bedrooms matters
far less. BHK and Size carry overlapping information — this is called
**correlated features**, and it's why BHK ranks only #7 on permutation importance
(₹279). Its information has already been counted.

---

### ⑥ Bathrooms

| Bathrooms | Prediction | vs baseline |
|---|---|---|
| 1 | ₹14,911 | **−12%** |
| 2 *(baseline)* | ₹16,887 | — |
| 3 | ₹17,065 | +1% |
| 4–5 | ₹17,507 | +4% |

The jump from 1 → 2 bathrooms is worth more than every step after it. Going from
one bathroom to two is a real change in flat class; going from three to four is
not.

**Measured importance:** #3 by permutation (**₹2,804**) — higher than BHK,
because bathroom count is a sharper signal of how premium a unit is.

---

### ⑦ Floor number

`0` = ground floor, `−1` = lower basement, `−0.5` = upper basement.

| Floor (of 6) | Prediction | vs baseline |
|---|---|---|
| −1 (lower basement) | ₹16,807 | −0.5% |
| 0 (ground) | ₹16,625 | −2% |
| 1 | ₹16,869 | −0.1% |
| 3 *(baseline)* | ₹16,887 | — |
| 5 | ₹17,056 | +1% |
| 6 (top) | ₹17,225 | +2% |

**Almost no effect.** Floor is the weakest of all the meaningful inputs
(permutation importance ₹244, rank #8). In this dataset, which floor you're on
barely moves rent.

> 📌 This field exists because of a data-cleaning problem worth knowing about.
> The raw dataset stores floor as **text**: `"Ground out of 2"`, `"3 out of 5"`.
> That's **two facts crammed into one string**. The project splits it into
> `floor_num` and `total_floors`, mapping `Ground → 0`, `Upper Basement → −0.5`,
> `Lower Basement → −1`. Four listings have no "out of" part at all — those are
> the **hidden missing values**: the dataset has *zero* explicit blanks, but real
> missing data was buried inside the text.

---

### ⑧ Total floors in the building

| Total floors (you on floor 3) | Prediction | vs baseline |
|---|---|---|
| 3 | ₹15,100 | **−11%** |
| 6 *(baseline)* | ₹16,887 | — |
| 10 | ₹16,984 | +1% |
| 20 | ₹16,870 | −0.1% |
| 40 | ₹17,792 | +5% |

Mostly a proxy for building type — a 3-storey walk-up versus a 20-storey tower
with lifts. The model also computes a derived feature:

$$
\text{floor\_ratio} = \frac{\text{floor\_num}}{\text{total\_floors}}
$$

which captures "how high up in the building you are" independently of absolute
height.

---

### ⑨ Furnishing Status

| Status | Prediction | vs baseline |
|---|---|---|
| Unfurnished | ₹16,277 | −4% |
| Semi-Furnished *(baseline)* | ₹16,887 | — |
| **Furnished** | ₹19,043 | **+13%** |

Full furnishing is worth a real premium; the gap between unfurnished and
semi-furnished is small. Permutation importance ₹334 (rank #6).

---

### ⑩ Tenant Preferred

| Preference | Prediction | vs baseline |
|---|---|---|
| Bachelors | ₹16,928 | +0.2% |
| Bachelors/Family *(baseline)* | ₹16,887 | — |
| Family | ₹16,538 | −2% |

**The least important input in the entire model** — permutation importance ₹23,
dead last. It describes who the landlord wants, not what the property is worth.

---

### ⑪ Point of Contact — *the one to be careful with*

| Contact | Prediction | vs baseline |
|---|---|---|
| Contact Owner *(baseline)* | ₹16,887 | — |
| **Contact Agent** | ₹21,717 | **+29%** |
| Contact Builder | ₹17,244 | +2% |

**This is real, and it is in the data:**

| | Listings | Median rent |
|---|---|---|
| Contact Agent | 1,529 | **₹45,000** |
| Contact Owner | 3,216 | **₹12,500** |

A **3.6× gap**. And it survives controls — comparing like with like inside the
same (City, BHK) group, agent rent is higher in **18 of 18** comparable groups,
median ratio **1.62×**. Within-city rent per sq ft is 1.09×–1.71× higher.

**But read it correctly.** The arrow runs *backwards* from what the UI implies:

- ❌ "Using an agent makes my flat more expensive"
- ✅ "Expensive flats get listed through agents"

A ₹8,000 Kolkata room isn't worth an agent's commission; a ₹1,00,000 Bandra flat
is. It is also likely standing in for **features this dataset doesn't have** —
building age, amenities, security, parking — since agents tend to handle newer,
better-equipped properties. This is called a **proxy variable**: statistically
powerful, causally misleading.

> ⚠️ **Practical consequence.** Flipping only this dropdown raises a Mumbai
> estimate by **1.81×** for an identical flat. The model isn't wrong, but it's
> answering *"what market segment is this listing in?"* rather than *"what is
> this property worth?"*

---

## 1.3 The outputs

### Estimated monthly rent

The model's single best guess. How it is produced:

```
your inputs → feature engineering → encoding → Gradient Boosting → expm1() → ₹
```

Critically, the model was **not** trained to predict rupees directly. It was
trained to predict **log(1 + rent)**, then the result is converted back. Section
[A.2](#a2-the-log-transform--the-single-most-important-decision) explains why.

### Expected range (80% prediction interval)

Not a guess at how confident the model feels — it comes from **two additional
models** trained specifically to predict the 10th and 90th percentile of rent:

```
₹23,049 ─────────── ₹56,177 ─────────── ₹1,36,881
  10th percentile     estimate        90th percentile
          └──────── 80% of outcomes ────────┘
```

**Measured honestly:** on the 950 held-out test listings the true rent fell
inside this range **77.8% of the time**, against the advertised 80%. Slightly
narrow, and stated as such rather than claimed as exact.

The median interval is **74% of the actual rent wide** — a wide band, which is an
honest reflection of how much this data can actually tell you.

![Prediction interval coverage](reports/figures/prediction_interval_coverage.png)

### Rent per sq ft

Simply `estimate ÷ size`. A sanity check — compare it to the city medians in
field ① above. If your number is wildly off those, something in your inputs is
probably wrong.

### "Model behind this estimate"

Repeats the model's accuracy next to your result, and echoes back exactly what
was sent to the model — useful for catching a mis-typed input.

---

# Tab 2 — Model comparison & analysis

This tab is the **evidence** that the estimate on Tab 1 can be trusted. It
answers: why this model, how was it tested, and where does it fail?

---

## 2.1 Model comparison

Six algorithms were trained and compared. All numbers are on the **950 held-out
listings** the models never saw during training or tuning.

| Model | CV RMSE (mean ± std) | Test R² | Test MSE | Test RMSE | Test MAE | Test MAPE |
|---|---|---|---|---|---|---|
| Linear Regression | 53,464 ± 38,571 | **0.792** | 7.78 × 10⁸ | **27,897** | 10,760 | 28.4% |
| **Gradient Boosting** ← selected | 49,039 ± 39,836 | 0.714 | 1.07 × 10⁹ | 32,716 | **10,135** | **26.7%** |
| Decision Tree | 49,757 ± 38,656 | 0.677 | 1.21 × 10⁹ | 34,818 | 11,640 | 32.0% |
| Random Forest | 49,625 ± 39,816 | 0.671 | 1.23 × 10⁹ | 35,092 | 10,245 | 26.8% |
| Polynomial (deg 2) + Ridge | 55,364 ± 37,036 | −4.49 | 2.06 × 10¹⁰ | 143,423 | 15,257 | 29.6% |
| Polynomial (deg 2) | 58,573 ± 35,070 | **−223.59** | 8.42 × 10¹¹ | 917,495 | 42,321 | 43.0% |

![Model RMSE comparison](reports/figures/model_rmse_comparison.png)
![All metrics panel](reports/figures/model_metric_panel.png)

### What the six models are, in one line each

| Model | How it works |
|---|---|
| **Linear Regression** | Fits one straight-line equation: rent = a weighted sum of the features |
| **Polynomial (deg 2)** | Same, but after adding every squared term and every pairwise product, letting the fit bend. Our ~24 features become ~324 terms |
| **Polynomial + Ridge** | Same expansion, but with a penalty that shrinks the coefficients and stops them exploding |
| **Decision Tree** | A flowchart of yes/no questions ("is Size > 900?") ending in a predicted rent |
| **Random Forest** | Hundreds of trees, each on a random slice of the data, averaged (**bagging**) |
| **Gradient Boosting** | Trees built one after another, each correcting the previous ones' mistakes (**boosting**) |

### Why the polynomial models exploded

R² of **−224** means the model is 224× worse than just guessing the average every
time. This is **not a bug — it's the lesson.**

Expanding 24 features to degree 2 creates ~324 terms that are heavily correlated
with each other (**multicollinearity**). The fitting procedure compensates with
enormous coefficients. On a test flat slightly outside the training range, those
coefficients produce a wild number in log-space — and because the final step is
`expm1`, a log prediction of 20 becomes ₹48 crore.

Ridge shrinks the coefficients and improves things by a factor of ~6 (R² −224 →
−4.5), which is exactly the point the case study asks to demonstrate: **why
regularisation is needed**. The search range was deliberately wide (α from 0.01
to 10,000,000) so this is not an artefact of under-tuning.

### The selected model's settings

Gradient Boosting, tuned by random search over 30 configurations × 5 folds:

| Hyperparameter | Chosen | Plain meaning |
|---|---|---|
| `n_estimators` | 535 | Build 535 trees in sequence |
| `learning_rate` | 0.0121 | Each tree corrects only ~1.2% of the remaining error — slow and careful |
| `max_depth` | 5 | Each tree asks at most 5 nested questions |
| `min_samples_leaf` | 6 | Never make a decision based on fewer than 6 listings |
| `subsample` | 0.755 | Each tree sees a random 75% of the data, which reduces overfitting |

Single prediction takes **6.5 ms**.

### Why Gradient Boosting was selected when Linear Regression has better RMSE

This is the most important subtlety on the tab.

| | Linear Regression | Gradient Boosting |
|---|---|---|
| Test RMSE | **₹27,897** ✅ | ₹32,716 |
| Test R² | **0.792** ✅ | 0.714 |
| Test MAE | ₹10,760 | **₹10,135** ✅ |
| Test MAPE | 28.4% | **26.7%** ✅ |

They win on different metrics. **The reason is extrapolation.**

A linear model fits a trend line and keeps extending it forever, so it can
predict a rent higher than anything it was trained on. **A tree model
structurally cannot** — its predictions are averages of training examples, so it
can never output more than the most expensive thing it ever saw.

The test set contains a ₹12,00,000 Mumbai flat. Linear Regression extrapolates
towards it; Gradient Boosting is capped and badly under-predicts it. Because RMSE
*squares* errors, that one listing dominates the whole metric:

| Worst *k* of 950 test rows | Share of total MSE |
|---|---|
| 1 | **42%** |
| 2 | 51% |
| 5 | 66% |
| 10 | **77%** |
| 50 | 94% |

**RMSE over all rows: ₹32,716. Excluding just the worst 1% of rows: ₹16,234.**

So test RMSE is largely measuring how a model handles a handful of mansions, not
how accurate it is on ordinary flats. Since the app estimates rent for ordinary
flats, the model with the better *typical* error was selected — and the trade-off
is documented rather than hidden.

> This also drove a methodology decision: the **5-fold CV RMSE has a standard
> deviation of 60–83% of its own mean** — it is essentially noise. CV MAE is
> 10–15%. So hyperparameters were tuned on **MAE**, while all four metrics are
> still reported.

---

## 2.2 The outlier experiment

**The question:** a handful of ₹5,00,000+ listings sit among thousands of
₹10,000 ones. Should they be removed before training?

Every model was trained under **four settings** and all were scored on the **same
untreated test set**:

| Setting | Target | Outlier handling |
|---|---|---|
| 1 | Raw rent (₹) | None |
| 2 | `log1p(rent)` | None ← **selected** |
| 3a | `log1p(rent)` | Outlier rows **deleted** |
| 3b | `log1p(rent)` | Extreme rents **capped** at each city's 1st/99th percentile |

> 🔑 **The critical design rule.** Outlier treatment is applied to the **training
> data only**. If you also removed outliers from the test set, your score would
> improve simply because you deleted the hard questions from the exam. Scoring
> every setting against one untreated test set is what makes the comparison
> honest.

![Outlier experiment](reports/figures/outlier_experiment.png)

### How outliers are detected — per city, not globally

The standard rule uses the **Interquartile Range**:

$$
\text{IQR} = Q_3 - Q_1 \qquad
\text{outlier if } y < Q_1 - 1.5\,\text{IQR} \ \text{ or } \ y > Q_3 + 1.5\,\text{IQR}
$$

Applied **globally**, this gives an upper bound of ₹67,500 — which is barely above
**Mumbai's median rent of ₹52,000**. It would throw away over a third of Mumbai:

| City | Flagged, per-city rule | Flagged, global rule |
|---|---|---|
| Mumbai | 10.7% | **37.8%** ❌ |
| Hyderabad | 9.8% | 3.0% |
| Kolkata | 6.9% | 3.1% |

₹1.5 lakh is genuinely bizarre in Kolkata and completely ordinary in South
Mumbai. So bounds are computed **inside each city**.

![Per-city vs global outliers](reports/figures/eda_outlier_per_city_vs_global.png)

### The result — the key finding of the whole project

**MAE by rent band** for Gradient Boosting (lower is better):

| Actual rent band | Raw target | log1p only | log1p + removed | log1p + capped |
|---|---|---|---|---|
| < ₹15,000 | 3,539 | 2,528 | **2,482** | 2,524 |
| ₹15,000 – ₹40,000 | 5,843 | 5,345 | **5,067** | 5,279 |
| ₹40,000 – ₹1,00,000 | 18,036 | 14,798 | 17,360 | **14,531** |
| **> ₹1,00,000 (premium)** | **70,554** | **73,887** | **1,05,137** ❌ | 79,468 |

![Premium trade-off](reports/figures/outlier_premium_tradeoff.png)

**Read the bottom row.** Deleting outliers improves cheap flats by a trivial
₹46 and makes premium flats worse by **₹31,250 — a 42% deterioration**.

The mechanism is simple: a model trained with expensive flats deleted has never
seen one, so it drags them down towards the ordinary market.

### The trap this experiment exposes

Under the "outliers removed" setting, the **cross-validated** RMSE looks
spectacular — ₹10,142 versus ₹48,196 for the untreated version. A five-fold
improvement!

It is completely fake. Those CV folds no longer *contain* the hard rows. The
model didn't get better; the exam got easier. Only the test scores — all measured
on the same untreated set — are comparable.

**Conclusion:** the case study's premise is confirmed (a few premium properties
*do* distort the regression badly on a raw target), **but the correct remedy is
the log transform, not deletion.** Deleting the evidence trades an error you can
measure for one you cannot.

---

## 2.3 Per-city performance

The same model, scored separately in each city.

| City | Test n | Mean rent | RMSE | MAE | R² | MAPE | **RMSE as % of mean** | Unseen localities |
|---|---|---|---|---|---|---|---|---|
| Chennai | 178 | 21,614 | 13,146 | 5,894 | 0.794 | 26.4% | **61%** ✅ | 25.3% |
| Bangalore | 177 | 19,382 | 12,748 | 5,357 | 0.664 | 24.2% | 66% | 36.2% |
| Hyderabad | 174 | 19,846 | 13,369 | 4,967 | 0.688 | 20.7% | 67% | 22.4% |
| Mumbai | 195 | 85,782 | 63,564 | 25,789 | 0.664 | 32.7% | 74% | 45.6% |
| Delhi | 121 | 28,708 | 29,721 | 10,100 | 0.390 | 30.3% | 104% | 31.4% |
| **Kolkata** | 105 | 13,676 | 17,302 | 4,911 | **0.102** | 26.6% | **127%** ❌ | 35.2% |

![Per-city performance](reports/figures/per_city_performance.png)

**Why "RMSE as % of mean" is the column that matters:** Mumbai's RMSE of ₹63,564
looks catastrophic next to Kolkata's ₹17,302 — but Mumbai rents are 6× higher.
Relative to its own market, Mumbai is the *better* prediction.

**Why Kolkata is worst (R² = 0.102):**
1. **Fewest listings** — only 105 in test, the smallest sample
2. **Lowest rents** — small absolute errors become large percentages
3. **35% unseen localities** — a third of Kolkata test flats are in neighbourhoods the model never saw

**Practical implication:** a deployed tool should show a *wider* band in cities
and localities where coverage is thin, instead of one uniformly confident number.

---

## 2.4 Feature importance

Two different measures, which **disagree** — and the disagreement is the finding.

| Feature | MAE rise when shuffled | R² drop when shuffled |
|---|---|---|
| **City** | **₹7,290** | 0.340 |
| **Size** | ₹5,506 | **0.330** |
| Bathroom | ₹2,804 | 0.219 |
| Point of Contact | ₹2,775 | 0.196 |
| Area Locality | ₹1,998 | 0.091 |
| Furnishing Status | ₹334 | 0.002 |
| BHK | ₹279 | 0.004 |
| Floor | ₹244 | −0.010 |
| Area Type | ₹146 | 0.007 |
| Tenant Preferred | ₹23 | −0.001 |

![Feature importance](reports/figures/feature_importance.png)

### How permutation importance works

Take the trained model. Randomly shuffle one column — so that column now contains
real values but attached to the wrong flats. Re-measure accuracy. **However much
worse it got is that column's importance.**

It is reliable because it measures what the column is *actually worth* on data
the model has never seen.

### Why two scorers, and why they disagree

- **R² drop** is dominated by expensive listings (it squares errors), so `Size`
  wins — for a ₹3 lakh flat, floor area is what sets the price.
- **MAE rise** weights every listing equally, so `City` wins — for an ordinary
  ₹15,000 flat, which city you're in matters more than a few hundred sq ft.

Both are correct. They answer different questions.

### The impurity-importance trap

The tree's own internal importance scores tell a *different* story again:

| Feature (internal) | Impurity importance |
|---|---|
| `Point of Contact_Contact Agent` | **0.334** ← #1 |
| `Bathroom` | 0.216 |
| `City_Mumbai` | 0.113 |
| `Size` | 0.080 |

**Why is `Contact Agent` #1 here but only #4 on permutation?**

Impurity importance credits whichever feature *makes* a split, weighted by how
much error it removed. `Contact Agent` is a clean binary split separating
₹45,000 flats from ₹12,500 ones — the single best split available at the **root**
of a tree, across all rows. It banks an enormous share before any other feature
gets a turn.

Permutation importance asks the harder question: *if I destroy this column, how
much worse does the model actually get?* Not much — because `City`, `Size` and
`Bathroom` are correlated with it and recover most of the signal.

> **The rule:** impurity importance gives all the credit to whichever correlated
> feature splits first. Permutation importance discounts redundancy. **Trust
> permutation importance.**

---

## 2.5 Does locality dominate? (the ablation study)

An **ablation study** means deliberately removing a group of features and
re-measuring, to find out how much they were really contributing.

| Model | Features | Test RMSE | Test MAE | Test R² |
|---|---|---|---|---|
| **A** | Everything | ₹32,716 | ₹10,135 | **0.714** |
| **B** | **Without** Locality and City | ₹44,728 | ₹13,156 | 0.466 |
| **C** | **Only** Locality and City | ₹47,537 | ₹16,514 | 0.397 |

![Locality ablation](reports/figures/locality_ablation.png)

**The answer: no, locality does not dominate.**

- Remove location → RMSE worsens by **37%**, R² falls 0.714 → 0.466. Location is
  clearly the strongest single block.
- But location *alone* reaches only R² **0.397** — about **56%** of the full
  model's explanatory power.

**Knowing where a flat is tells you more than any other single fact about it —
but you still need to know how big it is. Neither block is sufficient alone.**

---

## 2.6 Residual diagnostics

A **residual** is one prediction's error:

$$\text{residual} = y_{\text{actual}} - y_{\text{predicted}}$$

Positive = the model under-predicted. These plots check whether the errors are
*well behaved* — random scatter is good; patterns mean something is being missed.

![Residual diagnostics](reports/figures/best_model_residual_diagnostics.png)

| Diagnostic | Value | Reading |
|---|---|---|
| Mean residual | ₹2,905 | Close to zero — **no systematic bias** ✅ |
| Median residual | −₹38 | Essentially zero for the typical flat ✅ |
| Skew | 9.71 | Heavy right tail — a few huge under-predictions |
| Kurtosis | 185.1 | Extreme errors far more common than a bell curve predicts |
| corr(prediction, abs error) | 0.642 | **Clear funnel** — error grows with rent |
| Within ±10% of actual | 25.5% | |
| Within ±25% of actual | 59.9% | |

**The funnel shape is expected, not a defect.** The model was trained on
`log1p(rent)`, so it controls *percentage* error. A constant 25% error is ₹2,500
on a ₹10,000 flat and ₹75,000 on a ₹3,00,000 flat. The percentage-error panel
(bottom-right of the figure) is the honest view.

**The Q-Q plot** compares your errors against a perfect bell curve. Points on the
straight line = normally distributed. Ours bends away sharply at both ends,
confirming the extreme errors are not normal — which is exactly why MAE and MAPE
are reported alongside RMSE.

### Residuals by city

![Residuals by city](reports/figures/best_model_residuals_by_city.png)

### The ten worst predictions

| Flat | Actual | Predicted | Error | Why it failed |
|---|---|---|---|---|
| Mumbai / Juhu, 4 BHK, 5,000 sq ft | ₹12,00,000 | ₹5,44,973 | −55% | Premium listing far above the local distribution; trees cannot extrapolate |
| Delhi / Green Park, 4 BHK, 800 sq ft | ₹3,50,000 | ₹54,298 | **−84%** | Locality appears **once** in training — target encoding smoothed it to the city average |
| Mumbai / Lake Superior Powai, 4 BHK | ₹2,30,000 | ₹4,97,219 | +116% | Locality **unseen** in training — fell back to the global average |

Every one of the top failures is a **premium property** or a **data-poor
locality**. That is a consistent, explainable failure mode, not random noise.

### All models compared on residuals

![Residuals for all models](reports/figures/residuals_all_models.png)

Note the **Decision Tree** panel — its points form vertical stripes. A single
tree can only output one value per leaf, so predictions cluster onto a few
discrete levels. Random Forest and Gradient Boosting average many trees and
produce a smooth spread.

---

# Tab 3 — Limitations

Everything above says what the model *can* do. This tab says what it cannot — and
reading it is the difference between using the tool and misusing it.

## The ten limitations, ranked by how much they should worry you

| # | Limitation | The number behind it |
|---|---|---|
| 1 | **Missing features** — no amenities, building age, parking, metro proximity, security, society quality | This is most of the unexplained ~29% of variance |
| 2 | **Premium properties are unreliable** | MAE in the >₹1 lakh band is ₹73,887 |
| 3 | **Thin locality coverage** | 2,235 localities, **1,465 appear exactly once** (31% of rows) |
| 4 | **Unseen localities fall back to a global average** — and the interval does **not** widen to warn you | 22–46% of test listings per city |
| 5 | **Asking price ≠ negotiated rent** | The target is what landlords *advertise* |
| 6 | **A 89-day snapshot** (13 Apr – 11 Jul 2022) | No inflation, no seasonality; rupee levels are already stale |
| 7 | **Only six cities** | Nothing transfers to Pune, Ahmedabad or any tier-2 city |
| 8 | **Small dataset** | 4,746 listings, 3,796 used for training |
| 9 | **Accuracy varies sharply by city** | R² from 0.794 (Chennai) down to **0.102 (Kolkata)** |
| 10 | **`Area Type` is reported inconsistently** | Observed super-to-carpet ratio ≈ 1.00, not the conventional 0.70 |

## Real-world applicability

**Who would use this:** landlords sanity-checking an asking price; tenants judging
whether a listing is fair; rental platforms pre-filling a suggested rent or
flagging suspicious listings.

**How it would be integrated:** as a suggested-rent field in a listing form —
showing the **range**, always overridable, labelled as an estimate from
comparable listings. Never a single authoritative number.

**Retraining cadence:** monthly ideally, quarterly at minimum. A full retrain
takes ~8 minutes, so cost is not the constraint. Trigger on schedule *and* on
drift detection.

**What data would help most, in order:**
1. **Geocoordinates instead of locality names** — would fix the unseen-locality problem entirely, since an unknown point still has neighbours
2. **Amenities and building age** — the largest block of missing signal
3. **Actual transacted rent** instead of asking price — removes a systematic bias
4. **Time-to-let** as a demand signal
5. **More history** so seasonality becomes learnable

**What could go wrong:**
- **Anchoring loop** — if landlords price to the model's suggestion, next year's training data reflects the model's own predictions and it ends up measuring itself
- **Entrenched inequity** — thin-data localities get pulled to the average, so systematic under-estimation in underserved areas could suppress prices there while aggregate accuracy looks fine
- **False precision** — a single rupee figure implies confidence the model doesn't have
- **Premium-segment failure** — the segment with the most money at stake is predicted worst
- **Silent staleness** — nothing in a deployed model announces its training data is years old

---

# Appendix A — Every formula used

## A.1 The four evaluation metrics

With `y` = actual rent, `ŷ` = predicted rent, `n` = number of listings,
`ȳ` = mean actual rent:

**Mean Squared Error** — average squared miss. Punishes big errors hard.
$$\text{MSE} = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2$$

**Root Mean Squared Error** — the same, square-rooted back into rupees.
$$\text{RMSE} = \sqrt{\text{MSE}} = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2} = ₹32{,}716$$

**Mean Absolute Error** — average unsigned miss. Every listing counts equally.
$$\text{MAE} = \frac{1}{n}\sum_{i=1}^{n}|y_i - \hat{y}_i| = ₹10{,}135$$

**Mean Absolute Percentage Error** — the same as a percentage.
$$\text{MAPE} = \frac{100}{n}\sum_{i=1}^{n}\left|\frac{y_i - \hat{y}_i}{y_i}\right| = 26.7\%$$

**R² (Coefficient of Determination)** — fraction of variation explained.
$$R^2 = 1 - \frac{\sum(y_i - \hat{y}_i)^2}{\sum(y_i - \bar{y})^2} = 0.7144$$

R² = 1 is perfect, 0 means no better than always guessing the average, and
**negative means worse than that** — which is how the plain polynomial model
reached −223.59.

### Why RMSE ≠ MAE, intuitively

Two models, both off by ₹20,000 total across 2 flats:

| | Flat A error | Flat B error | MAE | RMSE |
|---|---|---|---|---|
| Model 1 | ₹10,000 | ₹10,000 | ₹10,000 | ₹10,000 |
| Model 2 | ₹0 | ₹20,000 | ₹10,000 | **₹14,142** |

Same MAE. RMSE punishes Model 2 for concentrating its error. **That is why RMSE
on this dataset is decided by a handful of mansions.**

---

## A.2 The log transform — the single most important decision

Raw rent is extremely lopsided:

| | Value |
|---|---|
| Median | ₹16,000 |
| Mean | ₹34,993 |
| Maximum | **₹35,00,000** — 219× the median |
| **Skewness** | **21.41** |

Skewness measures lopsidedness; 0 is symmetric. 21.41 is extreme. Because
regression minimises *squared* error, that one ₹35 lakh flat counts as much as
thousands of ordinary ones, and the model learns to chase it.

**The fix:**
$$y' = \log(1 + y) \qquad \text{(log1p)}$$

which compresses the big values and brings skewness to **0.91** — nearly
symmetric.

![Rent distribution before and after log](reports/figures/eda_rent_distribution.png)

Predictions are converted back with the exact inverse:
$$\hat{y} = e^{y'} - 1 \qquad \text{(expm1)}$$

**So every number the app shows you is in rupees, never in log units.** An RMSE
of "0.34 log-rupees" would mean nothing to a landlord.

### A bonus property that makes the prediction interval exact

`log1p` is a **monotone** transformation — it never changes the ordering of
values. Therefore the 10th percentile of log-rent, converted back, is *exactly*
the 10th percentile of rent:

$$\text{expm1}\big(q_{0.1}(\log\text{-rent})\big) = q_{0.1}(\text{rent})$$

No approximation. The 80% interval is mathematically correct, not an estimate.

---

## A.3 Outlier detection — the IQR rule

$$Q_1 = \text{25th percentile},\quad Q_3 = \text{75th percentile},\quad \text{IQR} = Q_3 - Q_1$$

$$\text{lower bound} = Q_1 - 1.5 \times \text{IQR}, \qquad \text{upper bound} = Q_3 + 1.5 \times \text{IQR}$$

**Computed per city**, for the reason shown in §2.2.

A second, independent check flags listings whose rent is implausible *for their
size*:
$$\text{rent\_per\_sqft} = \frac{\text{Rent}}{\text{carpet\_equiv\_size}}$$

> ⚠️ **`rent_per_sqft` is never a model input.** It is calculated *from the
> target*, so feeding it to the model would hand over the answer — a mistake
> called **data leakage**, which produces a suspiciously perfect R² that
> collapses in production. It is used for exploration and outlier detection only.

**Capping (winsorizing)** keeps every row but pulls extremes inwards:
$$y_{\text{capped}} = \min\big(\max(y,\; P_{1}^{\text{city}}),\; P_{99}^{\text{city}}\big)$$

---

## A.4 Feature engineering

**Carpet-equivalent size** — makes the three measurement conventions comparable:
$$\text{carpet\_equiv\_size} = \text{Size} \times r, \qquad
r = \begin{cases} 1.00 & \text{Carpet Area}\\ 0.85 & \text{Built Area}\\ 0.70 & \text{Super Area}\end{cases}$$

**Floor features** — parsed out of the text `"3 out of 5"`:
$$\text{floor\_ratio} = \frac{\text{floor\_num}}{\text{total\_floors}}$$
with `Ground → 0`, `Upper Basement → −0.5`, `Lower Basement → −1`.

**Size imputation** — 16 listings report under 50 sq ft (smallest: 10). Treated
as missing and replaced:
$$\text{Size} \leftarrow \text{median}\big(\text{Size} \mid \text{same City, same BHK}\big)$$

**Standardisation** — for linear and polynomial models only (trees don't care):
$$z = \frac{x - \mu}{\sigma}$$

**Target encoding with smoothing** — for the 2,235 localities:
$$\text{enc}(L) = \frac{n_L \,\bar{y}_L + m\,\bar{y}}{n_L + m}$$

---

## A.5 How Gradient Boosting actually works

Start with one constant guess, then add 535 small corrections:

$$F_0(x) = \text{a single starting value}$$
$$F_m(x) = F_{m-1}(x) + \nu \cdot h_m(x)$$

where $h_m$ is a small tree trained on the **errors** the model has made so far,
and $\nu = 0.0121$ is the learning rate.

In plain English:

```
Guess ₹20,000 for everyone.
  → Look at the errors. Build a small tree that explains them.
  → Nudge every prediction 1.2% in that direction.
  → Look at the new errors. Build another tree.
  → ... repeat 535 times.
```

A small learning rate means each tree is only allowed a tiny correction, so the
model improves cautiously instead of overreacting to noise. **This is why it
cannot extrapolate** — every prediction is built from averages of training
examples, so it can never exceed the most expensive flat it ever saw.

---

## A.6 Prediction intervals — the quantile loss

Normal regression predicts the *average*. Quantile regression predicts a chosen
*percentile*, using an asymmetric penalty:

$$
L_\alpha(y, \hat{y}) = \begin{cases}
\alpha\,(y - \hat{y}) & \text{if } y \geq \hat{y} \quad \text{(under-predicted)}\\[4pt]
(1-\alpha)\,(\hat{y} - y) & \text{if } y < \hat{y} \quad \text{(over-predicted)}
\end{cases}
$$

With **α = 0.9**, under-predicting is penalised 9× more than over-predicting, so
the model learns to sit high — at the 90th percentile. With **α = 0.1** the
reverse, giving the 10th percentile. Together: an **80% interval**.

---

## A.7 Ridge regularisation

Ordinary regression minimises error alone. Ridge adds a penalty on coefficient
size:

$$\min_{\beta} \ \underbrace{\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}_{\text{fit the data}} \ + \ \underbrace{\alpha\sum_{j=1}^{p}\beta_j^2}_{\text{keep coefficients small}}$$

Larger α = more shrinkage = a simpler, more stable model. This is what stopped
the polynomial model from producing ₹48 crore predictions — improving its R² from
−223.59 to −4.49.

---

# Appendix B — The pipeline end to end

## B.1 What happens when you press "Estimate rent"

```
┌──────────────────────────────────────────────────────────────────┐
│  YOUR 11 INPUTS                                                  │
│  City · Locality · Area Type · Size · BHK · Bathrooms            │
│  Floor · Total floors · Furnishing · Tenant · Point of Contact   │
└────────────────────────────┬─────────────────────────────────────┘
                             ▼
┌──────────────────────────────────────────────────────────────────┐
│  STEP 1 — FeatureEngineer                                        │
│  • parse Floor text → floor_num, total_floors                    │
│  • fill implausible Size from (City, BHK) medians                │
│  • compute floor_ratio and carpet_equiv_size                     │
│  • build Size × Area Type interaction terms                      │
└────────────────────────────┬─────────────────────────────────────┘
                             ▼
┌──────────────────────────────────────────────────────────────────┐
│  STEP 2 — ColumnTransformer (different treatment per column)      │
│  numbers    → median fill (+ scaling for linear models)          │
│  categories → one-hot encode, handle_unknown="ignore"            │
│  locality   → target encode with smoothing                       │
└────────────────────────────┬─────────────────────────────────────┘
                             ▼
┌──────────────────────────────────────────────────────────────────┐
│  STEP 3 — Gradient Boosting (535 trees, depth 5, lr 0.0121)       │
│  outputs a prediction in LOG space                               │
└────────────────────────────┬─────────────────────────────────────┘
                             ▼
┌──────────────────────────────────────────────────────────────────┐
│  STEP 4 — expm1()  →  back into rupees                           │
└────────────────────────────┬─────────────────────────────────────┘
                             ▼
              ₹56,177   +   ₹23,049 – ₹1,36,881
            (estimate)      (from 2 quantile models)
```

**Why this matters:** all four steps live inside **one saved file**
(`models/pipeline.joblib`). The app does **no preprocessing of its own** — it
just builds a table with the right column names and calls `predict`. That
guarantees the app treats your inputs exactly the way training treated them,
eliminating a whole class of bug called *training/serving skew*.

## B.2 How the model was validated

```
                   4,746 listings
                         │
         split once, stratified by city
                         │
         ┌───────────────┴────────────────┐
         ▼                                ▼
   3,796 TRAINING                    950 TEST
         │                      ╔═══════════════════════╗
   split into 5 folds           ║ LOCKED AWAY           ║
         │                      ║ never outlier-treated ║
   ┌──┬──┬──┬──┬──┐             ║ never seen in tuning  ║
   │1 │2 │3 │4 │5 │             ║ used once, at the end ║
   └──┴──┴──┴──┴──┘             ╚═══════════════════════╝
   train on 4, validate on 1
   rotate 5 times, average
         │
   pick best hyperparameters
   (ranked on MAE — CV RMSE is too noisy)
```

Every fitted quantity — the group medians, the one-hot categories, and
critically the **target encoding** — is recomputed inside each fold. If the
target encoder were fitted once on all the training data before
cross-validating, it would leak each fold's answers into its own validation
score. Keeping it inside the pipeline is what prevents that.

## B.3 Reproducibility

`RANDOM_STATE = 42` is threaded through the split, every random search, every CV
splitter, the target encoder's internal cross-fitting, and every model. scikit-learn
is pinned to exactly 1.9.1. Re-running `python train.py` reproduces every number
in this document.

---

## Where everything lives

| What | Where |
|---|---|
| All the charts | `reports/figures/` |
| All the tables as CSV | `reports/*.csv` |
| Every number in one file | `models/metrics.json` |
| The trained model | `models/pipeline.joblib` |
| The full narrative analysis | `notebooks/rent_prediction_analysis.ipynb` |
| Term definitions | `GLOSSARY.md` |
| Project overview and results | `README.md` |
