"""Streamlit app: residential rent estimation.

The app performs NO preprocessing of its own. It builds a one-row DataFrame with
the correct column names and hands it to the saved pipeline, which contains the
feature engineering, imputation, encoding and the model. That is the whole point
of keeping every transformation inside the sklearn Pipeline.

Run with:
    streamlit run app/app.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# The saved pipeline references `src.preprocess.FeatureEngineer`, so the project
# root must be importable before joblib.load is called. This is the single most
# common reason a Streamlit Community Cloud deployment fails with
# ModuleNotFoundError while working fine locally.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import pandas as pd
import streamlit as st

from src import config as cfg
from src.evaluate import format_inr

st.set_page_config(page_title="Residential Rent Prediction",
                   page_icon="house", layout="wide")

OTHER_LOCALITY = "Other / not listed"


# --------------------------------------------------------------------------- #
# Loading (cached so the model is deserialised once per session)
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Loading the trained pipeline...")
def load_pipeline():
    if not cfg.PIPELINE_PATH.exists():
        return None
    return joblib.load(cfg.PIPELINE_PATH)


@st.cache_resource(show_spinner=False)
def load_quantile_models():
    """The 10th/90th percentile models that form the 80% prediction interval."""
    try:
        if cfg.QUANTILE_LOW_PATH.exists() and cfg.QUANTILE_HIGH_PATH.exists():
            return joblib.load(cfg.QUANTILE_LOW_PATH), joblib.load(cfg.QUANTILE_HIGH_PATH)
    except Exception:
        pass
    return None, None


@st.cache_data(show_spinner=False)
def load_json(path: Path):
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def missing_artifacts_screen() -> None:
    st.error("Trained model artifacts were not found in `models/`.")
    st.markdown(
        "Train the model first, from the project root:\n\n"
        "```bash\n"
        "python -m venv .venv && source .venv/bin/activate\n"
        "pip install -r requirements.txt\n"
        "python train.py\n"
        "```\n\n"
        "That writes `models/pipeline.joblib`, `models/metrics.json` and "
        "`models/feature_meta.json`, which this app reads."
    )


# --------------------------------------------------------------------------- #
# Tab 1 -- prediction
# --------------------------------------------------------------------------- #
def render_predict_tab(pipeline, meta: dict, metrics: dict) -> None:
    best = metrics.get("best_model", {})
    opts = meta["categorical_options"]
    ranges = meta["numeric_ranges"]

    st.subheader("Property details")
    c1, c2, c3 = st.columns(3)

    with c1:
        city = st.selectbox("City", meta["cities"],
                            index=meta["cities"].index("Mumbai")
                            if "Mumbai" in meta["cities"] else 0)
        localities = meta["localities_by_city"].get(city, [])
        locality = st.selectbox(
            f"Area Locality  ({len(localities):,} known in {city})",
            [OTHER_LOCALITY] + localities,
            help="Pick 'Other / not listed' for a locality the model has never "
                 "seen. The estimate then relies on the city and the property "
                 "attributes alone.",
        )
        area_type = st.selectbox(
            cfg.COL_AREA_TYPE, opts[cfg.COL_AREA_TYPE],
            index=(opts[cfg.COL_AREA_TYPE].index("Carpet Area")
                   if "Carpet Area" in opts[cfg.COL_AREA_TYPE] else 0),
            help="Which convention the size figure is measured under. Carpet "
                 "area is the usable floor area; super area includes a share of "
                 "common space.",
        )
        size = st.number_input(
            "Size (sq ft)", min_value=int(ranges[cfg.COL_SIZE]["min"]),
            max_value=int(ranges[cfg.COL_SIZE]["max"]),
            value=int(ranges[cfg.COL_SIZE]["median"]), step=25)

    with c2:
        bhk = st.slider("BHK (bedrooms, hall, kitchen)",
                        int(ranges[cfg.COL_BHK]["min"]),
                        int(ranges[cfg.COL_BHK]["max"]), 2)
        bathroom = st.slider("Bathrooms",
                             int(ranges[cfg.COL_BATHROOM]["min"]),
                             int(ranges[cfg.COL_BATHROOM]["max"]), 2)
        floor_num = st.number_input(
            "Floor number", min_value=int(ranges[cfg.FEAT_FLOOR_NUM]["min"]),
            max_value=int(ranges[cfg.FEAT_FLOOR_NUM]["max"]), value=2, step=1,
            help="0 = ground floor, -1 = lower basement.")
        total_floors = st.number_input(
            "Total floors in the building",
            min_value=int(ranges[cfg.FEAT_TOTAL_FLOORS]["min"]),
            max_value=int(ranges[cfg.FEAT_TOTAL_FLOORS]["max"]), value=5, step=1)

    with c3:
        furnishing = st.selectbox(cfg.COL_FURNISHING, opts[cfg.COL_FURNISHING])
        tenant = st.selectbox(cfg.COL_TENANT, opts[cfg.COL_TENANT])
        contact = st.selectbox(cfg.COL_CONTACT, opts[cfg.COL_CONTACT])

    if total_floors < max(floor_num, 1):
        st.warning("Total floors is lower than the floor number; the pipeline "
                   "will correct this automatically.")

    st.divider()
    if not st.button("Estimate rent", type="primary", width="stretch"):
        st.info("Set the property details above and press **Estimate rent**.")
        return

    # The app's ONLY job: build the input frame with the correct column names.
    row = pd.DataFrame([{
        cfg.COL_CITY: city,
        cfg.COL_LOCALITY: (locality if locality != OTHER_LOCALITY
                           else "__unseen_locality__"),
        cfg.COL_SIZE: float(size),
        cfg.COL_BHK: int(bhk),
        cfg.COL_BATHROOM: int(bathroom),
        cfg.FEAT_FLOOR_NUM: float(floor_num),
        cfg.FEAT_TOTAL_FLOORS: float(total_floors),
        cfg.COL_FURNISHING: furnishing,
        cfg.COL_TENANT: tenant,
        cfg.COL_AREA_TYPE: area_type,
        cfg.COL_CONTACT: contact,
    }])

    try:
        point = float(max(pipeline.predict(row)[0], 0.0))
    except Exception as exc:
        st.error(f"Prediction failed: {type(exc).__name__}: {exc}")
        return

    low, high, interval_source = None, None, None
    q_low_model, q_high_model = load_quantile_models()
    if q_low_model is not None and q_high_model is not None:
        try:
            low = float(max(q_low_model.predict(row)[0], 0.0))
            high = float(max(q_high_model.predict(row)[0], 0.0))
            low, high = min(low, high), max(low, high)
            # Keep the point estimate inside its own interval.
            low, high = min(low, point), max(high, point)
            interval_source = "80% prediction interval (quantile models)"
        except Exception:
            low = high = None
    if low is None:
        mae = float(metrics.get("prediction_interval", {}).get(
            "fallback_mae", best.get("test_mae", 0.0)))
        low, high = max(point - mae, 0.0), point + mae
        interval_source = "± test MAE (quantile models unavailable)"

    st.subheader("Estimate")
    m1, m2, m3 = st.columns([1.2, 1.4, 1])
    m1.metric("Estimated monthly rent", format_inr(point))
    m2.metric("Expected range", f"{format_inr(low)} – {format_inr(high)}")
    m3.metric("Rent per sq ft", f"{point / max(size, 1):.1f}")
    st.caption(interval_source)

    if locality == OTHER_LOCALITY:
        st.warning(
            "This locality is not in the training data, so the locality effect "
            "falls back to the global average. The city, size and other "
            "attributes still drive the estimate, but treat it as a wider "
            "guess than the range above suggests."
        )

    with st.expander("Model behind this estimate", expanded=True):
        a, b, c, d = st.columns(4)
        a.metric("Model", best.get("name", "n/a"))
        b.metric("Test RMSE", format_inr(best.get("test_rmse", 0)))
        c.metric("Test MAE", format_inr(best.get("test_mae", 0)))
        d.metric("Test R²", f"{best.get('test_r2', 0):.4f}")
        st.caption(
            f"Trained under: {best.get('setting_label', 'n/a')}. "
            f"Scores are on a held-out 20% test set of "
            f"{metrics.get('split', {}).get('n_test', 0):,} listings that the "
            f"model never saw during training or tuning."
        )
        # Transposing a one-row frame mixes strings and floats in a single
        # object column, which Arrow cannot serialise for st.dataframe. Cast to
        # string first -- this is a read-only echo of the inputs.
        echo = row.T.rename(columns={row.index[0]: "value"}).astype(str)
        st.dataframe(echo, width="stretch")


# --------------------------------------------------------------------------- #
# Tab 2 -- the analysis behind the model
# --------------------------------------------------------------------------- #
def _df(records, cols=None, rename=None):
    d = pd.DataFrame(records)
    if cols:
        d = d[[c for c in cols if c in d.columns]]
    if rename:
        d = d.rename(columns=rename)
    return d


def render_analysis_tab(metrics: dict) -> None:
    best = metrics.get("best_model", {})

    st.subheader("Model comparison")
    st.caption("All six model variants under the selected training setting, "
               "scored on the held-out test set. CV columns are 5-fold means "
               "± standard deviation across folds.")
    comp = _df(metrics.get("model_comparison", []),
               ["model", "cv_rmse_mean", "cv_rmse_std", "test_r2", "test_mse",
                "test_rmse", "test_mae", "test_mape", "cv_overfit_gap_rmse",
                "predict_latency_ms"],
               {"model": "Model", "cv_rmse_mean": "CV RMSE",
                "cv_rmse_std": "CV RMSE std", "test_r2": "Test R²",
                "test_mse": "Test MSE", "test_rmse": "Test RMSE",
                "test_mae": "Test MAE", "test_mape": "Test MAPE %",
                "cv_overfit_gap_rmse": "CV gap", 
                "predict_latency_ms": "Latency (ms)"})
    st.dataframe(comp.style.format({
        "CV RMSE": "{:,.0f}", "CV RMSE std": "{:,.0f}", "Test R²": "{:.4f}",
        "Test MSE": "{:,.0f}", "Test RMSE": "{:,.0f}", "Test MAE": "{:,.0f}",
        "Test MAPE %": "{:.1f}", "CV gap": "{:,.0f}", "Latency (ms)": "{:.1f}",
    }), width="stretch")
    _show_figure(metrics, "model_rmse_comparison")

    st.divider()
    st.subheader("The outlier experiment")
    st.caption(
        "Every model under four target / outlier settings. Crucially, all of "
        "them are scored on the SAME untreated test set — otherwise removing "
        "outliers would flatter the metrics simply by making the test set "
        "easier. The CV columns are NOT comparable across settings, because "
        "each setting cross-validates a different training set."
    )
    exp = _df(metrics.get("outlier_experiment", []),
              ["setting", "model", "test_rmse", "test_mae", "test_r2",
               "test_mape", "n_train_rows"],
              {"setting": "Setting", "model": "Model", "test_rmse": "Test RMSE",
               "test_mae": "Test MAE", "test_r2": "Test R²",
               "test_mape": "Test MAPE %", "n_train_rows": "Train rows"})
    st.dataframe(exp.style.format({
        "Test RMSE": "{:,.0f}", "Test MAE": "{:,.0f}", "Test R²": "{:.4f}",
        "Test MAPE %": "{:.1f}", "Train rows": "{:,.0f}",
    }), width="stretch", height=420)
    _show_figure(metrics, "outlier_experiment")
    _show_figure(metrics, "outlier_premium_tradeoff")

    st.divider()
    st.subheader("Per-city performance")
    st.caption("Where the model works and where it does not. RMSE as a "
               "percentage of each city's mean rent is the fair comparison, "
               "since rent levels differ roughly fivefold between cities.")
    pc = _df(metrics.get("per_city", []),
             ["City", "n_test", "mean_rent", "rmse", "mae", "r2", "mape",
              "rmse_pct_of_mean", "unseen_locality_pct"],
             {"n_test": "Test listings", "mean_rent": "Mean rent",
              "rmse": "RMSE", "mae": "MAE", "r2": "R²", "mape": "MAPE %",
              "rmse_pct_of_mean": "RMSE % of mean",
              "unseen_locality_pct": "Unseen localities %"})
    st.dataframe(pc.style.format({
        "Mean rent": "{:,.0f}", "RMSE": "{:,.0f}", "MAE": "{:,.0f}",
        "R²": "{:.4f}", "MAPE %": "{:.1f}", "RMSE % of mean": "{:.1f}",
        "Unseen localities %": "{:.1f}",
    }), width="stretch")
    _show_figure(metrics, "per_city_performance")

    st.divider()
    st.subheader("Feature importance")
    st.caption("Permutation importance: how much test MAE rises, in rupees, "
               "when each input column is randomly shuffled. More reliable than "
               "impurity-based importance, which is biased towards "
               "high-cardinality columns such as locality.")
    fi = _df(metrics.get("feature_importance_permutation", []),
             ["feature", "importance_mean", "importance_std", "r2_drop_mean"],
             {"feature": "Feature", "importance_mean": "MAE increase",
              "importance_std": "± std", "r2_drop_mean": "R² drop"})
    st.dataframe(fi.style.format({
        "MAE increase": "{:,.0f}", "± std": "{:,.0f}", "R² drop": "{:.4f}",
    }), width="stretch")
    _show_figure(metrics, "feature_importance")

    st.divider()
    st.subheader("Does locality dominate? (ablation)")
    ab = _df(metrics.get("ablation", []),
             ["label", "test_rmse", "test_mae", "test_r2"],
             {"label": "Model", "test_rmse": "Test RMSE",
              "test_mae": "Test MAE", "test_r2": "Test R²"})
    st.dataframe(ab.style.format({
        "Test RMSE": "{:,.0f}", "Test MAE": "{:,.0f}", "Test R²": "{:.4f}",
    }), width="stretch")
    if metrics.get("ablation_conclusion"):
        st.info(metrics["ablation_conclusion"])
    _show_figure(metrics, "locality_ablation")

    st.divider()
    st.subheader("Residual diagnostics of the selected model")
    _show_figure(metrics, "best_model_residuals")
    _show_figure(metrics, "best_model_residuals_by_city")
    _show_figure(metrics, "residuals_all_models")


def _show_figure(metrics: dict, key: str) -> None:
    rel = metrics.get("figures", {}).get(key)
    if not rel:
        return
    path = cfg.PROJECT_ROOT / rel
    if path.exists():
        st.image(str(path), width="stretch")


# --------------------------------------------------------------------------- #
# Tab 3 -- limitations
# --------------------------------------------------------------------------- #
def render_limitations_tab(metrics: dict) -> None:
    split = metrics.get("split", {})
    prof = metrics.get("profile", {})
    best = metrics.get("best_model", {})
    n_rows = prof.get("n_rows", 4746)
    span = prof.get("posted_on", {}).get("span_days", 89)
    loc = prof.get("locality", {})

    st.subheader("Limitations")
    st.markdown(f"""
**Read every estimate as a starting point for negotiation, not a valuation.**

- **Small dataset.** {n_rows:,} listings in total, and only
  {split.get('n_train', 0):,} used for training. Thin evidence for a problem with
  this much local variation.
- **Thin per-locality coverage.** {loc.get('n_unique', 0):,} distinct localities,
  {loc.get('appearing_once', 0):,} of which appear exactly once
  ({loc.get('share_of_rows_in_singleton_localities', 0):.0%} of all rows). Most
  localities have almost no evidence behind them.
- **Only six cities.** Mumbai, Delhi, Bangalore, Chennai, Hyderabad and Kolkata.
  The model will not generalise to Pune, Ahmedabad or any tier-2 city.
- **A {span}-day snapshot.** All listings were posted within roughly three
  months of 2022, so the model captures no inflation, seasonality or market
  drift, and its absolute rupee levels are already dated.
- **Missing features.** No amenities, building age, parking, metro proximity,
  society quality, security, or view. These genuinely drive rent and their
  absence is the main reason similar-looking flats get the same estimate.
- **Listing price is not negotiated rent.** The target is the asking price, which
  is systematically above what tenants actually pay.
- **Unseen localities fall back to the global average.** A locality absent from
  training contributes nothing specific; only the city and the physical
  attributes carry the estimate.
- **Outlier treatment makes premium predictions unreliable.** Any setting that
  removes or caps extreme rents trains a model that systematically
  under-predicts genuinely expensive properties. See the outlier experiment.
- **Accuracy varies sharply by city.** See the per-city table: typical error is
  around {best.get('test_mape', 0):.0f}% of actual rent overall, but materially
  worse in some cities than others.
""")

    st.subheader("Real-world applicability")
    st.markdown("""
**Who would use it.** Landlords sanity-checking an asking price; tenants
judging whether a listing is priced fairly; a rental platform pre-filling a
suggested rent or flagging suspicious listings.

**How it would be integrated.** As a suggestion inside a listing form — a
suggested range with an explicit confidence band, never a single authoritative
number, and always overridable by the user.

**Retraining cadence.** Monthly at minimum, quarterly at absolute minimum.
Rental markets move faster than this three-month snapshot can represent, and a
model trained on 2022 prices is already stale in rupee terms.

**What data would most improve it.** Amenities and building age; actual
transacted rent rather than asking price; geocoordinates instead of free-text
locality names (which would fix the unseen-locality problem entirely); listing
duration and time-to-let as a demand signal.

**What could go wrong.** A platform that showed these numbers as authoritative
would anchor the market to its own predictions, creating a feedback loop where
the model's errors become next year's training data. Under-prediction in
underserved localities could also entrench existing pricing inequities. Any
deployment needs a human override and monitoring of prediction-versus-actual
drift.
""")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    st.title("Residential Rent Prediction")
    st.caption("Case Study 36 — Semester V Machine Learning. Estimates monthly "
               "rent for a residential unit across six Indian cities.")

    pipeline = load_pipeline()
    metrics = load_json(cfg.METRICS_PATH)
    meta = load_json(cfg.FEATURE_META_PATH)
    if pipeline is None or metrics is None or meta is None:
        missing_artifacts_screen()
        return

    best = metrics.get("best_model", {})
    with st.sidebar:
        st.header("Selected model")
        st.metric(best.get("name", "n/a"), format_inr(best.get("test_rmse", 0)),
                  help="Held-out test RMSE")
        st.write(f"**Test R²** {best.get('test_r2', 0):.4f}")
        st.write(f"**Test MAE** {format_inr(best.get('test_mae', 0))}")
        st.write(f"**Test MAPE** {best.get('test_mape', 0):.1f}%")
        st.caption(f"Trained under {best.get('setting_label', 'n/a')} on "
                   f"{metrics.get('split', {}).get('n_train', 0):,} listings; "
                   f"scored on {metrics.get('split', {}).get('n_test', 0):,} "
                   f"held-out listings.")
        st.divider()
        st.caption(f"scikit-learn {meta.get('sklearn_version', '?')} · "
                   f"random_state {metrics.get('run', {}).get('random_state', 42)}")

    tab1, tab2, tab3 = st.tabs(
        ["Estimate rent", "Model comparison & analysis", "Limitations"])
    with tab1:
        render_predict_tab(pipeline, meta, metrics)
    with tab2:
        render_analysis_tab(metrics)
    with tab3:
        render_limitations_tab(metrics)


main()
