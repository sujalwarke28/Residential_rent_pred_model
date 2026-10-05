"""Central configuration: paths, column schema and reproducibility settings.

Everything that the notebook, `train.py` and the Streamlit app need to agree on
is defined exactly once, here.
"""

from pathlib import Path

# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #
# A single global seed, threaded through every train/test split, every
# RandomizedSearchCV, every cross-validation shuffle and every model that
# accepts a random_state. Re-running train.py reproduces identical numbers.
RANDOM_STATE = 42

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

RAW_CSV = DATA_DIR / "House_Rent_Dataset.csv"
GLOSSARY = DATA_DIR / "Dataset Glossary.txt"

PIPELINE_PATH = MODELS_DIR / "pipeline.joblib"
QUANTILE_LOW_PATH = MODELS_DIR / "quantile_low.joblib"
QUANTILE_HIGH_PATH = MODELS_DIR / "quantile_high.joblib"
METRICS_PATH = MODELS_DIR / "metrics.json"
FEATURE_META_PATH = MODELS_DIR / "feature_meta.json"

MODEL_COMPARISON_CSV = REPORTS_DIR / "model_comparison.csv"
OUTLIER_EXPERIMENT_CSV = REPORTS_DIR / "outlier_experiment.csv"
PER_CITY_CSV = REPORTS_DIR / "per_city_performance.csv"
ABLATION_CSV = REPORTS_DIR / "locality_ablation.csv"
FEATURE_IMPORTANCE_CSV = REPORTS_DIR / "feature_importance.csv"


def ensure_dirs() -> None:
    """Create every output directory that the project writes into."""
    for d in (DATA_DIR, MODELS_DIR, REPORTS_DIR, FIGURES_DIR):
        d.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# Column schema
# --------------------------------------------------------------------------- #
# NOTE: these spellings were verified against the actual CSV header rather than
# assumed -- the spaces and capitalisation matter because the ColumnTransformer
# selects columns by exact name.
TARGET = "Rent"

COL_POSTED_ON = "Posted On"
COL_BHK = "BHK"
COL_SIZE = "Size"
COL_FLOOR = "Floor"
COL_AREA_TYPE = "Area Type"
COL_LOCALITY = "Area Locality"
COL_CITY = "City"
COL_FURNISHING = "Furnishing Status"
COL_TENANT = "Tenant Preferred"
COL_BATHROOM = "Bathroom"
COL_CONTACT = "Point of Contact"

EXPECTED_RAW_COLUMNS = [
    COL_POSTED_ON, COL_BHK, TARGET, COL_SIZE, COL_FLOOR, COL_AREA_TYPE,
    COL_LOCALITY, COL_CITY, COL_FURNISHING, COL_TENANT, COL_BATHROOM,
    COL_CONTACT,
]

# `Posted On` is dropped after profiling. The listings span only 2022-04-13 to
# 2022-07-11 (~90 days), far too short a window to carry inflation or market
# drift, so any date-derived feature would be noise. See notebook section 2.
DROP_COLUMNS = [COL_POSTED_ON]

# --------------------------------------------------------------------------- #
# Engineered feature names (produced by preprocess.FeatureEngineer)
# --------------------------------------------------------------------------- #
FEAT_FLOOR_NUM = "floor_num"
FEAT_TOTAL_FLOORS = "total_floors"
FEAT_FLOOR_RATIO = "floor_ratio"
FEAT_CARPET_EQUIV = "carpet_equiv_size"
FEAT_RENT_PER_SQFT = "rent_per_sqft"  # EDA / outlier detection ONLY -- never a feature

# --------------------------------------------------------------------------- #
# Area Type handling
# --------------------------------------------------------------------------- #
# `Size` is reported under three different conventions and is NOT directly
# comparable across them. Conventional Indian real-estate ratios, expressed as
# a multiplier that converts the reported figure into a carpet-area equivalent:
#     carpet area  ~= 1.00 x carpet area      (already carpet)
#     carpet area  ~= 0.85 x built-up area
#     carpet area  ~= 0.70 x super built-up area
# Section 3 of the notebook tests these ratios empirically against the data.
AREA_TYPES = ["Carpet Area", "Built Area", "Super Area"]
AREA_TYPE_RATIOS = {
    "Carpet Area": 1.00,
    "Built Area": 0.85,
    "Super Area": 0.70,
}


def interaction_feature_name(area_type: str) -> str:
    """Stable column name for the `Size` x `Area Type` interaction term."""
    return "size_x_" + area_type.lower().replace(" ", "_")


# Size x Area Type interaction, used by the Linear and Polynomial models only.
# Tree ensembles discover this split on their own; a linear model cannot express
# "a square foot means something different depending on the measurement basis"
# without an explicit product term, and is unfairly penalised without it.
INTERACTION_FEATURES = [interaction_feature_name(a) for a in AREA_TYPES]

# --------------------------------------------------------------------------- #
# Feature groups consumed by the ColumnTransformer
# --------------------------------------------------------------------------- #
NUMERIC_FEATURES = [
    COL_BHK,
    COL_SIZE,            # raw reported size, kept alongside the carpet equivalent
    COL_BATHROOM,
    FEAT_FLOOR_NUM,
    FEAT_TOTAL_FLOORS,
    FEAT_FLOOR_RATIO,
    FEAT_CARPET_EQUIV,
]

ONEHOT_FEATURES = [
    COL_CITY,
    COL_FURNISHING,
    COL_TENANT,
    COL_AREA_TYPE,
    COL_CONTACT,
]

# ~2,235 distinct localities, 1,465 of which appear exactly once. One-hot
# encoding would add thousands of near-empty columns and overfit badly, so this
# column is target-encoded with smoothing instead.
TARGET_ENCODE_FEATURES = [COL_LOCALITY]

# Columns the Streamlit app must supply when building its one-row input frame.
APP_INPUT_COLUMNS = [
    COL_CITY, COL_LOCALITY, COL_SIZE, COL_BHK, COL_BATHROOM,
    FEAT_FLOOR_NUM, FEAT_TOTAL_FLOORS, COL_FURNISHING, COL_TENANT,
    COL_AREA_TYPE, COL_CONTACT,
]

# --------------------------------------------------------------------------- #
# Cleaning thresholds
# --------------------------------------------------------------------------- #
# 16 listings report a Size below 50 sq ft (the smallest is 10). These are data
# entry errors, not micro-studios: they are treated as missing and imputed.
MIN_VALID_SIZE = 50.0

# Floor tokens that are words rather than numbers.
FLOOR_TOKEN_MAP = {
    "ground": 0.0,
    "upper basement": -0.5,
    "lower basement": -1.0,
}

# --------------------------------------------------------------------------- #
# Splitting / cross-validation
# --------------------------------------------------------------------------- #
TEST_SIZE = 0.20
N_SPLITS = 5          # 5-fold CV on the 80% training portion
STRATIFY_ON = COL_CITY  # keeps all 6 cities proportionally represented

# --------------------------------------------------------------------------- #
# Outlier treatment
# --------------------------------------------------------------------------- #
IQR_MULTIPLIER = 1.5
WINSOR_LOWER_PCT = 0.01
WINSOR_UPPER_PCT = 0.99
# Rent-per-sqft values beyond this quantile are flagged as implausible in
# addition to the per-city Rent IQR rule.
RENT_PER_SQFT_UPPER_Q = 0.995
RENT_PER_SQFT_LOWER_Q = 0.005

# --------------------------------------------------------------------------- #
# Prediction interval
# --------------------------------------------------------------------------- #
QUANTILE_LOW = 0.10
QUANTILE_HIGH = 0.90   # -> a proper 80% prediction interval
