"""Central configuration for Waste Intelligence.

All paths are relative to the project folder (no absolute paths), and every
default below is traceable to Table 3 / Table 4.1 of the project paper/report
unless it is explicitly marked "IMPLEMENTATION CHOICE".
"""
from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

# Folders can be redirected with environment variables (used by the tests).
DATA_DIR = Path(os.environ.get("WASTE_DATA_DIR", PROJECT_ROOT / "data"))
MODELS_DIR = Path(os.environ.get("WASTE_MODELS_DIR", PROJECT_ROOT / "models"))
OUTPUTS_DIR = Path(os.environ.get("WASTE_OUTPUTS_DIR", PROJECT_ROOT / "outputs"))

SYNTHETIC_DAILY_PATH = DATA_DIR / "synthetic_daily.parquet"
SYNTHETIC_ZONES_PATH = DATA_DIR / "synthetic_zones.parquet"
SYNTHETIC_DAILY_CSV = DATA_DIR / "synthetic_daily.csv"
CLASSIFIED_RECORDS_PATH = DATA_DIR / "classified_records.parquet"
REVIEW_QUEUE_PATH = OUTPUTS_DIR / "review_queue.csv"

# Image classifier files (you place these yourself - see models/README.md)
MODEL_PATH = MODELS_DIR / "waste_classifier.keras"
CLASS_NAMES_PATH = MODELS_DIR / "class_names.json"

# ---------------------------------------------------------------------------
# Waste categories (C = 6)
# ---------------------------------------------------------------------------
CATEGORIES = ["Paper", "Plastic", "Glass", "Metal", "Organic", "Other"]
CATEGORY_COLS = [c.lower() for c in CATEGORIES]          # column names in data
RECYCLABLE_CATEGORIES = ["Paper", "Plastic", "Glass", "Metal"]
RECYCLABLE_COLS = [c.lower() for c in RECYCLABLE_CATEGORIES]

# ---------------------------------------------------------------------------
# Image classification (Table 4.1: 224x224x3, MobileNetV2, p_min = 0.80)
# ---------------------------------------------------------------------------
IMAGE_SIZE = (224, 224)
CONFIDENCE_THRESHOLD = 0.80
MAX_UPLOAD_MB = 10
MIN_IMAGE_SIDE_PX = 32
ALLOWED_IMAGE_FORMATS = ("JPEG", "PNG", "WEBP", "BMP")
# Must match how YOU trained the model. Options:
#   "mobilenet_v2" -> x / 127.5 - 1   (same as keras mobilenet_v2.preprocess_input)
#   "rescale_0_1"  -> x / 255
#   "none"         -> raw 0-255 floats (use when the saved model already contains
#                     its own Rescaling / preprocess_input layer)
PREPROCESS_MODE = "none"
# The paper lists "denoise" in preprocessing. It is OFF by default because a
# denoise step at inference time that was not used in training can reduce
# accuracy. Turn on only if you also used it when training.
APPLY_DENOISE = False
OUTPUT_SUM_TOLERANCE = 1e-3          # softmax outputs must sum to 1 +/- this

# ---------------------------------------------------------------------------
# Analytics defaults (Table 4.1)
# ---------------------------------------------------------------------------
RANDOM_SEED = 42
K_MIN, K_MAX = 2, 10                 # K searched over 2-10
SILHOUETTE_SAMPLE = 5000             # IMPLEMENTATION CHOICE: sampled silhouette for speed
RF_TREES = 100
RF_MAX_FEATURES = 1 / 3              # one third of predictors per split
LAGS = (1, 7, 14)                    # lagged volumes (days before forecast origin)
HORIZON = 7                          # seven-day-ahead forecast
TEST_DAYS = 183                      # last 183 target days form the held-out test set

# ---------------------------------------------------------------------------
# Decision rules (Table 4.1: theta_z = municipal capacity, rho_max = 0.15)
# ---------------------------------------------------------------------------
CONTAMINATION_MAX = 0.15
CONTAMINATION_WINDOW_DAYS = 28       # IMPLEMENTATION CHOICE: trailing window for rho
CAPACITY_QUANTILE = 0.90             # paper's *illustrative* choice: 90th pct of training volume
CAPACITY_MULTIPLIER = 1.0
TRIGGER_MARGIN = 1.0                 # 1.0 = nominal trigger; 0.95 / 0.90 = earlier warning
DOMINANT_RECYCLABLE_MIN_SHARE = 0.20  # IMPLEMENTATION CHOICE: paper does not define "dominant"

# Illustrative pickup templates (NOT real municipal schedules)
PICKUP_TEMPLATES = {
    "Paper": "Weekly paper/cardboard pickup",
    "Plastic": "Twice-weekly plastics pickup",
    "Glass": "Fortnightly glass pickup",
    "Metal": "Fortnightly metal pickup",
}

# Planned (announced) holiday calendar, month-day. Used as the "holiday flag".
# SIMULATED calendar - replace with your own if you use real data.
PLANNED_HOLIDAYS_MMDD = ("01-01", "01-26", "08-15", "10-02", "12-25")

# ---------------------------------------------------------------------------
# Synthetic data generator defaults (24 zones x 730 days)
# ---------------------------------------------------------------------------
N_ZONES = 24
N_DAYS = 730
START_DATE = "2024-01-01"

SIMULATED_TAG = "SIMULATED"

# ---------------------------------------------------------------------------
# Optional integrations (never required)
# ---------------------------------------------------------------------------
HDFS_URL_ENV = "WASTE_HDFS_URL"      # e.g. http://namenode:9870  (WebHDFS)
HDFS_USER_ENV = "WASTE_HDFS_USER"
HDFS_REPLICATION = 3                 # Table 4.1: Parquet, replication factor 3
