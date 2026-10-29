# Waste Intelligence

Local, runnable implementation of **"AI-Driven Intelligent Waste Classification and Big Data Analytics
Framework for Smart Waste Management"** (Sathyabama Institute of Science and Technology, P2BL report
and conference-style paper, Nov 2026).

> **Read this first**
> * The dashboard works **without** a trained CNN, GPU, Docker, Spark or HDFS.
> * **No image model is included, trained or claimed.** You train MobileNetV2 yourself (see
>   [Your trained model](#your-trained-model)). Until then the *Classify Image* page shows only a clear
>   "model missing" message - never a made-up prediction.
> * The bundled data is **SIMULATED**. Every chart, table and metric derived from it is labelled
>   `[SIMULATED]`. Nothing here is real city data or real model performance, and the numbers will **not**
>   reproduce the paper's simulation tables exactly (different generator details and random draws).
> * Analytics run on one machine with Pandas / scikit-learn. **No distributed processing is running**,
>   even if you later install the optional Spark/HDFS packages.

---

## Windows setup (VS Code / Google Antigravity terminal, CPU only)

Open a terminal in the extracted `waste-intelligence` folder (PowerShell or Command Prompt):

```powershell
py -3 -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe generate_data.py
.venv\Scripts\python.exe -m streamlit run app.py
```

Streamlit prints a local URL (normally http://localhost:8501) and opens your browser.
(Commands call the venv's Python directly, so you do not need to "activate" it or change PowerShell
execution policy.) Shortcut: double-click `setup_windows.bat` once, then `run_app.bat`.

Use Python 3.10-3.12 (`py -3 --version`). If the page says data is missing, click **Generate synthetic
data now** in the app instead of running `generate_data.py`.

Run the automated tests (optional):

```powershell
.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.venv\Scripts\python.exe -m pytest -q
```

First visit to the **7-Day Forecast** page trains models (about 1 minute on a laptop CPU); results are cached afterwards.

---

## Your trained model

Place two files in `models/` (details in `models/README.md`):

| File | Content |
|------|---------|
| `models/waste_classifier.keras` | your trained Keras model: MobileNetV2, 224x224x3 input, **6-way softmax** output |
| `models/class_names.json` | `["Paper", "Plastic", "Glass", "Metal", "Organic", "Other"]` in your model's output order |

Then install TensorFlow (optional dependency) and restart:

```powershell
.venv\Scripts\python.exe -m pip install -r requirements-ml.txt
```

Behaviour of the classifier page:

* A category and confidence appear **only** when the model loads, its output is a valid probability vector
  (right shape, finite, in [0, 1], sums to 1) **and** confidence >= threshold (default **0.80**, adjustable in the sidebar).
* Below the threshold: a "manual review" message; the top class is shown as **not accepted** and can be queued
  (metadata only) in `outputs/review_queue.csv`.
* Missing model / class names / TensorFlow, invalid or oversized images, wrong output shape: each gives a specific message.
* **Preprocessing must match your training.** Default is MobileNetV2 scaling (`x/127.5 - 1`, bilinear resize to 224x224).
  Change `PREPROCESS_MODE` in `config.py` if you trained differently, or use `"none"` when the saved model already
  contains its own preprocessing layer. The paper's optional "denoise" step is off by default (`APPLY_DENOISE`).
* Accepted records (category, confidence, zone, timestamp - **never the image**) can be saved to
  `data/classified_records.parquet`, the local stand-in for the paper's HDFS append (Eq. 4.1).
  The sidebar data source *Local classified-records store* aggregates them (Eq. 4.2) into the same analytics.

---

## Project layout

```
app.py                  Streamlit dashboard (7 pages + sidebar configuration)
analytics.py            aggregation, composition, K-Means, Random Forest forecasting, evaluation
generate_data.py        reproducible SIMULATED generator (24 zones x 730 days) + CLI
image_classifier.py     validation, preprocessing, model/class-name loading, thresholding (no training)
recommendations.py      three transparent decision rules + contamination metric
data_io.py              loading/validation, classified-record store, daily aggregation of records
config.py               all parameters (Table 4.1 defaults), relative paths
calendar_utils.py       planned-holiday flag
integrations/           OPTIONAL spark_processor.py, hdfs_storage.py
tests/                  pytest suite
data/  models/  outputs/
requirements*.txt  setup_windows.bat  run_app.bat
```

## Requirement -> implementation map

Sources: PBL report (Chapters 3-4) and paper (Sections IV-VI), Table 4.1 / Table 3 parameters.

| Requirement (document reference) | Status | Where |
|---|---|---|
| Six categories: paper, plastic, glass, metal, organic, other (C = 6) | Done | `config.CATEGORIES` |
| FR1 / Module 1 - accept camera or mobile upload, tag with zone and timestamp | Done | `app.py` (upload + camera tabs, zone select); `data_io.append_record` |
| FR2 / Module 2 - resize 224x224, denoise, normalise | Done (denoise optional, off) | `image_classifier.validate_image`, `preprocess_image` |
| FR3 / Module 3 - CNN (MobileNetV2) with confidence, route < 0.80 to manual review | Interface done; **model is yours** | `image_classifier.ClassifierService`; review queue in `data_io.append_review_item` |
| CNN training (Adam, lr 1e-4, batch 32, 30 epochs) | **Not implemented by design** - you train | - |
| FR4 / Module 4 - append records <id, c, p, z, tau> in Parquet | Local Parquet done; HDFS **pending/untested** | `data_io.append_record`; `integrations/hdfs_storage.py` |
| Eq. 4.2 - daily zone-level volume per category | Done | `analytics.aggregate_period`, `daily_zone_category_long`, `data_io.aggregate_records_to_daily` |
| FR5 / Module 5 - Spark aggregation | Module provided, **not run on a real Spark**; not wired into the dashboard | `integrations/spark_processor.py` |
| Module 6 - K-Means, 7 standardised features (volume + 6 shares), K = 2-10, silhouette + elbow | Done (scikit-learn) | `analytics.fit_clustering`, `k_search`, `elbow_k`, page 4 |
| Spark MLlib K-Means / RF at cluster scale | **PENDING** | - |
| Module 7 - Random Forest, T = 100, 1/3 of predictors, lags 1/7/14, horizon 7, cluster label + previous shares as predictors | Done | `analytics.build_forecast_frame`, `evaluate_forecasters`, `forecast_future` |
| Chronological train/test split (last 183 days test); cluster model fit on training rows only | Done | `analytics.chronological_split`, `evaluate_forecasters` |
| Benchmarks: persistence, 7-day moving average, ridge, gradient boosting (200 trees, depth 3); MAE, RMSE, R2, MAPE | Done | `analytics.evaluate_forecasters` |
| Overflow-warning precision/recall at trigger 1.0 / 0.95 / 0.90 x capacity | Done | `analytics.overflow_metrics` |
| Module 8 - dashboard: cluster view, observed vs forecast series, recommendation list | Done | `app.py` |
| Module 9 / Rule 1: forecast > theta_z -> extra capacity | Done | `recommendations.capacity_rule` |
| Rule 2: rho > rho_max (0.15) -> awareness / signage review | Done | `recommendations.contamination_rule` |
| Rule 3: dominant recyclable category in cluster centroid -> pickup schedule | Done (threshold is an implementation choice) | `recommendations.schedule_rule` |
| Calibrating rho with classifier sensitivity/specificity (report Sec. 5 / 6.3) | Optional, off by default | `recommendations.calibrate_contamination` |
| Synthetic simulation: 24 zones, 730 days, 4 archetypes, weekly/annual/trend/holiday/noise, unannounced 5-day festival in the test period | Done | `generate_data.py` |
| Sensitivity of downstream stages to classifier accuracy (report Table 5.5) | **PENDING** (not implemented) | - |
| Real image-stream validation, IoT dustbins, streaming ingestion, YOLO, route optimisation, quantile forecasts, event calendar | **PENDING** (paper's future work) | - |
| Privacy NFR4: keep only category, confidence, zone, timestamp | Done (images never stored) | `data_io.append_record` |

### How rho (contamination) is calculated

`rho(z) = items in recyclable-designated bins labelled non-recyclable / all items in recyclable-designated bins`,
summed over a trailing window (default 28 days, `config.CONTAMINATION_WINDOW_DAYS`) for zone `z`, using columns
`recyc_bin_nonrecyc_items` and `recyc_bin_items`. In the simulated data both counts are generated
synthetically (35% of items go to recyclable bins; 5 of 24 zones get a true rate of 0.20-0.30, the rest 0.05-0.12).
Rule 2 fires when `rho > 0.15` (strictly greater).

### Implementation choices not specified by the documents (change in `config.py`)

* Glass/Other split the share left over after the four categories listed in the paper's Table 4 (55% / 45%).
* "Lags 1, 7, 14" = volumes 1, 7 and 14 days before the forecast origin, plus the latest observed volume `V_t`.
* Capacity `theta_z` in simulated data = 90th percentile of each zone's training-period volume (the paper's illustrative choice).
* "Dominant recyclable" = largest recyclable share in the cluster centroid, at least 20%.
* Silhouette is computed on a 5,000-point sample when there are more points.
* The dashboard clusters one row per zone by default (K searched 2-10); the paper's zone-day clustering is a sidebar option.
  On the simulated data, silhouette and elbow may favour different K - the app shows both and lets you choose.
* Pickup-schedule texts are illustrative templates, not municipal schedules.

---

## Data

Default: `python generate_data.py` -> `data/synthetic_daily.parquet`, `data/synthetic_zones.parquet`
(`--csv` also writes a CSV; `--seed`, `--zones`, `--days` configurable). All rows carry `is_simulated = True`.

You may upload your own daily file in the sidebar (CSV/Parquet):
`date, zone_id, paper, plastic, glass, metal, organic, other` (+ optional `recyc_bin_items`, `recyc_bin_nonrecyc_items`),
one row per zone per day, no gaps. Zone capacity is then *derived* from the data as a placeholder - set your
real capacity before using any capacity recommendation.
Image-derived counts measure items presented to the system, not total waste generated; calibrate against weighed
collection data before real planning (paper, Sec. VIII).

## Enabling the optional integrations later

Nothing below is required. The dashboard never calls these automatically.

```powershell
.venv\Scripts\python.exe -m pip install -r requirements-optional.txt   # pyspark + hdfs
```

* **Spark** needs Java 11/17 on PATH. `integrations/spark_processor.py` aggregates an item-level records Parquet
  file into daily counts via `aggregate_daily_counts(path)`. Untested against a real Spark install by the package author.
  Some Windows Spark operations need `HADOOP_HOME`/winutils.
* **HDFS** uses WebHDFS (`hdfs` package). Set `WASTE_HDFS_URL` (e.g. `http://namenode:9870`) and optionally
  `WASTE_HDFS_USER`, then use `HdfsStorage().write_parquet(df, "/waste/records.parquet")` (replication 3).
  Tested only with a mocked client.
* Missing Java/PySpark/hdfs only produces a status message on page 7; the app keeps running.

## Tests and verification performed before packaging

`pytest` (49 tests): generator reproducibility, aggregation/composition, K search validity, clustering,
no look-ahead / chronological split, forecasting pipeline, rule thresholds, contamination metric, image validation and
threshold logic (with a stub model that exists only inside the tests), optional-integration fallbacks, and a headless
run of every dashboard page with no model/Spark/HDFS. Run on Python 3.12 with pandas 3.0 and again with pandas 2.2.
**Not tested:** Windows itself, TensorFlow model loading (no model exists), real Spark or HDFS.
