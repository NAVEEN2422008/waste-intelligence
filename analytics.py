"""Analytics layer (local Pandas / scikit-learn implementation).

Modules 5-7 of the paper:
  * daily aggregation and composition (Eq. 4.2)
  * standardised K-Means clustering, K searched 2-10 with silhouette (Eq. 4.5-4.6)
  * Random Forest seven-day-ahead forecasting (Eq. 4.7) with a chronological split

This runs on one machine with scikit-learn (the paper's simulation did the same).
It is NOT distributed Spark MLlib processing - see integrations/spark_processor.py.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import adjusted_rand_score, mean_absolute_error, mean_squared_error, r2_score, silhouette_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import config
from calendar_utils import is_planned_holiday

CAT_COLS = config.CATEGORY_COLS
SHARE_COLS = [f"share_{c}" for c in CAT_COLS]
CLUSTER_FEATURES = ["total_items"] + SHARE_COLS          # d = 1 + C = 7


# ---------------------------------------------------------------------------
# Aggregation and composition
# ---------------------------------------------------------------------------
def ensure_total(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["total_items"] = out[CAT_COLS].sum(axis=1)
    return out


def aggregate_period(df: pd.DataFrame, freq: str = "D") -> pd.DataFrame:
    """Aggregate to zone x period (D, W, MS) category counts."""
    d = ensure_total(df)
    if freq == "D":
        return d[["date", "zone_id"] + CAT_COLS + ["total_items"]].copy()
    g = d.groupby(["zone_id", pd.Grouper(key="date", freq=freq)])[CAT_COLS + ["total_items"]].sum()
    return g.reset_index()


def daily_zone_category_long(df: pd.DataFrame) -> pd.DataFrame:
    """Long format: date, zone_id, category, items (daily waste volume by zone and category)."""
    long = df.melt(id_vars=["date", "zone_id"], value_vars=CAT_COLS, var_name="category", value_name="items")
    long["category"] = long["category"].str.title()
    return long


def composition(df: pd.DataFrame, by: str | None = "zone_id") -> pd.DataFrame:
    """Six-category composition: item totals and shares (share columns sum to 1)."""
    d = ensure_total(df)
    g = d.groupby(by)[CAT_COLS + ["total_items"]].sum() if by else d[CAT_COLS + ["total_items"]].sum().to_frame().T
    for c in CAT_COLS:
        g[f"share_{c}"] = g[c] / g["total_items"].replace(0, np.nan)
    return g.reset_index() if by else g.reset_index(drop=True)


def weekday_profile(df: pd.DataFrame) -> pd.DataFrame:
    """Average items/day by weekday and category."""
    d = df.copy()
    d["weekday"] = d["date"].dt.dayofweek
    prof = d.groupby("weekday")[CAT_COLS].mean()
    prof.index = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][: len(prof)]
    prof.columns = [c.title() for c in prof.columns]
    return prof


def zone_day_features(df: pd.DataFrame) -> pd.DataFrame:
    """Eq. 4.5 feature vector per zone-day: total volume + six category shares."""
    d = ensure_total(df)
    out = d[["zone_id", "date", "total_items"]].copy()
    for c in CAT_COLS:
        out[f"share_{c}"] = d[c] / d["total_items"].replace(0, np.nan)
    return out.fillna(0.0)


# ---------------------------------------------------------------------------
# K-Means clustering
# ---------------------------------------------------------------------------
def _silhouette(X: np.ndarray, labels: np.ndarray, sample: int, seed: int) -> float:
    n_labels = len(np.unique(labels))
    if n_labels < 2 or n_labels >= len(X):
        return float("nan")
    size = sample if len(X) > sample else None
    return float(silhouette_score(X, labels, sample_size=size, random_state=seed))


def k_search(X: np.ndarray, k_min: int = config.K_MIN, k_max: int = config.K_MAX,
             seed: int = config.RANDOM_SEED, sample: int = config.SILHOUETTE_SAMPLE) -> pd.DataFrame:
    """Fit K-Means for each K and report inertia and silhouette (NaN where not valid)."""
    rows = []
    for k in range(k_min, k_max + 1):
        if k >= len(X):
            rows.append({"k": k, "inertia": np.nan, "silhouette": np.nan, "note": "K >= number of samples"})
            continue
        km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(X)
        sil = _silhouette(X, km.labels_, sample, seed)
        rows.append({"k": k, "inertia": float(km.inertia_), "silhouette": sil,
                     "note": "" if not np.isnan(sil) else "silhouette not valid"})
    return pd.DataFrame(rows)


def elbow_k(k_table: pd.DataFrame) -> int | None:
    """Elbow = point farthest from the straight line joining the first and last inertia values."""
    t = k_table.dropna(subset=["inertia"])
    if len(t) < 3:
        return None
    x = t["k"].to_numpy(float)
    y = t["inertia"].to_numpy(float)
    xn = (x - x.min()) / (x.max() - x.min())
    yn = (y - y.min()) / (y.max() - y.min() + 1e-12)
    dist = np.abs((yn[-1] - yn[0]) * xn - (xn[-1] - xn[0]) * yn + xn[-1] * yn[0] - yn[-1] * xn[0])
    return int(x[int(np.argmax(dist))])


def describe_centroids(centroids: pd.DataFrame) -> list[str]:
    """Plain-language centroid descriptions such as 'High volume, plastic-heavy'."""
    k = len(centroids)
    order = centroids["items_per_day"].rank(method="first").to_numpy() - 1
    frac = order / max(k - 1, 1)
    out = []
    for i in range(k):
        if k == 1:
            vol = "Typical volume"
        elif k == 2:
            vol = "High volume" if frac[i] > 0.5 else "Low volume"
        elif k == 3:
            vol = ["Low volume", "Mid volume", "High volume"][int(order[i])]
        else:
            vol = ("Low volume" if frac[i] < 0.25 else "Mid volume" if frac[i] < 0.5
                   else "High volume" if frac[i] < 0.8 else "Very high volume")
        shares = centroids.loc[centroids.index[i], [f"share_{c}" for c in CAT_COLS]].to_numpy(float)
        top = np.argsort(shares)[::-1]
        a, b = CAT_COLS[top[0]], CAT_COLS[top[1]]
        if shares[top[0]] >= 0.40:
            comp = f"{a}-dominant"
        elif shares[top[1]] >= 0.20:
            comp = f"{a}/{b}-heavy"
        elif shares[top[0]] >= 0.28:
            comp = f"{a}-heavy"
        else:
            comp = "mixed composition"
        out.append(f"{vol}, {comp}")
    return out


@dataclass
class ClusteringResult:
    level: str
    k_table: pd.DataFrame
    best_k: int | None            # K with the highest valid silhouette
    elbow_k: int | None
    chosen_k: int
    features: pd.DataFrame        # input rows with a 'cluster' column
    centroids: pd.DataFrame       # in ORIGINAL units (items/day and shares), with descriptions
    zone_assignment: pd.DataFrame
    scaler: StandardScaler
    model: KMeans
    feature_cols: list[str] = field(default_factory=lambda: list(CLUSTER_FEATURES))
    ari_vs_simulation: float | None = None


def fit_clustering(features: pd.DataFrame, level: str = "zone_day", k: int | None = None,
                   k_min: int = config.K_MIN, k_max: int = config.K_MAX,
                   seed: int = config.RANDOM_SEED, sample: int = config.SILHOUETTE_SAMPLE
                   ) -> ClusteringResult:
    """Standardise the 7 features, search K, fit K-Means and describe the centroids."""
    if len(features) < 3:
        raise ValueError("Need at least 3 samples to cluster.")
    k_max = min(k_max, len(features) - 1)
    if k_min > k_max:
        raise ValueError("K range is invalid for the amount of data available.")
    scaler = StandardScaler()
    X = scaler.fit_transform(features[CLUSTER_FEATURES].to_numpy(float))
    table = k_search(X, k_min, k_max, seed, sample)
    valid = table.dropna(subset=["silhouette"])
    best = int(valid.loc[valid["silhouette"].idxmax(), "k"]) if not valid.empty else None
    chosen = int(k) if k is not None else (best if best is not None else k_min)
    chosen = min(max(chosen, 1), len(features) - 1)
    km = KMeans(n_clusters=chosen, n_init=10, random_state=seed).fit(X)

    feats = features.copy()
    feats["cluster"] = km.labels_
    cent = pd.DataFrame(scaler.inverse_transform(km.cluster_centers_), columns=CLUSTER_FEATURES)
    cent = cent.rename(columns={"total_items": "items_per_day"})
    cent.insert(0, "cluster", range(chosen))
    cent["n_samples"] = np.bincount(km.labels_, minlength=chosen)
    cent["description"] = describe_centroids(cent)

    zone_assign = pd.DataFrame()
    if "zone_id" in feats.columns:
        rows = []
        for zone, g in feats.groupby("zone_id"):
            counts = g["cluster"].value_counts()
            rows.append({"zone_id": zone, "cluster": int(counts.index[0]),
                         "purity": float(counts.iloc[0] / len(g))})
        zone_assign = pd.DataFrame(rows)
        zone_assign["description"] = zone_assign["cluster"].map(dict(zip(cent["cluster"], cent["description"])))
    return ClusteringResult(level, table, best, elbow_k(table), chosen, feats, cent, zone_assign, scaler, km)


def cluster_zones(df: pd.DataFrame, level: str = "zone_day", k: int | None = None,
                  k_min: int = config.K_MIN, k_max: int = config.K_MAX, seed: int = config.RANDOM_SEED,
                  zones: pd.DataFrame | None = None) -> ClusteringResult:
    """Cluster zones. level='zone_day' (paper: zone-time windows) or 'zone' (one row per zone)."""
    zd = zone_day_features(df)
    if level == "zone":
        feats = zd.groupby("zone_id")[CLUSTER_FEATURES].mean().reset_index()
    elif level == "zone_day":
        feats = zd
    else:
        raise ValueError("level must be 'zone_day' or 'zone'")
    res = fit_clustering(feats, level, k, k_min, k_max, seed)
    if level == "zone":
        res.zone_assignment = res.features[["zone_id", "cluster"]].copy()
        res.zone_assignment["purity"] = 1.0
        res.zone_assignment["description"] = res.zone_assignment["cluster"].map(
            dict(zip(res.centroids["cluster"], res.centroids["description"])))
    if zones is not None and "sim_archetype" in zones.columns and not res.zone_assignment.empty:
        merged = res.zone_assignment.merge(zones[["zone_id", "sim_archetype"]], on="zone_id")
        if merged["sim_archetype"].nunique() > 1:
            res.ari_vs_simulation = float(adjusted_rand_score(merged["sim_archetype"], merged["cluster"]))
    return res


def assign_clusters(res: ClusteringResult, features: pd.DataFrame) -> np.ndarray:
    X = res.scaler.transform(features[res.feature_cols].to_numpy(float))
    return res.model.predict(X)


# ---------------------------------------------------------------------------
# Forecasting (Random Forest, seven days ahead)
# ---------------------------------------------------------------------------
def regression_metrics(y: np.ndarray, yhat: np.ndarray) -> dict[str, float]:
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    nz = y != 0
    return {
        "MAE": float(mean_absolute_error(y, yhat)),
        "RMSE": float(np.sqrt(mean_squared_error(y, yhat))),
        "R2": float(r2_score(y, yhat)) if len(y) > 1 else float("nan"),
        "MAPE_%": float(np.mean(np.abs((y[nz] - yhat[nz]) / y[nz])) * 100) if nz.any() else float("nan"),
        "Bias": float(np.mean(yhat - y)),
    }


def predictor_columns(lags: tuple[int, ...] = config.LAGS) -> list[str]:
    return (["vol_t"] + [f"vol_lag_{k}" for k in lags] + [f"share_prev_{c}" for c in CAT_COLS]
            + ["dow", "month", "is_holiday", "cluster"])


def build_forecast_frame(df: pd.DataFrame, horizon: int = config.HORIZON,
                         lags: tuple[int, ...] = config.LAGS, for_future: bool = False) -> pd.DataFrame:
    """Build one row per (zone, forecast origin t) predicting volume at t + horizon.

    Predictors use ONLY information available at the origin day t (volumes up to t,
    category shares of day t) plus calendar facts about the target day (day of week,
    month, planned-holiday flag). With for_future=True only each zone's last day is
    kept (target lies in the future, so no label exists).
    """
    d = ensure_total(df).sort_values(["zone_id", "date"]).reset_index(drop=True)
    g = d.groupby("zone_id", sort=False)["total_items"]
    out = pd.DataFrame({"zone_id": d["zone_id"], "origin_date": d["date"], "vol_t": d["total_items"]})
    for k in lags:
        out[f"vol_lag_{k}"] = g.shift(k)
    out["ma7"] = g.transform(lambda s: s.rolling(7).mean())
    for c in CAT_COLS:
        out[f"share_prev_{c}"] = d[c] / d["total_items"].replace(0, np.nan)
    out["target_date"] = out["origin_date"] + pd.Timedelta(days=horizon)
    out["dow"] = out["target_date"].dt.dayofweek
    out["month"] = out["target_date"].dt.month
    out["is_holiday"] = is_planned_holiday(out["target_date"]).astype(int)
    out["y"] = g.shift(-horizon)
    if "sim_unannounced_event" in d.columns:
        ev = d.set_index(["zone_id", "date"])["sim_unannounced_event"]
        key = pd.MultiIndex.from_arrays([out["zone_id"], out["target_date"]])
        out["event_target"] = ev.reindex(key).eq(True).to_numpy()
    needed = [f"vol_lag_{k}" for k in lags] + ["ma7"] + [f"share_prev_{c}" for c in CAT_COLS]
    out = out.dropna(subset=needed)
    if for_future:
        last = out["origin_date"].max()
        return out[out["origin_date"] == last].reset_index(drop=True)
    return out.dropna(subset=["y"]).reset_index(drop=True)


def _cluster_feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    f = pd.DataFrame({"total_items": frame["vol_t"]})
    for c in CAT_COLS:
        f[f"share_{c}"] = frame[f"share_prev_{c}"]
    return f


def _onehot_design(frame: pd.DataFrame, cols: list[str], all_clusters: list[int]) -> np.ndarray:
    base = frame[[c for c in cols if c not in ("dow", "cluster")]].to_numpy(float)
    dow = np.eye(7)[frame["dow"].to_numpy(int)]
    cl = np.stack([(frame["cluster"].to_numpy(int) == k) for k in all_clusters], axis=1).astype(float)
    return np.hstack([base, dow, cl])


def chronological_split(frame: pd.DataFrame, test_days: int = config.TEST_DAYS
                        ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    """Split by TARGET date: train targets <= cutoff < test targets. Raises if leakage is possible."""
    max_target = frame["target_date"].max()
    cutoff = max_target - pd.Timedelta(days=test_days)
    train = frame[frame["target_date"] <= cutoff]
    test = frame[frame["target_date"] > cutoff]
    if len(train) < 100 or len(test) < 10:
        raise ValueError("Not enough data for a chronological train/test split "
                         f"(train={len(train)}, test={len(test)}). Use a longer history or fewer test days.")
    assert train["target_date"].max() < test["target_date"].min()
    return train, test, cutoff


@dataclass
class ForecastEvaluation:
    horizon: int
    cutoff: pd.Timestamp
    n_train: int
    n_test: int
    metrics: pd.DataFrame            # per method
    predictions: pd.DataFrame        # held-out test predictions (origin_date, target_date, ...)
    importances: pd.DataFrame
    overflow: pd.DataFrame
    cluster_k: int
    has_event_flag: bool


METHOD_LABELS = {"persistence": "Persistence (V_t)", "ma7": "7-day moving average",
                 "ridge": "Ridge regression", "gb": "Gradient boosting (200 trees, depth 3)",
                 "rf": "Random Forest"}


def overflow_metrics(pred: pd.DataFrame, capacity: pd.Series, margins=(1.0, 0.95, 0.90),
                     methods=("persistence", "ridge", "gb", "rf")) -> pd.DataFrame:
    """Precision/recall/F1 of 'forecast > margin x capacity' against actual > capacity."""
    cap = pred["zone_id"].map(capacity).to_numpy(float)
    actual = pred["actual"].to_numpy(float) > cap
    rows = []
    for m in methods:
        for mg in margins:
            if m != "rf" and mg != 1.0:
                continue
            flagged = pred[m].to_numpy(float) > mg * cap
            tp = int(np.sum(flagged & actual))
            prec = tp / flagged.sum() if flagged.sum() else float("nan")
            rec = tp / actual.sum() if actual.sum() else float("nan")
            f1 = (2 * prec * rec / (prec + rec)) if prec == prec and rec == rec and (prec + rec) > 0 else float("nan")
            rows.append({"method": METHOD_LABELS[m], "trigger": f"{mg:g} x capacity", "precision": prec,
                         "recall": rec, "F1": f1, "actual_overflows": int(actual.sum()),
                         "flagged": int(flagged.sum())})
    return pd.DataFrame(rows)


def evaluate_forecasters(df: pd.DataFrame, zones: pd.DataFrame, horizon: int = config.HORIZON,
                         test_days: int = config.TEST_DAYS, n_estimators: int = config.RF_TREES,
                         seed: int = config.RANDOM_SEED, lags: tuple[int, ...] = config.LAGS,
                         k_min: int = config.K_MIN, k_max: int = config.K_MAX) -> ForecastEvaluation:
    """Backtest on a held-out, chronologically LATER test period. Models never see test targets."""
    frame = build_forecast_frame(df, horizon, lags)
    train, test, cutoff = chronological_split(frame, test_days)
    train, test = train.copy(), test.copy()

    # Cluster label as predictor: clustering is fitted on TRAINING rows only (no leakage).
    cl_feats = _cluster_feature_frame(train)
    cl_feats["zone_id"] = train["zone_id"].to_numpy()
    cres = fit_clustering(cl_feats, "zone_day", None, k_min, k_max, seed)
    train["cluster"] = assign_clusters(cres, cl_feats)
    test["cluster"] = assign_clusters(cres, _cluster_feature_frame(test))

    cols = predictor_columns(lags)
    rf = RandomForestRegressor(n_estimators=n_estimators, max_features=config.RF_MAX_FEATURES,
                               n_jobs=-1, random_state=seed)
    rf.fit(train[cols].to_numpy(float), train["y"].to_numpy(float))
    clusters = list(range(cres.chosen_k))
    ridge = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    ridge.fit(_onehot_design(train, cols, clusters), train["y"].to_numpy(float))
    gb = GradientBoostingRegressor(n_estimators=200, max_depth=3, random_state=seed)
    gb.fit(train[cols].to_numpy(float), train["y"].to_numpy(float))

    pred = test[["zone_id", "origin_date", "target_date"]].copy()
    pred["actual"] = test["y"].to_numpy(float)
    pred["persistence"] = test["vol_t"].to_numpy(float)
    pred["ma7"] = test["ma7"].to_numpy(float)
    pred["ridge"] = ridge.predict(_onehot_design(test, cols, clusters))
    pred["gb"] = gb.predict(test[cols].to_numpy(float))
    pred["rf"] = rf.predict(test[cols].to_numpy(float))
    has_event = "event_target" in test.columns
    if has_event:
        pred["event_target"] = test["event_target"].to_numpy()

    rows = []
    base_mae = regression_metrics(pred["actual"], pred["persistence"])["MAE"]
    for m, label in METHOD_LABELS.items():
        row = {"method": label, **regression_metrics(pred["actual"], pred[m])}
        row["MAE_vs_persistence_%"] = (1 - row["MAE"] / base_mae) * 100
        if has_event:
            normal = ~pred["event_target"].to_numpy()
            if normal.any():
                row["MAE_normal_days"] = regression_metrics(pred.loc[normal, "actual"], pred.loc[normal, m])["MAE"]
            if (~normal).any():
                row["MAE_event_days"] = regression_metrics(pred.loc[~normal, "actual"], pred.loc[~normal, m])["MAE"]
        rows.append(row)
    metrics = pd.DataFrame(rows)

    imp = pd.DataFrame({"predictor": cols, "importance": rf.feature_importances_}).sort_values(
        "importance", ascending=False).reset_index(drop=True)
    cap = zones.set_index("zone_id")["capacity_items_per_day"]
    return ForecastEvaluation(horizon, cutoff, len(train), len(test), metrics, pred, imp,
                              overflow_metrics(pred, cap), cres.chosen_k, has_event)


def forecast_future(df: pd.DataFrame, horizons: tuple[int, ...] = tuple(range(1, config.HORIZON + 1)),
                    n_estimators: int = config.RF_TREES, seed: int = config.RANDOM_SEED,
                    lags: tuple[int, ...] = config.LAGS, k_min: int = config.K_MIN,
                    k_max: int = config.K_MAX) -> pd.DataFrame:
    """Refit on ALL available history and forecast volumes for days after the last observed day.

    One direct Random Forest per horizon h = 1..7 (each trained on rows whose target h days
    ahead is already observed). Returned rows are FORECASTS, never observations.
    """
    zd = zone_day_features(df)
    cres = fit_clustering(zd, "zone_day", None, k_min, k_max, seed)
    cols = predictor_columns(lags)
    out = []
    for h in horizons:
        train = build_forecast_frame(df, h, lags)
        fut = build_forecast_frame(df, h, lags, for_future=True)
        if len(train) < 50 or fut.empty:
            raise ValueError("Not enough history to forecast.")
        train = train.copy()
        fut = fut.copy()
        train["cluster"] = assign_clusters(cres, _cluster_feature_frame(train))
        fut["cluster"] = assign_clusters(cres, _cluster_feature_frame(fut))
        rf = RandomForestRegressor(n_estimators=n_estimators, max_features=config.RF_MAX_FEATURES,
                                   n_jobs=-1, random_state=seed)
        rf.fit(train[cols].to_numpy(float), train["y"].to_numpy(float))
        res = fut[["zone_id", "origin_date", "target_date"]].copy()
        res["horizon"] = h
        res["forecast"] = np.maximum(rf.predict(fut[cols].to_numpy(float)), 0.0)
        out.append(res)
    return pd.concat(out, ignore_index=True)
