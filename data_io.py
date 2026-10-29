"""Loading, validating and storing data (all local, no HDFS/Spark needed)."""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import config
from calendar_utils import is_planned_holiday


class DataError(Exception):
    """Raised with a user-readable message when data is missing or invalid."""


@dataclass
class DataBundle:
    daily: pd.DataFrame
    zones: pd.DataFrame
    source_name: str
    is_simulated: bool
    notes: list[str] = field(default_factory=list)

    @property
    def tag(self) -> str:
        """Suffix for chart titles / metric labels."""
        return f" [{config.SIMULATED_TAG}]" if self.is_simulated else " [USER DATA - unverified]"

    @property
    def cache_key(self) -> str:
        d = self.daily
        return f"{self.source_name}|{len(d)}|{d['date'].min()}|{d['date'].max()}|{int(d['total_items'].sum())}"


REQUIRED_DAILY_COLS = ["date", "zone_id"] + config.CATEGORY_COLS


# ---------------------------------------------------------------------------
# Validation of daily zone-category data
# ---------------------------------------------------------------------------
def validate_daily(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Validate a daily zone-category table. Returns (clean_df, warnings)."""
    warnings: list[str] = []
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    missing = [c for c in REQUIRED_DAILY_COLS if c not in df.columns]
    if missing:
        raise DataError(f"Missing required column(s): {', '.join(missing)}. "
                        f"Required: {', '.join(REQUIRED_DAILY_COLS)}.")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if df["date"].isna().any():
        raise DataError("Some values in the 'date' column could not be parsed.")
    df["date"] = df["date"].dt.normalize()
    df["zone_id"] = df["zone_id"].astype(str).str.strip()
    for c in config.CATEGORY_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    if df[config.CATEGORY_COLS].isna().any().any():
        raise DataError("Category columns contain missing or non-numeric values.")
    if (df[config.CATEGORY_COLS] < 0).any().any():
        raise DataError("Category columns contain negative values.")
    if df.duplicated(["zone_id", "date"]).any():
        raise DataError("Duplicate (zone_id, date) rows found; expected one row per zone per day.")
    df = df.sort_values(["zone_id", "date"]).reset_index(drop=True)
    df["total_items"] = df[config.CATEGORY_COLS].sum(axis=1)

    # Forecasting needs a gap-free daily series per zone.
    span = df.groupby("zone_id")["date"].agg(["min", "max", "count"])
    expected = (span["max"] - span["min"]).dt.days + 1
    n_missing = int((expected - span["count"]).sum())
    if n_missing > 0:
        raise DataError(f"{n_missing} zone-days are missing (gaps in the daily series). "
                        "Fill or remove gaps; the forecaster needs continuous daily data.")
    if span["count"].min() < 60:
        warnings.append("Some zones have fewer than 60 days; forecasting may be unreliable.")
    if "is_holiday" not in df.columns:
        df["is_holiday"] = is_planned_holiday(df["date"])
    for col in ("recyc_bin_items", "recyc_bin_nonrecyc_items"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
    if "recyc_bin_items" not in df.columns:
        warnings.append("No contamination columns (recyc_bin_items, recyc_bin_nonrecyc_items): "
                        "the contamination rule cannot be evaluated.")
    return df, warnings


def derive_zones(daily: pd.DataFrame, quantile: float = config.CAPACITY_QUANTILE,
                 test_days: int = config.TEST_DAYS) -> pd.DataFrame:
    """Derive a per-zone capacity from the training part of the data (clearly flagged)."""
    rows = []
    for zone, g in daily.groupby("zone_id"):
        g = g.sort_values("date")
        train = g.iloc[:-test_days] if len(g) > test_days + 30 else g
        rows.append({
            "zone_id": zone, "zone_name": str(zone),
            "capacity_items_per_day": int(round(train["total_items"].quantile(quantile))),
            "capacity_basis": f"DERIVED: {int(quantile * 100)}th percentile of training-period volume "
                              "(placeholder - set real municipal capacity)",
            "is_simulated": False,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------
def synthetic_data_exists() -> bool:
    return config.SYNTHETIC_DAILY_PATH.exists() and config.SYNTHETIC_ZONES_PATH.exists()


def load_synthetic() -> DataBundle:
    if not synthetic_data_exists():
        raise DataError("Synthetic data not found. Run `python generate_data.py` "
                        "or use the 'Generate' button in the dashboard.")
    try:
        daily = pd.read_parquet(config.SYNTHETIC_DAILY_PATH)
        zones = pd.read_parquet(config.SYNTHETIC_ZONES_PATH)
    except Exception as exc:  # corrupt file, missing pyarrow, ...
        raise DataError(f"Could not read synthetic Parquet files: {exc}") from exc
    daily, warns = validate_daily(daily)
    return DataBundle(daily, zones, "Synthetic (local Parquet)", True, warns)


def load_uploaded_daily(file_like, name: str) -> DataBundle:
    """Load a user-supplied daily file (CSV or Parquet) in the documented schema."""
    try:
        if str(name).lower().endswith(".parquet"):
            raw = pd.read_parquet(file_like)
        else:
            raw = pd.read_csv(file_like)
    except Exception as exc:
        raise DataError(f"Could not read '{name}': {exc}") from exc
    daily, warns = validate_daily(raw)
    zones = derive_zones(daily)
    warns.append("Capacity per zone was DERIVED from the data (placeholder).")
    return DataBundle(daily, zones, f"Uploaded: {name}", False, warns)


def load_from_records_store() -> DataBundle:
    """Aggregate locally stored classified-image records into daily counts."""
    rec = load_records()
    if rec.empty:
        raise DataError("The local classified-records store is empty. Classify images and save "
                        "records on the 'Classify Image' page first.")
    daily = aggregate_records_to_daily(rec)
    daily, warns = validate_daily(daily)
    zones = derive_zones(daily)
    warns.append("Built from locally stored classified records; capacity DERIVED (placeholder).")
    return DataBundle(daily, zones, "Local classified-records store", False, warns)


# ---------------------------------------------------------------------------
# Classified-record store (Module 4, local Parquet equivalent of the HDFS append)
# Record = <id, category, confidence, zone_id, timestamp>  (Eq. 4.1)
# Only metadata is stored - never the raw image (NFR4 privacy).
# ---------------------------------------------------------------------------
RECORD_COLUMNS = ["id", "category", "confidence", "zone_id", "timestamp"]


def load_records(path: Path | None = None) -> pd.DataFrame:
    path = Path(path) if path else config.CLASSIFIED_RECORDS_PATH
    if not path.exists():
        return pd.DataFrame(columns=RECORD_COLUMNS)
    try:
        return pd.read_parquet(path)
    except Exception as exc:
        raise DataError(f"Could not read records store {path}: {exc}") from exc


def append_record(category: str, confidence: float, zone_id: str,
                  timestamp: datetime | None = None, path: Path | None = None) -> str:
    """Append an ACCEPTED classification record; returns the record id."""
    path = Path(path) if path else config.CLASSIFIED_RECORDS_PATH
    ts = timestamp or datetime.now(timezone.utc).replace(tzinfo=None)
    rec_id = uuid.uuid4().hex
    new = pd.DataFrame([{"id": rec_id, "category": str(category), "confidence": float(confidence),
                         "zone_id": str(zone_id), "timestamp": pd.Timestamp(ts)}])
    old = load_records(path)
    out = new if old.empty else pd.concat([old, new], ignore_index=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(path, index=False)
    return rec_id


def append_review_item(category: str | None, confidence: float | None, zone_id: str,
                       reason: str, path: Path | None = None) -> None:
    """Add a below-threshold prediction to the manual review queue (metadata only)."""
    path = Path(path) if path else config.REVIEW_QUEUE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    row = pd.DataFrame([{
        "id": uuid.uuid4().hex, "timestamp": datetime.now(timezone.utc).replace(tzinfo=None).isoformat(),
        "zone_id": zone_id, "top_category": category, "top_confidence": confidence, "reason": reason}])
    row.to_csv(path, mode="a", header=not path.exists(), index=False)


def aggregate_records_to_daily(records: pd.DataFrame, min_confidence: float = config.CONFIDENCE_THRESHOLD
                               ) -> pd.DataFrame:
    """Eq. 4.2: count accepted records per zone, day and category (wide format)."""
    if records.empty:
        return pd.DataFrame(columns=REQUIRED_DAILY_COLS)
    r = records.copy()
    r = r[r["confidence"] >= min_confidence]
    r["date"] = pd.to_datetime(r["timestamp"]).dt.normalize()
    r["category"] = r["category"].astype(str).str.lower()
    wide = (r.groupby(["zone_id", "date", "category"]).size().unstack("category", fill_value=0))
    for c in config.CATEGORY_COLS:
        if c not in wide.columns:
            wide[c] = 0
    wide = wide[config.CATEGORY_COLS].reset_index()
    # fill missing days so each zone has a continuous series
    full = []
    for zone, g in wide.groupby("zone_id"):
        idx = pd.date_range(g["date"].min(), g["date"].max(), freq="D")
        g = g.set_index("date")[config.CATEGORY_COLS].reindex(idx, fill_value=0)
        g["zone_id"] = zone
        full.append(g.rename_axis("date").reset_index())
    return pd.concat(full, ignore_index=True)[REQUIRED_DAILY_COLS]


def load_bundle(source: str, uploaded=None, uploaded_name: str = "") -> DataBundle:
    """Dispatch on the data-source choice made in the dashboard sidebar."""
    if source == "synthetic":
        return load_synthetic()
    if source == "records":
        return load_from_records_store()
    if source == "upload":
        if uploaded is None:
            raise DataError("Upload a daily CSV/Parquet file (schema in README) to use this source.")
        return load_uploaded_daily(uploaded, uploaded_name)
    raise DataError(f"Unknown data source: {source}")
